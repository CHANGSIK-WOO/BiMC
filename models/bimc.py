import torch
import torch.nn as nn
import torch.nn.functional as F
import models.clip.clip as clip
import json
import os
import numpy as np
import torchvision.utils as vutils
import torchvision
from tqdm import tqdm


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

def get_unique_path(root_dir):
    os.makedirs(root_dir, exist_ok=True)
    idx = 0
    while True:
        path = os.path.join(root_dir, f"{idx}.png")
        if not os.path.exists(path):
            return path, idx
        idx += 1

def denormalize(x):
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]
    mean = torch.tensor(mean, device=x.device).view(3, 1, 1)
    std = torch.tensor(std, device=x.device).view(3, 1, 1)
    return x * std + mean


class BiMC(nn.Module):

    def __init__(self, cfg, template, device):
        super(BiMC, self).__init__()
        self.cfg = cfg
        self.device = device
        self.task_id = 0
        self.building = True
        self.meta_training = False
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        print(f"Prompt template:{template}")
        self.template = template
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.BiMC.PREC == "fp32" or cfg.TRAINER.BiMC.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        clip_model.eval()
        self.clip_model = clip_model.to(self.device)
        self.text_proto = None
        self.description_proto = None
        self.vision_proto = None

        # Method selection: bimc, bimc_ensemble, edge
        self.method = cfg.TRAINER.BiMC.get('METHOD', 'bimc')
        print(f"Method: {self.method}")

        # EDGE-specific parameters
        if 'edge' in self.method:
            self.edge = True
            edge_cfg = cfg.TRAINER.BiMC.get('EDGE', {})
            self.gamma = edge_cfg.get('GAMMA', 0.5)
            self.inference_edge = edge_cfg.get('INFERENCE_EDGE', False)
            self.edge_sigma = edge_cfg.get('SIGMA', 1.0)  # Default sigma for Gaussian blur
            print(f"EDGE parameters - gamma: {self.gamma}, sigma: {self.edge_sigma}, inference_edge: {self.inference_edge}")
        else:
            self.edge = False
            self.gamma = None
            self.inference_edge = False
            self.edge_sigma = 1.0

        # Meta-learning: Learnable Prompts
        self.use_prompts = False
        self.prompt_pool = None  # Dictionary to store per-task prompts
        if 'prompt' in self.method:
            meta_cfg = cfg.TRAINER.BiMC.META
            prompt_length = meta_cfg.PROMPT_LENGTH
            prompt_dim = meta_cfg.PROMPT_DIM

            # Initialize prompt pool (will be populated during meta-training)
            self.prompt_pool = {}
            self.prompt_length = prompt_length
            self.prompt_dim = prompt_dim

            print(f"Meta-learning enabled: Learnable Prompts initialized")
            print(f"Prompt length: {prompt_length}, Prompt dim: {prompt_dim}")

        # Logging
        self.save_imag = False
        self.output_dir = 'outputs'
        self.save_class = [40, 52, 250]

    @torch.no_grad()
    def inference_text_feature(self, class_names, template, cls_begin_index):
        print(f'class names: {class_names}')
        clip_weights = []
        all_targets = []
        k = cls_begin_index
        for classname in class_names:
            targets = torch.full((len(template),), k)
            all_targets.append(targets)
            k += 1
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            classname = classname.replace('-', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = self.clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)
        clip_weights = torch.stack(clip_weights, dim=0)
        clip_weights = F.normalize(clip_weights, dim=-1)
        all_targets = torch.cat(all_targets, dim=0)
        return clip_weights, all_targets

    @torch.no_grad()
    def inference_all_img_feature(self, loader, cls_begin_index):
        all_features = []
        all_labels = []

        # === EDGE: Invariant feature collector ===
        all_edge_features = []

        for batch in tqdm(loader, desc="Extracting image features", leave=False):
            images, labels = self.parse_batch(batch)

            # ---- Original CLIP features (with prompt if available) ----
            features = self.clip_model.encode_image(images, use_prompt=True)
            features = F.normalize(features, dim=-1)
            all_features.append(features)
            all_labels.append(labels)

            # ---- EDGE: Invariant features (LoG edge extraction, NO PROMPT) ----
            if self.edge:
                # Always use fixed parameters (no router) and NO prompt
                edge_feat = self._extract_edge_features(images, labels)
                all_edge_features.append(edge_feat)

        # ---- merge ----
        all_features = torch.cat(all_features, dim=0)
        if self.edge and len(all_edge_features) > 0:
            all_edge_features = torch.cat(all_edge_features, dim=0)
        else:
            all_edge_features = None
        all_labels = torch.cat(all_labels, dim=0)

        unique_labels = torch.unique(all_labels)
        print(f'all targets:{unique_labels}')

        # ---- Original vision prototype ----
        prototypes = []
        for c in unique_labels:
            idx = torch.where(c == all_labels)[0]
            class_features = all_features[idx]
            class_prototype = class_features.mean(dim=0)
            prototypes.append(class_prototype)
        prototypes = torch.stack(prototypes, dim=0)
        prototypes = F.normalize(prototypes, dim=-1)

        # ---- EDGE: Edge prototype ----
        if self.edge and all_edge_features is not None:
            edge_prototypes = []
            for c in unique_labels:
                idx = torch.where(c == all_labels)[0]
                class_inv_features = all_edge_features[idx]
                inv_proto = class_inv_features.mean(dim=0)
                edge_prototypes.append(inv_proto)
            edge_prototypes = torch.stack(edge_prototypes, dim=0)
            edge_prototypes = F.normalize(edge_prototypes, dim=-1)
        else:
            edge_prototypes = None

        # ---- return all ----
        return {
            "targets": all_labels,
            "orig_features": all_features,
            "orig_proto": prototypes,
            "edge_features": all_edge_features,
            "edge_proto": edge_prototypes
        }

    @torch.no_grad()
    def _extract_edge_features(self, images, labels):
        """
        Generate domain-invariant EDGE structural features using
        Laplacian of Gaussian (LoG) edge extraction with FIXED hyperparameters.

        Args:
            images: Input images (B, 3, H, W)
            labels: Labels (B,)
        """

        # Use fixed sigma from config
        sigma = self.edge_sigma
        laplacian_kernel = None

        # === Gaussian smoothing ===
        # Note: torchvision GaussianBlur doesn't support batched sigma
        # So we use the aggregated sigma value
        if isinstance(sigma, torch.Tensor):
            sigma = sigma.item()

        blur = torchvision.transforms.GaussianBlur(kernel_size=5, sigma=sigma)
        img_blur = blur(images)

        # Convert to grayscale
        gray = img_blur.mean(dim=1, keepdim=True)  # (B,1,H,W)

        # === Laplacian kernel ===
        if laplacian_kernel is None:
            # Default Laplacian kernel
            laplacian_kernel = torch.tensor(
                [[
                    [0, 1, 0],
                    [1, -4, 1],
                    [0, 1, 0]
                ]],
                dtype=gray.dtype,
                device=gray.device
            ).unsqueeze(0)  # (1,1,3,3)

        # === LoG edge ===
        edge = F.conv2d(gray, laplacian_kernel, padding=1).abs()

        # Normalize
        max_val = edge.amax(dim=[1, 2, 3], keepdim=True).clamp(min=1e-6)
        edge = edge / max_val

        # For CLIP: 3 channels needed
        edge_img = edge.repeat(1, 3, 1, 1)

        # === Optional image saving ===
        if not self.meta_training and self.save_imag:
            for i in range(images.size(0)):
                label = labels[i].item()
                if label not in self.save_class:
                    continue

                vis_dir = 'vis_train' if self.building else 'vis_test'
                # Save original (denormalized)
                orig_path, idx = get_unique_path(f"{self.output_dir}/{self.edge_sigma}/{vis_dir}/origin/{self.task_id}/{label}")
                if idx < 100:
                    vutils.save_image(denormalize(images[i]).clamp(0, 1), orig_path)

                # Save invariant LoG edge
                edge_path, idx = get_unique_path(f"{self.output_dir}/{self.edge_sigma}/{vis_dir}/edge/{self.task_id}/{label}")
                if idx < 100:
                    vutils.save_image(edge_img[i].clamp(0, 1), edge_path)
        # if not self.meta_training:
        #     print(router_params)
        # === Encode with CLIP (NO PROMPT for edge features) ===
        inv_feat = self.clip_model.encode_image(edge_img, use_prompt=False)
        inv_feat = F.normalize(inv_feat, dim=-1)

        return inv_feat

    @torch.no_grad()
    def inference_all_description_feature(self, class_names, gpt_path, cls_begin_index):
        description_embeddings = []
        mean_embeddings = []
        all_targets = []
        file = open(gpt_path, "r")
        GPT_prompt_dict = json.load(file)
        # The order of embeddings should follow strictly order of classname variable
        # Keys name should match classnames so that we could do fetching from the dict.
        # Convert the dict to lower case
        GPT_prompt_dict = {k.lower().replace("_", " "): v for k, v in GPT_prompt_dict.items()}
        k = cls_begin_index
        for single_key in class_names:
            single_class_prompts = GPT_prompt_dict[single_key.lower().replace("_", " ")]
            targets = torch.full((len(single_class_prompts),), k)

            k += 1
            x_tokenized = torch.cat([clip.tokenize(p) for p in single_class_prompts])
            with torch.no_grad():
                text_features = self.clip_model.encode_text(x_tokenized.cuda())
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            mean_embeddings.append(text_features.mean(0).unsqueeze(0))
            description_embeddings.append(text_features)
            all_targets.append(targets)
        description_embeddings = torch.cat(description_embeddings, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        mean_embeddings = torch.cat(mean_embeddings, dim=0)
        mean_embeddings = F.normalize(mean_embeddings, dim=-1)
        return description_embeddings, all_targets, mean_embeddings

    def soft_calibration(self, base_protos, cur_protos):
        shift_weight = self.cfg.TRAINER.BiMC.LAMBDA_I
        tau = self.cfg.TRAINER.BiMC.TAU
        base_protos = F.normalize(base_protos, p=2, dim=-1)
        cur_protos = F.normalize(cur_protos, p=2, dim=-1)
        weights = torch.mm(cur_protos, base_protos.T) * tau
        norm_weights = torch.softmax(weights, dim=1)
        delta_protos = torch.matmul(norm_weights, base_protos)
        delta_protos = F.normalize(delta_protos, p=2, dim=-1)
        updated_protos = (1 - shift_weight) * cur_protos + shift_weight * delta_protos
        updated_protos = F.normalize(updated_protos, dim=-1)
        return updated_protos

    def build_task_statistics(self, class_names, loader,
                              class_index, calibrate_novel_vision_proto=False):

        def shrink_cov(cov, alpha1=1.0, alpha2=0.0):
            diag_mean = torch.mean(torch.diagonal(cov))
            off_diag = cov.clone()
            off_diag.fill_diagonal_(0.0)
            mask = off_diag != 0.0
            off_diag_mean = (off_diag * mask).sum() / mask.sum()
            iden = torch.eye(cov.shape[0]).to(cov.device)
            cov_ = cov + (alpha1 * diag_mean * iden) + (alpha2 * off_diag_mean * (1 - iden))
            return cov_

        cls_begin_index = class_index[0]

        text_features, text_targets = self.inference_text_feature(class_names, self.template, cls_begin_index)

        description_features, description_targets, description_proto = \
            self.inference_all_description_feature(class_names=class_names,
                                                   gpt_path=self.cfg.DATASET.GPT_PATH,
                                                   cls_begin_index=cls_begin_index)

        img_stats = self.inference_all_img_feature(loader, cls_begin_index)
        images_targets = img_stats["targets"]
        images_features = img_stats["orig_features"]
        images_proto = img_stats["orig_proto"]
        edge_features = img_stats["edge_features"]
        edge_proto = img_stats["edge_proto"]

        if cls_begin_index != 0:
            if calibrate_novel_vision_proto:
                print(f'calibrate vision proto on class [{class_index}]')
                images_proto = self.soft_calibration(self.base_vision_prototype, images_proto)
        else:
            self.base_vision_prototype = images_proto

        cov_images = torch.cov(images_features.T)

        if cls_begin_index == 0:
            cov_images = shrink_cov(cov_images, alpha1=self.cfg.TRAINER.BiMC.GAMMA_BASE)
        else:
            cov_images = shrink_cov(cov_images, alpha1=self.cfg.TRAINER.BiMC.GAMMA_INC)

        print('finish loading covariance')

        return {
            'description_proto': description_proto,
            'description_features': description_features,
            'description_targets': description_targets,

            'text_features': text_features,
            'text_targets': text_targets,

            'image_proto': images_proto,
            'images_features': images_features,
            'images_targets': images_targets,
            'cov_image': cov_images,

            'class_index': class_index,
            'sample_cnt': len(images_features),

            'edge_proto': edge_proto,
        }

    def forward_ours(self, images, labels, num_cls, num_base_cls,
                     image_proto, cov_image,
                     description_proto,
                     description_features, description_targets,
                     text_features,
                     edge_proto,
                     beta):

        def knn_similarity_scores(queries, support_features, support_labels):
            """
            Compute the similarity between each query sample and all support samples,
            and retrieve the maximum score for each class per query.
            """
            # Ensure all inputs are on the same device
            device = queries.device
            support_features = support_features.to(device)
            support_labels = support_labels.to(device)
            similarity_scores = torch.matmul(queries, support_features.T)
            k = torch.max(support_labels) + 1
            max_scores = torch.full((queries.size(0), k), float('-inf'), device=device)
            expanded_labels = support_labels.unsqueeze(0).expand(queries.size(0), -1)
            for label in range(k):
                label_mask = (expanded_labels == label)
                masked_scores = similarity_scores.masked_fill(~label_mask, float('-inf'))
                max_scores[:, label] = torch.max(masked_scores, dim=1).values
            return max_scores

        def _mahalanobis(dist, cov_inv):
            """
            Compute the Mahalanobis distance between feature vectors and a class prototype.
            """
            left_term = torch.matmul(dist, cov_inv)
            mahal = torch.matmul(left_term, dist.T)
            return torch.diag(mahal)

        def _cov_forward(feat, proto, cov):
            """
            Perform a forward pass computing negative Mahalanobis distance between
            features and each class prototype using a shared covariance matrix.
            """
            maha_dist = []
            inv_covmat = torch.pinverse(cov.to(dtype=torch.float32))
            inv_covmat = inv_covmat.to(dtype=proto.dtype)
            for cl in range(num_cls):
                distance = feat - proto[cl]
                dist = _mahalanobis(distance, inv_covmat)
                maha_dist.append(dist)
            maha_dist = torch.stack(maha_dist)
            logits = -maha_dist.T
            return logits

        # Normalize the image features
        img_feat = self.extract_img_feature(images)
        img_feat = F.normalize(img_feat, dim=-1)

        if self.cfg.TRAINER.BiMC.TEXT_CALIBRATION:
            lambda_t = self.cfg.TRAINER.BiMC.LAMBDA_T
        else:
            lambda_t = 0.0

        # Here we compute the classifier after modality calibration.
        # Note that image_proto has already been calibrated in the `build_task_statistics` function.
        # --- fused prototype computation ---
        fused_proto = (
                beta * ((1 - lambda_t) * text_features + lambda_t * description_proto)
                + (1 - beta) * image_proto
        )

        # ============================================================
        # EDGE: Domain-invariant feature fusion (fixed gamma)
        # ============================================================
        if self.edge:
            gamma = self.gamma
            if self.inference_edge:
                edge_feat = self._extract_edge_features(images, labels)
                edge_feat = F.normalize(edge_feat, dim=-1)
                img_feat = (1 - gamma) * img_feat + gamma * edge_feat
                img_feat = F.normalize(img_feat, dim=-1)

            fused_proto = (1 - gamma) * fused_proto + gamma * edge_proto
        # ============================================================

        # normalize prototype
        fused_proto = F.normalize(fused_proto, dim=-1)
        logits_proto_fused = img_feat @ fused_proto.t()
        prob_fused_proto = F.softmax(logits_proto_fused, dim=-1)

        logits_cov = _cov_forward(img_feat, image_proto, cov_image)
        logits_knn = knn_similarity_scores(img_feat, description_features, description_targets)
        prob_cov = F.softmax(logits_cov / 512, dim=-1)
        prob_knn = F.softmax(logits_knn, dim=-1)

        NUM_BASE_CLS = num_base_cls
        use_diversity = self.cfg.TRAINER.BiMC.USING_ENSEMBLE
        if use_diversity:
            ensemble_alpha = self.cfg.DATASET.ENSEMBLE_ALPHA
        else:
            ensemble_alpha = 1.0

        base_probs = ensemble_alpha * prob_fused_proto[:, :NUM_BASE_CLS] + (1 - ensemble_alpha) * prob_cov[:, :NUM_BASE_CLS]
        inc_probs = ensemble_alpha * prob_fused_proto[:, NUM_BASE_CLS:] + (1 - ensemble_alpha) * prob_knn[:, NUM_BASE_CLS:]

        prob_fused = torch.cat([base_probs, inc_probs], dim=1)
        logits = prob_fused
        return logits

    def extract_img_feature(self, images, use_prompt=True):
        """
        Extract image features using CLIP.

        Args:
            images: Input images
            use_prompt: Whether to use prompt (default: True)

        Returns:
            image_features: CLIP image embeddings
        """
        images = images.to(self.device)
        image_features = self.clip_model.encode_image(images, use_prompt=use_prompt)
        return image_features

    @torch.no_grad()
    def forward(self, images):
        img_feat = self.extract_img_feature(images)
        img_feat = F.normalize(img_feat, dim=-1)
        classifier = F.normalize(self.classifier_weights, dim=-1)
        logits = 100. * img_feat @ classifier.t()
        return logits

    def parse_batch(self, batch):
        data = batch['image']
        targets = batch['label']
        data = data.to(self.device)
        targets = targets.to(self.device)
        return data, targets

    # ======================================================
    # Meta-Learning: Learnable Prompt Methods
    # ======================================================

    def create_prompt(self, task_id=None):
        """
        Create a new learnable prompt for a task.

        Args:
            task_id: Task ID (optional, for naming)

        Returns:
            prompt: nn.Parameter (prompt_length, prompt_dim)
        """
        # Initialize prompt with small random values
        scale = self.prompt_dim ** -0.5
        prompt = nn.Parameter(scale * torch.randn(self.prompt_length, self.prompt_dim, device=self.device))

        if task_id is not None:
            print(f"[Prompt] Created new prompt for task {task_id}: shape {prompt.shape}")

        return prompt

    def set_prompt(self, prompt):
        """
        Set the current prompt for visual encoder.

        Args:
            prompt: Tensor (prompt_length, prompt_dim) or (B, prompt_length, prompt_dim)
        """
        # Set prompt in the CLIP visual encoder
        # Use object.__setattr__ to avoid PyTorch trying to register it as a parameter
        if hasattr(self.clip_model, 'visual'):
            object.__setattr__(self.clip_model.visual, 'current_prompt', prompt)
        else:
            print("[Warning] CLIP model does not have visual encoder with prompt support")

    def clear_prompt(self):
        """Clear the current prompt from visual encoder."""
        if hasattr(self.clip_model, 'visual'):
            object.__setattr__(self.clip_model.visual, 'current_prompt', None)

    def freeze_all_except_prompt(self, prompt):
        """
        Freeze all parameters except the given prompt.

        Args:
            prompt: nn.Parameter to keep trainable
        """
        # Freeze CLIP model
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Unfreeze prompt
        prompt.requires_grad = True

        print("[BiMC] Froze all parameters except prompt")

    def enable_prompts(self):
        """Enable prompts for inference."""
        self.use_prompts = True
        print("[BiMC] Prompts enabled for inference")

    def disable_prompts(self):
        """Disable prompts for inference."""
        self.use_prompts = False
        self.clear_prompt()
        print("[BiMC] Prompts disabled for inference")