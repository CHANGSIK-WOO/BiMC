import torch
import torch.nn as nn
import models.clip as clip
from datasets.data_manager import DatasetManager
from torch.nn import functional as F
from tqdm import tqdm
from utils.evaluator import AccuracyEvaluator
from models.bimc import BiMC
import numpy as np
import time
import os, json


class Runner:

    def __init__(self, cfg):
        self.cfg = cfg

        # DomainNet이면 DGFSCILDataManager 사용, 아니면 기존 DatasetManager 사용
        if cfg.DATASET.NAME.lower() == 'domainnet':
            from datasets.data_manager_dgfscil import DGFSCILDataManager
            self.data_manager = DGFSCILDataManager(cfg)
            self.is_dgfscil = True
            print("[Runner] Using DGFSCILDataManager for DomainNet")
        else:
            self.data_manager = DatasetManager(cfg)
            self.is_dgfscil = False
            print("[Runner] Using DatasetManager")

        self.device = cfg.DEVICE.DEVICE_NAME

        self.model = BiMC(cfg, self.data_manager.template, self.device)

        # device
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
            self.is_distributed = True
        else:
            self.is_distributed = False

        self.acc_list = []
        self.task_acc_list = []
        self.evaluator = AccuracyEvaluator(self.data_manager.class_index_in_task)

    def _apply_hyperparameters(self, hyperparam_dict):
        """Apply hyperparameters to the model dynamically"""
        print(f"Applying hyperparameters: {hyperparam_dict}")

        # For edge method - sigma and edge_mix_weight
        if 'sigma' in hyperparam_dict:
            # Update sigma in the model (for Gaussian blur in edge extraction)
            self.model.edge_sigma = hyperparam_dict['sigma']
            print(f"  - Edge sigma set to {hyperparam_dict['sigma']}")

        if 'edge_mix_weight' in hyperparam_dict:
            # Update gamma (edge mixing weight)
            if hasattr(self.model, 'gamma'):
                self.model.gamma = hyperparam_dict['edge_mix_weight']
                print(f"  - Edge mix weight (gamma) set to {hyperparam_dict['edge_mix_weight']}")

        if 'inference_edge' in hyperparam_dict:
            if hasattr(self.model, 'inference_edge'):
                self.model.inference_edge = hyperparam_dict['inference_edge']
                print(f"  - Inference edge set to {hyperparam_dict['inference_edge']}")

    def merge_dicts(self, dict_list):
        result = {}

        keys_to_merge = [
            'description_proto',
            'description_features',
            'description_targets',
            'text_features',
            'text_targets',
            'image_proto',
            'images_features',
            'images_targets',
            'edge_proto',
        ]

        for key in keys_to_merge:
            tensors = [d[key] for d in dict_list if d[key] is not None]
            if len(tensors) > 0:
                result[key] = torch.cat(tensors, dim=0)
            else:
                result[key] = None

        weights = [len(d['class_index']) for d in dict_list]

        cov_keys = [
            'cov_image',
        ]
        cov_sums = {key: torch.zeros_like(dict_list[0][key]) for key in cov_keys}
        weight_sum = sum(weights)

        for i, d in enumerate(dict_list):
            for key in cov_keys:
                cov_sums[key] += d[key] * weights[i]

        for key in cov_keys:
            if weight_sum > 0:
                result[key] = cov_sums[key] / weight_sum

        return result

    @torch.no_grad()
    def run(self, hyperparam_dict=None, use_meta_router=False):
        print(f'Start inferencing on all tasks: [0, {self.data_manager.num_tasks - 1}]')
        state_dict_list = []
        # get dataset and trainer names
        data_name = os.path.splitext(os.path.basename(self.cfg.DATA_CFG_PATH))[0]
        train_name = os.path.splitext(os.path.basename(self.cfg.TRAIN_CFG_PATH))[0]
        prefix = f"{data_name}_{train_name}"
        print("PREFIX:", prefix)
        print("TOTAL LEN", self.data_manager.num_tasks)

        # Apply hyperparameters if provided
        if hyperparam_dict is not None:
            self._apply_hyperparameters(hyperparam_dict)

        # Enable router for inference if requested
        if use_meta_router:
            if hasattr(self.model, 'router_network') and self.model.router_network is not None:
                self.model.enable_router()
                print("[Runner] Using trained router network for inference")
            else:
                print("[Warning] Router requested but not available, using default parameters")
                use_meta_router = False
        for task_id in range(self.data_manager.num_tasks):
            print(f"TASK {task_id}")
            self.model.eval()
            self.model.task_id = task_id

            current_class_name = np.array(self.data_manager.class_names)[self.data_manager.class_index_in_task[task_id]]
            loader = self.data_manager.get_dataloader(task_id, source='train', mode='test', accumulate_past=False)

            self.model.building = True
            current_state_dict = self.model.build_task_statistics(current_class_name, loader,
                                                                  class_index=self.data_manager.class_index_in_task[task_id],
                                                                  calibrate_novel_vision_proto=self.cfg.TRAINER.BiMC.VISION_CALIBRATION, )
            self.model.building = False

            state_dict_list.append(current_state_dict)
            merged_state_dict = self.merge_dicts(state_dict_list)

            start_time = time.time()
            acc = self.inference_task_covariance(task_id, merged_state_dict)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f'+++++++++++  task {task_id}, time: {elapsed_time} ++++++++++++++++')

            print(f'=> Task [{task_id}], Acc: {acc["mean_acc"]:.3f}')
            self.acc_list.append(round(acc["mean_acc"], 3))
            self.task_acc_list.append(acc['task_acc'])

        print(f'Final acc:{self.acc_list}')
        print('Task-wise acc:')
        for i, task_acc in enumerate(self.task_acc_list):
            print(f'task {i:2d}, acc:{task_acc}')
        # ===========================================
        # DG-FSCIL: Target 도메인 테스트 (TS4, TS5)
        # ===========================================
        target_acc_dict = {}
        if self.is_dgfscil:
            print('\n' + '=' * 50)
            print('DG-FSCIL: Target Domain Evaluation')
            print('=' * 50)

            for target_domain in ['clipart', 'quickdraw']:
                print(f'\n=== Target Domain: {target_domain} ===')
                self.model.building = False
                acc = self.inference_target_domain(target_domain, merged_state_dict)
                target_acc_dict[target_domain] = acc
                print(f'=> {target_domain} Acc: {acc["mean_acc"]:.3f}')
                print(f'   Task-wise: {acc["task_acc"]}')

            print('\n' + '=' * 50)
            print('DG-FSCIL Final Results Summary')
            print('=' * 50)
            print(f'Source Domain Results (TS0-TS3): {self.acc_list}')
            for domain, acc in target_acc_dict.items():
                print(f'Target Domain ({domain}): {acc["mean_acc"]:.3f}')
            print('=' * 50)

        # ===========================================
        # SAVE RESULTS
        # ===========================================
        # Create output directory with prefix name
        output_dir = os.path.join("outputs", prefix)
        os.makedirs(output_dir, exist_ok=True)

        save_dict = {
            "acc_list": self.acc_list,
            "task_acc_list": self.task_acc_list
        }

        # DG-FSCIL 결과도 저장하도록 확장
        if self.is_dgfscil:
            save_dict["target_domain"] = {
                domain: {
                    "mean_acc": acc["mean_acc"],
                    "task_acc": acc["task_acc"]
                }
                for domain, acc in target_acc_dict.items()
            }

        # Add hyperparameters to save_dict if provided
        if hyperparam_dict is not None:
            save_dict["hyperparameters"] = hyperparam_dict

        # Generate filename from hyperparameters
        if hyperparam_dict is not None:
            # Create filename from hyperparameter values
            hyperparam_str = "_".join([f"{v}" for k, v in sorted(hyperparam_dict.items())])
            filename = f"{hyperparam_str}.json"
        else:
            filename = "results.json"

        # Save JSON
        json_path = os.path.join(output_dir, filename)
        with open(json_path, "w") as f:
            json.dump(save_dict, f, indent=2)
        print(f"Results saved to {json_path}")

    @torch.no_grad()
    def inference_task_covariance(self, task_id, state_dict):

        beta = self.cfg.DATASET.BETA
        self.model.task_id = task_id

        image_proto = state_dict['image_proto']
        cov_image = state_dict['cov_image']
        text_features = state_dict['text_features']
        description_proto = state_dict['description_proto']
        description_features = state_dict['description_features']
        description_targets = state_dict['description_targets']
        edge_proto = state_dict['edge_proto']

        num_base_class = len(self.data_manager.class_index_in_task[0])
        num_accumulated_class = max(self.data_manager.class_index_in_task[task_id]) + 1

        test_loader = self.data_manager.get_dataloader(task_id, source='test', mode='test')
        all_logits = []
        all_targets = []

        for i, batch in enumerate(tqdm(test_loader)):
            data, targets = self.parse_batch(batch)
            logits = self.model.forward_ours(data, targets, num_accumulated_class, num_base_class,
                                             image_proto,
                                             cov_image,
                                             description_proto,
                                             description_features,
                                             description_targets,
                                             text_features,
                                             edge_proto,
                                             beta=beta)

            all_logits.append(logits)
            all_targets.append(targets)

        all_logits = torch.cat(all_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        eval_acc = self.evaluator.calc_accuracy(all_logits, all_targets, task_id)
        print(f"Test acc mean: {eval_acc['mean_acc']}, task-wise acc: {eval_acc['task_acc']}")
        return eval_acc

    @torch.no_grad()
    def inference_target_domain(self, target_domain, state_dict):
        """Target 도메인(clipart/quickdraw)에서 평가"""
        beta = self.cfg.DATASET.BETA
        self.model.task_id = target_domain

        image_proto = state_dict['image_proto']
        cov_image = state_dict['cov_image']
        text_features = state_dict['text_features']
        description_proto = state_dict['description_proto']
        description_features = state_dict['description_features']
        description_targets = state_dict['description_targets']
        edge_proto = state_dict['edge_proto']

        num_base_class = len(self.data_manager.class_index_in_task[0])
        num_accumulated_class = self.data_manager.num_total_classes  # 345

        # Target 도메인 DataLoader
        test_loader = self.data_manager.get_target_domain_dataloader(target_domain)

        all_logits = []
        all_targets = []

        for i, batch in enumerate(tqdm(test_loader, desc=f'Testing {target_domain}')):
            data, targets = self.parse_batch(batch)
            logits = self.model.forward_ours(
                data, targets, num_accumulated_class, num_base_class,
                image_proto,
                cov_image,
                description_proto,
                description_features,
                description_targets,
                text_features,
                edge_proto,
                beta=beta,
            )
            all_logits.append(logits)
            all_targets.append(targets)

        all_logits = torch.cat(all_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # 마지막 태스크 기준으로 평가 (모든 345 클래스)
        eval_acc = self.evaluator.calc_accuracy(all_logits, all_targets, self.data_manager.num_tasks - 1)
        print(f"Target domain ({target_domain}) acc: {eval_acc['mean_acc']}, task-wise: {eval_acc['task_acc']}")
        return eval_acc

    def parse_batch(self, batch):
        data = batch['image']
        targets = batch['label']
        data = data.to(self.device)
        targets = targets.to(self.device)
        return data, targets

    # ======================================================
    # Meta-Learning Methods
    # ======================================================

    def meta_run(self):
        """
        Meta-learning training loop for router network.

        Episode structure:
            - 4 training domains: [real, infograph, painting, sketch]
            - Each episode: 3 support domains + 1 query domain
            - Inner loop: Update on support domains (3 domains sequentially)
            - Outer loop: Meta-update on query domain
        """
        print("\n" + "=" * 60)
        print("Meta-Learning: Training Router Network")
        print("=" * 60)

        # Ensure this is DG-FSCIL dataset
        if not self.is_dgfscil:
            raise ValueError("Meta-learning is only supported for DG-FSCIL (DomainNet) dataset!")

        # Freeze all except router
        self.model.freeze_all_except_router()

        # Get meta-learning config
        meta_cfg = self.cfg.TRAINER.BiMC.META
        num_episodes = meta_cfg.NUM_EPISODES
        inner_lr = meta_cfg.INNER_LR
        outer_lr = meta_cfg.OUTER_LR
        inner_steps = meta_cfg.INNER_STEPS
        k_support = meta_cfg.SUPPORT_SHOT
        k_query = meta_cfg.QUERY_SHOT

        # Setup optimizer for router network
        router_optimizer = torch.optim.Adam(
            self.model.router_network.parameters(),
            lr=outer_lr
        )

        print(f"Episodes: {num_episodes}")
        print(f"Inner LR: {inner_lr}, Outer LR: {outer_lr}")
        print(f"Inner Steps: {inner_steps}")
        print(f"Support/Query Split: {k_support}/{k_query}")
        print("=" * 60 + "\n")

        # Meta-training loop
        for episode in range(num_episodes):
            print(f"\n[Episode {episode + 1}/{num_episodes}]")

            # Sample support and query domains
            support_domains, query_domain, support_tasks, query_task = \
                self.data_manager.get_meta_episode_domains()

            print(f"  Support domains: {support_domains} (tasks: {support_tasks})")
            print(f"  Query domain: {query_domain} (task: {query_task})")

            # ==========================================
            # Inner Loop: Adapt on Support Domains
            # ==========================================
            # Save initial router parameters
            initial_params = {
                name: param.clone().detach()
                for name, param in self.model.router_network.named_parameters()
            }

            # Inner loop: Train on each support domain sequentially
            for sup_idx, (sup_domain, sup_task) in enumerate(zip(support_domains, support_tasks)):
                print(f"\n  Inner Loop [{sup_idx + 1}/3]: {sup_domain}")

                # Get support and query split for this domain
                support_dataset, query_dataset_inner = \
                    self.data_manager.get_k_shot_split(sup_task, k_support, k_query)

                # Use meta-specific batch size
                meta_batch_size = meta_cfg.BATCH_SIZE

                support_loader = self.data_manager.get_meta_dataloader(
                    support_dataset, batch_size=meta_batch_size, shuffle=False
                )

                query_loader_inner = self.data_manager.get_meta_dataloader(
                    query_dataset_inner, batch_size=meta_batch_size, shuffle=False
                )

                # Inner gradient steps
                for inner_step in range(inner_steps):
                    inner_loss = self._meta_inner_step(
                        support_loader, query_loader_inner,
                        k_support, k_query
                    )

                    # Manual gradient descent (simple SGD)
                    with torch.no_grad():
                        for name, param in self.model.router_network.named_parameters():
                            if param.grad is not None:
                                param.data -= inner_lr * param.grad
                                param.grad.zero_()

                    print(f"    Step {inner_step + 1}/{inner_steps}, Loss: {inner_loss:.4f}")

            # ==========================================
            # Outer Loop: Meta-update on Query Domain
            # ==========================================
            print(f"\n  Outer Loop: {query_domain}")

            # Get query domain data
            query_support_dataset, query_query_dataset = \
                self.data_manager.get_k_shot_split(query_task, k_support, k_query)

            query_support_loader = self.data_manager.get_meta_dataloader(
                query_support_dataset, batch_size=meta_batch_size, shuffle=False
            )

            query_query_loader = self.data_manager.get_meta_dataloader(
                query_query_dataset, batch_size=meta_batch_size, shuffle=False
            )

            # Compute meta-objective on query domain
            meta_loss = self._meta_outer_step(
                query_support_loader, query_query_loader,
                k_support, k_query
            )

            # Reset to initial parameters before meta-update
            with torch.no_grad():
                for name, param in self.model.router_network.named_parameters():
                    param.data = initial_params[name]

            # Meta-update using optimizer
            router_optimizer.zero_grad()
            meta_loss.backward()
            router_optimizer.step()

            print(f"  Meta Loss: {meta_loss.item():.4f}")

            # Periodic logging
            if (episode + 1) % 10 == 0:
                print(f"\n{'=' * 60}")
                print(f"Episode {episode + 1}/{num_episodes} completed")
                print(f"{'=' * 60}\n")

        print("\n" + "=" * 60)
        print("Meta-Learning Completed!")
        print("Router network trained successfully")
        print("=" * 60 + "\n")

        # Save router checkpoint
        # Create prefix the same way as in run()
        data_name = os.path.splitext(os.path.basename(self.cfg.DATA_CFG_PATH))[0]
        train_name = os.path.splitext(os.path.basename(self.cfg.TRAIN_CFG_PATH))[0]
        prefix = f"{data_name}_{train_name}"
        output_dir = os.path.join("outputs", prefix)
        self.save_router_checkpoint(output_dir)

        # Enable router for evaluation
        self.model.enable_router()

    def _meta_inner_step(self, support_loader, query_loader, k_support, k_query):
        """
        Inner loop: Compute loss on support set to adapt router.

        Args:
            support_loader: DataLoader for support set (k_support shots)
            query_loader: DataLoader for query set within inner loop (k_query shots)
            k_support: Number of support shots
            k_query: Number of query shots

        Returns:
            loss: Scalar loss value
        """
        self.model.router_network.train()

        total_loss = 0.0
        num_batches = 0

        # Build prototypes using support set
        support_features = []
        support_labels = []

        for batch in support_loader:
            images, labels = self.parse_batch(batch)

            # Predict router params
            router_params = self.model.predict_router_params(images)

            # Extract features with router params
            img_feat = self.model.extract_img_feature(images)
            img_feat = F.normalize(img_feat, dim=-1)

            # Extract edge features with router params
            edge_feat = self.model._extract_edge_features(images, labels, router_params)
            edge_feat = F.normalize(edge_feat, dim=-1)

            # Fuse features using router's gamma
            gamma = router_params['gamma']
            fused_feat = (1 - gamma) * img_feat + gamma * edge_feat
            fused_feat = F.normalize(fused_feat, dim=-1)

            support_features.append(fused_feat)
            support_labels.append(labels)

        support_features = torch.cat(support_features, dim=0)
        support_labels = torch.cat(support_labels, dim=0)

        # Compute prototypes
        unique_labels = torch.unique(support_labels)
        prototypes = []
        for c in unique_labels:
            idx = (support_labels == c)
            proto = support_features[idx].mean(dim=0)
            prototypes.append(proto)
        prototypes = torch.stack(prototypes, dim=0)  # (N_classes, D)
        prototypes = F.normalize(prototypes, dim=-1)

        # Compute loss on query set
        total_loss_tensor = 0.0  # Accumulate as tensor
        num_batches = 0

        for batch in query_loader:
            images, labels = self.parse_batch(batch)

            # Predict router params
            router_params = self.model.predict_router_params(images)

            # Extract and fuse features
            img_feat = self.model.extract_img_feature(images)
            img_feat = F.normalize(img_feat, dim=-1)

            edge_feat = self.model._extract_edge_features(images, labels, router_params)
            edge_feat = F.normalize(edge_feat, dim=-1)

            gamma = router_params['gamma']
            fused_feat = (1 - gamma) * img_feat + gamma * edge_feat
            fused_feat = F.normalize(fused_feat, dim=-1)

            # Compute similarity to prototypes
            logits = fused_feat @ prototypes.t()  # (B, N_classes)

            # Map labels to prototype indices
            label_to_idx = {c.item(): idx for idx, c in enumerate(unique_labels)}
            targets_idx = torch.tensor(
                [label_to_idx[l.item()] for l in labels],
                device=labels.device
            )

            # Cross-entropy loss
            loss = F.cross_entropy(logits, targets_idx)

            # Accumulate loss as tensor (not .item())
            total_loss_tensor += loss
            num_batches += 1

        # Compute average loss
        avg_loss_tensor = total_loss_tensor / max(num_batches, 1)

        # Single backward call
        avg_loss_tensor.backward()

        # Return scalar for logging
        return avg_loss_tensor.item()

    def _meta_outer_step(self, support_loader, query_loader, k_support, k_query):
        """
        Outer loop: Compute meta-objective on query domain.

        Similar to inner step, but returns loss tensor for meta-update.

        Returns:
            meta_loss: Tensor (requires_grad=True)
        """
        self.model.router_network.train()

        # Build prototypes using support set
        support_features = []
        support_labels = []

        for batch in support_loader:
            images, labels = self.parse_batch(batch)

            router_params = self.model.predict_router_params(images)

            img_feat = self.model.extract_img_feature(images)
            img_feat = F.normalize(img_feat, dim=-1)

            edge_feat = self.model._extract_edge_features(images, labels, router_params)
            edge_feat = F.normalize(edge_feat, dim=-1)

            gamma = router_params['gamma']
            fused_feat = (1 - gamma) * img_feat + gamma * edge_feat
            fused_feat = F.normalize(fused_feat, dim=-1)

            support_features.append(fused_feat)
            support_labels.append(labels)

        support_features = torch.cat(support_features, dim=0)
        support_labels = torch.cat(support_labels, dim=0)

        # Compute prototypes
        unique_labels = torch.unique(support_labels)
        prototypes = []
        for c in unique_labels:
            idx = (support_labels == c)
            proto = support_features[idx].mean(dim=0)
            prototypes.append(proto)
        prototypes = torch.stack(prototypes, dim=0)
        prototypes = F.normalize(prototypes, dim=-1)

        # Compute loss on query set
        total_loss = 0.0
        num_batches = 0

        for batch in query_loader:
            images, labels = self.parse_batch(batch)

            router_params = self.model.predict_router_params(images)

            img_feat = self.model.extract_img_feature(images)
            img_feat = F.normalize(img_feat, dim=-1)

            edge_feat = self.model._extract_edge_features(images, labels, router_params)
            edge_feat = F.normalize(edge_feat, dim=-1)

            gamma = router_params['gamma']
            fused_feat = (1 - gamma) * img_feat + gamma * edge_feat
            fused_feat = F.normalize(fused_feat, dim=-1)

            logits = fused_feat @ prototypes.t()

            label_to_idx = {c.item(): idx for idx, c in enumerate(unique_labels)}
            targets_idx = torch.tensor(
                [label_to_idx[l.item()] for l in labels],
                device=labels.device
            )

            loss = F.cross_entropy(logits, targets_idx)

            total_loss += loss
            num_batches += 1

        meta_loss = total_loss / max(num_batches, 1)
        return meta_loss

    # ======================================================
    # Router Checkpoint Management
    # ======================================================

    def save_router_checkpoint(self, output_dir):
        """
        Save router network checkpoint.

        Args:
            output_dir: Base output directory (e.g., outputs/prefix)
        """
        if self.model.router_network is None:
            print("[Warning] No router network to save")
            return

        import os
        # Create router_checkpoints subdirectory
        checkpoint_dir = os.path.join(output_dir, "router_checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Create checkpoint filename with timestamp
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(checkpoint_dir, f"router_{timestamp}.pth")

        # Save router state dict
        checkpoint = {
            'router_state_dict': self.model.router_network.state_dict(),
            'timestamp': timestamp,
            'config': {
                'input_dim': 512,
                'hidden_dim': self.cfg.TRAINER.BiMC.META.ROUTER_HIDDEN_DIM,
            }
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"\n[Checkpoint] Router saved to: {checkpoint_path}")

        # Also save as 'latest'
        latest_path = os.path.join(checkpoint_dir, "router_latest.pth")
        torch.save(checkpoint, latest_path)
        print(f"[Checkpoint] Latest router saved to: {latest_path}\n")

    def load_router_checkpoint(self, checkpoint_path):
        """
        Load router network from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        if self.model.router_network is None:
            print("[Error] Router network not initialized!")
            return

        import os
        if not os.path.exists(checkpoint_path):
            print(f"[Error] Checkpoint not found: {checkpoint_path}")
            return

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load state dict
        self.model.router_network.load_state_dict(checkpoint['router_state_dict'])

        print(f"[Checkpoint] Router loaded from: {checkpoint_path}")
        if 'timestamp' in checkpoint:
            print(f"[Checkpoint] Saved at: {checkpoint['timestamp']}")

        # Enable router
        self.model.enable_router()