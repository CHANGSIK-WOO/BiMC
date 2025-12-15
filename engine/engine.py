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
    def run(self, hyperparam_dict=None, use_meta_prompts=False):
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

        # Enable prompts for inference if requested
        if use_meta_prompts:
            if hasattr(self.model, 'prompt_pool') and self.model.prompt_pool is not None:
                self.model.enable_prompts()
                print("[Runner] Using trained prompts for inference")

                # Average all task prompts for inference
                if len(self.model.prompt_pool) > 0:
                    prompt_tensors = [p.data for p in self.model.prompt_pool.values()]
                    avg_prompt = torch.stack(prompt_tensors, dim=0).mean(dim=0)
                    self.model.set_prompt(avg_prompt)
                    print(f"[Runner] Using averaged prompt from {len(self.model.prompt_pool)} tasks")
                else:
                    print("[Warning] Prompt pool is empty")
                    use_meta_prompts = False
            else:
                print("[Warning] Prompts requested but not available, using default")
                use_meta_prompts = False

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
        Meta-learning training loop for learnable prompts.

        Task structure (CORRECTED):
            - Task 1 (base): Meta-learning with random 35-class episodes (28 support / 7 query, 5-shot)
            - Task 2,3,4 (incremental): Standard training on 5-shot data from each task
            - Each incremental prompt starts from the base prompt

        Meta-objective:
            - Maximize image-text embedding similarity (cosine similarity)
        """
        print("\n" + "=" * 60)
        print("Meta-Learning: Training Learnable Prompts (Corrected Structure)")
        print("=" * 60)

        # Ensure this is DG-FSCIL dataset
        if not self.is_dgfscil:
            raise ValueError("Meta-learning is only supported for DG-FSCIL (DomainNet) dataset!")

        self.model.meta_training = True

        # Get meta-learning config
        meta_cfg = self.cfg.TRAINER.BiMC.META
        num_episodes = meta_cfg.NUM_EPISODES
        base_finetune_enabled = meta_cfg.BASE_FINETUNE_ENABLED
        base_finetune_epochs = meta_cfg.BASE_FINETUNE_EPOCHS
        inc_finetune_epochs = meta_cfg.INC_FINETUNE_EPOCHS
        inner_lr = meta_cfg.INNER_LR
        outer_lr = meta_cfg.OUTER_LR
        inner_steps = meta_cfg.INNER_STEPS

        base_support_classes = meta_cfg.BASE_SUPPORT_CLASSES
        base_query_classes = meta_cfg.BASE_QUERY_CLASSES
        k_shot = meta_cfg.SUPPORT_SHOT

        print(f"Base meta-learning episodes: {num_episodes}")
        print(f"Base fine-tuning: {'Enabled' if base_finetune_enabled else 'Disabled'}")
        if base_finetune_enabled:
            print(f"Base fine-tuning epochs: {base_finetune_epochs}")
        print(f"Incremental fine-tuning epochs: {inc_finetune_epochs}")
        print(f"Inner LR: {inner_lr}, Outer LR: {outer_lr}")
        print(f"Inner Steps: {inner_steps}")
        print(f"Base split per episode: {base_support_classes} support / {base_query_classes} query ({k_shot}-shot)")
        print("=" * 60 + "\n")

        # ==========================================
        # Step 1: Meta-learning on Base Task (Task 0)
        # ==========================================

        print(f"\n{'=' * 60}")
        print("Step 1: Meta-Learning on Base Task (Task 0)")
        print(f"{'=' * 60}")

        # Create initial prompt for meta-learning
        meta_prompt_param = self.model.create_prompt(task_id=0)
        prompt_optimizer = torch.optim.Adam([meta_prompt_param], lr=outer_lr)
        self.model.freeze_all_except_prompt(meta_prompt_param)

        print(f"[Meta-Learning] Each episode: {base_support_classes} support + {base_query_classes} query classes ({k_shot}-shot)")

        meta_batch_size = meta_cfg.BATCH_SIZE

        # Meta-training episodes on base task
        for episode in range(num_episodes):
            print(f"\n[Episode {episode + 1}/{num_episodes}]")

            # Get random class split from base task (with k-shot sampling)
            support_dataset, query_dataset = \
                self.data_manager.get_class_split(0, base_support_classes, base_query_classes, k_shot=k_shot)

            support_loader = self.data_manager.get_meta_dataloader(
                support_dataset, batch_size=meta_batch_size, shuffle=True
            )

            query_loader = self.data_manager.get_meta_dataloader(
                query_dataset, batch_size=meta_batch_size, shuffle=False
            )

            # ==========================================
            # Inner Loop: Adapt on Support Classes
            # ==========================================
            initial_prompt = meta_prompt_param.data.clone()

            for inner_step in range(inner_steps):
                inner_loss = self._prompt_inner_step(
                    meta_prompt_param, support_loader, query_loader
                )

                # Manual gradient descent (simple SGD)
                with torch.no_grad():
                    if meta_prompt_param.grad is not None:
                        meta_prompt_param.data -= inner_lr * meta_prompt_param.grad
                        meta_prompt_param.grad.zero_()

                if inner_step == 0 or (inner_step + 1) % inner_steps == 0:
                    print(f"  Inner Step {inner_step + 1}/{inner_steps}, Loss: {inner_loss:.4f}")

            # ==========================================
            # Outer Loop: Meta-update on Query Classes
            # ==========================================
            meta_loss = self._prompt_outer_step(
                meta_prompt_param, support_loader, query_loader
            )

            # Reset to initial prompt
            with torch.no_grad():
                meta_prompt_param.data.copy_(initial_prompt)

            # Meta-update using optimizer
            prompt_optimizer.zero_grad()
            meta_loss.backward()
            prompt_optimizer.step()

            if episode == 0 or (episode + 1) % 10 == 0 or episode == num_episodes - 1:
                print(f"  Meta Loss: {meta_loss.item():.4f}")

        # Save meta-learned prompt (this is the shared initialization for all tasks)
        meta_prompt = meta_prompt_param.detach().clone()
        print(f"\n[Meta-Learning] Completed! Meta-prompt saved.")

        # ==========================================
        # Step 2: Fine-tune on ALL tasks (0, 1, 2, 3) starting from meta_prompt
        # ==========================================

        print(f"\n{'=' * 60}")
        print("Step 2: Fine-Tuning All Tasks from Meta-Prompt")
        print(f"{'=' * 60}")

        for task_id in range(self.data_manager.num_tasks):
            print(f"\n{'=' * 60}")
            print(f"Fine-Tuning Task {task_id}")
            print(f"{'=' * 60}")

            # Create prompt initialized from meta_prompt
            task_prompt = self.model.create_prompt(task_id=task_id)
            with torch.no_grad():
                task_prompt.data.copy_(meta_prompt.data)
            print(f"[Task {task_id}] Initialized from meta-prompt")

            # Setup optimizer
            task_optimizer = torch.optim.Adam([task_prompt], lr=outer_lr)
            self.model.freeze_all_except_prompt(task_prompt)

            # ==========================================
            # Task 0: Fine-tune with FULL-SHOT data (if enabled)
            # ==========================================
            if task_id == 0:
                if base_finetune_enabled:
                    print(f"[Task {task_id}] Fine-tuning with full-shot data")

                    base_batch_size = meta_cfg.BASE_BATCH_SIZE
                    base_dataset = self.data_manager.get_dataset(
                        task_id, source='train', mode='train', accumulated_past=False, k_shot=None  # Full-shot!
                    )
                    train_loader = self.data_manager.get_meta_dataloader(
                        base_dataset, batch_size=base_batch_size, shuffle=True
                    )

                    print(f"[Task {task_id}] Batch size: {base_batch_size}, Epochs: {base_finetune_epochs}")

                    for epoch in range(base_finetune_epochs):
                        epoch_loss = self._train_prompt_standard(
                            task_prompt, train_loader, task_optimizer
                        )
                        print(f"  Epoch {epoch + 1}/{base_finetune_epochs}, Loss: {epoch_loss:.4f}")
                else:
                    print(f"[Task {task_id}] Skipping fine-tuning (BASE_FINETUNE_ENABLED=False)")

            # ==========================================
            # Tasks 1,2,3: Fine-tune with 5-SHOT data
            # ==========================================
            else:
                print(f"[Task {task_id}] Fine-tuning with 5-shot data")

                inc_batch_size = meta_cfg.INC_BATCH_SIZE
                inc_dataset = self.data_manager.get_dataset(
                    task_id, source='train', mode='train', accumulated_past=False, k_shot=k_shot
                )
                train_loader = self.data_manager.get_meta_dataloader(
                    inc_dataset, batch_size=inc_batch_size, shuffle=True
                )

                print(f"[Task {task_id}] Batch size: {inc_batch_size}, Epochs: {inc_finetune_epochs}")

                for epoch in range(inc_finetune_epochs):
                    epoch_loss = self._train_prompt_standard(
                        task_prompt, train_loader, task_optimizer
                    )
                    print(f"  Epoch {epoch + 1}/{inc_finetune_epochs}, Loss: {epoch_loss:.4f}")

            # Save task-specific prompt to pool
            self.model.prompt_pool[task_id] = task_prompt.detach().clone()
            print(f"\n[Task {task_id}] Fine-tuning completed, prompt saved to pool")

        print("\n" + "=" * 60)
        print("Meta-Learning Completed!")
        print(f"Trained prompts for {len(self.model.prompt_pool)} tasks")
        print("=" * 60 + "\n")

        # Save prompt checkpoint
        data_name = os.path.splitext(os.path.basename(self.cfg.DATA_CFG_PATH))[0]
        train_name = os.path.splitext(os.path.basename(self.cfg.TRAIN_CFG_PATH))[0]
        prefix = f"{data_name}_{train_name}"
        output_dir = os.path.join("outputs", prefix)
        latest_path = self.save_prompt_checkpoint(output_dir)

        # Enable prompts for evaluation
        self.model.enable_prompts()
        self.model.meta_training = False

        return latest_path

    def _prompt_inner_step(self, prompt, support_loader, query_loader):
        """
        Inner loop: Adapt prompt on support classes, evaluate on query classes.

        Meta-objective: Maximize image-text embedding similarity.

        Args:
            prompt: Current learnable prompt (requires_grad=True)
            support_loader: DataLoader for support classes
            query_loader: DataLoader for query classes

        Returns:
            loss: Scalar loss value
        """
        self.model.set_prompt(prompt)

        # Collect text embeddings for all classes in support set
        text_embeddings_dict = {}
        support_labels_set = set()

        # Build text embeddings for support classes
        for batch in tqdm(support_loader, desc="  Support", leave=False):
            images, labels = self.parse_batch(batch)
            for label in labels:
                label_idx = label.item()
                if label_idx not in text_embeddings_dict:
                    class_name = self.data_manager.class_names[label_idx]
                    # Get text embedding for this class
                    text_emb = self._get_text_embedding(class_name)
                    text_embeddings_dict[label_idx] = text_emb
                    support_labels_set.add(label_idx)

        # Compute loss on query set
        total_loss_tensor = 0.0
        num_samples = 0

        for batch in tqdm(query_loader, desc="  Query", leave=False):
            images, labels = self.parse_batch(batch)

            # Extract image features with prompt
            img_feat = self.model.extract_img_feature(images)
            img_feat = F.normalize(img_feat, dim=-1)

            # Get corresponding text embeddings
            batch_text_embs = []
            for label in labels:
                label_idx = label.item()
                if label_idx in text_embeddings_dict:
                    batch_text_embs.append(text_embeddings_dict[label_idx])
                else:
                    # Handle unseen labels (should not happen in query set)
                    class_name = self.data_manager.class_names[label_idx]
                    text_emb = self._get_text_embedding(class_name)
                    batch_text_embs.append(text_emb)

            batch_text_embs = torch.stack(batch_text_embs, dim=0)
            batch_text_embs = F.normalize(batch_text_embs, dim=-1)

            # Compute cosine similarity loss (negative similarity to minimize)
            similarity = (img_feat * batch_text_embs).sum(dim=1)  # (B,)
            loss = -similarity.mean()  # Maximize similarity = minimize negative similarity

            total_loss_tensor += loss * images.size(0)
            num_samples += images.size(0)

        avg_loss_tensor = total_loss_tensor / max(num_samples, 1)
        avg_loss_tensor.backward()

        return avg_loss_tensor.item()

    def _prompt_outer_step(self, prompt, support_loader, query_loader):
        """
        Outer loop: Compute meta-objective on query classes.

        Returns:
            meta_loss: Tensor (requires_grad=True)
        """
        self.model.set_prompt(prompt)

        # Build text embeddings for support classes
        text_embeddings_dict = {}

        for batch in tqdm(support_loader, desc="  Support", leave=False):
            images, labels = self.parse_batch(batch)
            for label in labels:
                label_idx = label.item()
                if label_idx not in text_embeddings_dict:
                    class_name = self.data_manager.class_names[label_idx]
                    text_emb = self._get_text_embedding(class_name)
                    text_embeddings_dict[label_idx] = text_emb

        # Compute loss on query set
        total_loss = 0.0
        num_samples = 0

        for batch in tqdm(query_loader, desc="  Query", leave=False):
            images, labels = self.parse_batch(batch)

            img_feat = self.model.extract_img_feature(images)
            img_feat = F.normalize(img_feat, dim=-1)

            batch_text_embs = []
            for label in labels:
                label_idx = label.item()
                if label_idx in text_embeddings_dict:
                    batch_text_embs.append(text_embeddings_dict[label_idx])
                else:
                    class_name = self.data_manager.class_names[label_idx]
                    text_emb = self._get_text_embedding(class_name)
                    batch_text_embs.append(text_emb)

            batch_text_embs = torch.stack(batch_text_embs, dim=0)
            batch_text_embs = F.normalize(batch_text_embs, dim=-1)

            similarity = (img_feat * batch_text_embs).sum(dim=1)
            loss = -similarity.mean()

            total_loss += loss * images.size(0)
            num_samples += images.size(0)

        meta_loss = total_loss / max(num_samples, 1)
        return meta_loss

    def _train_prompt_standard(self, prompt, train_loader, optimizer):
        """
        Standard training for base task: all classes, standard gradient descent.

        Args:
            prompt: Learnable prompt parameter
            train_loader: Training data loader
            optimizer: Optimizer for prompt

        Returns:
            avg_loss: Average loss for this epoch
        """
        self.model.set_prompt(prompt)

        total_loss = 0.0
        num_samples = 0

        for batch in tqdm(train_loader, desc="  Training", leave=False):
            images, labels = self.parse_batch(batch)

            # Extract image features with prompt
            img_feat = self.model.extract_img_feature(images)
            img_feat = F.normalize(img_feat, dim=-1)

            # Get text embeddings for all classes in batch
            batch_text_embs = []
            for label in labels:
                label_idx = label.item()
                class_name = self.data_manager.class_names[label_idx]
                text_emb = self._get_text_embedding(class_name)
                batch_text_embs.append(text_emb)

            batch_text_embs = torch.stack(batch_text_embs, dim=0)
            batch_text_embs = F.normalize(batch_text_embs, dim=-1)

            # Maximize cosine similarity
            similarity = (img_feat * batch_text_embs).sum(dim=1)
            loss = -similarity.mean()

            # Standard gradient descent
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            num_samples += images.size(0)

        avg_loss = total_loss / max(num_samples, 1)
        return avg_loss

    @torch.no_grad()
    def _get_text_embedding(self, class_name):
        """
        Get text embedding for a class name using CLIP.

        Args:
            class_name: Class name string

        Returns:
            text_emb: Text embedding (D,)
        """
        import models.clip.clip as clip

        # Format class name
        class_name = class_name.replace('_', ' ').replace('-', ' ')

        # Use single template
        text = f"a photo of a {class_name}"
        text_token = clip.tokenize([text]).to(self.device)

        # Encode text
        text_emb = self.model.clip_model.encode_text(text_token)
        text_emb = F.normalize(text_emb, dim=-1)
        text_emb = text_emb.squeeze(0)  # (D,)

        return text_emb

    # ======================================================
    # Prompt Checkpoint Management
    # ======================================================

    def save_prompt_checkpoint(self, output_dir):
        """
        Save prompt pool checkpoint.

        Args:
            output_dir: Base output directory (e.g., outputs/prefix)
        """
        if self.model.prompt_pool is None or len(self.model.prompt_pool) == 0:
            print("[Warning] No prompts to save")
            return

        import os
        # Create checkpoint directory
        checkpoint_dir = output_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Create checkpoint filename with timestamp
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Convert prompt pool to state dict
        prompt_state_dict = {}
        for task_id, prompt in self.model.prompt_pool.items():
            prompt_state_dict[f'task_{task_id}'] = prompt.cpu()

        checkpoint = {
            'prompt_pool': prompt_state_dict,
            'timestamp': timestamp,
            'config': {
                'prompt_length': self.cfg.TRAINER.BiMC.META.PROMPT_LENGTH,
                'prompt_dim': self.cfg.TRAINER.BiMC.META.PROMPT_DIM,
                'num_tasks': len(prompt_state_dict),
            }
        }

        # Save as 'latest'
        latest_path = os.path.join(checkpoint_dir, "prompts_latest.pth")
        torch.save(checkpoint, latest_path)
        print(f"[Checkpoint] Prompts saved to: {latest_path}")
        print(f"[Checkpoint] Saved {len(prompt_state_dict)} task prompts\n")

        torch.save(checkpoint, "outputs/prompts_latest.pth")
        return latest_path

    def load_prompt_checkpoint(self, checkpoint_path):
        """
        Load prompt pool from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        if self.model.prompt_pool is None:
            print("[Error] Prompt pool not initialized!")
            return

        import os
        if not os.path.exists(checkpoint_path):
            print(f"[Error] Checkpoint not found: {checkpoint_path}")
            return

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load prompt pool
        prompt_state_dict = checkpoint['prompt_pool']
        self.model.prompt_pool = {}

        for key, prompt_tensor in prompt_state_dict.items():
            task_id = int(key.split('_')[1])  # Extract task_id from 'task_0', 'task_1', etc.
            self.model.prompt_pool[task_id] = prompt_tensor.to(self.device)

        print(f"[Checkpoint] Prompts loaded from: {checkpoint_path}")
        if 'timestamp' in checkpoint:
            print(f"[Checkpoint] Saved at: {checkpoint['timestamp']}")
        print(f"[Checkpoint] Loaded {len(self.model.prompt_pool)} task prompts")

        # Enable prompts
        self.model.enable_prompts()