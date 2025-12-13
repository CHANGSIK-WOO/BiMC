import torch
import torch.nn as nn
import models.clip as clip
from datasets.data_manager import DatasetManager
from torch.nn import functional as F
from PIL import Image
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
    def run(self):
        print(f'Start inferencing on all tasks: [0, {self.data_manager.num_tasks - 1}]')
        state_dict_list = []
        # get dataset and trainer names
        data_name = os.path.splitext(os.path.basename(self.cfg.DATA_CFG_PATH))[0]
        train_name = os.path.splitext(os.path.basename(self.cfg.TRAIN_CFG_PATH))[0]
        prefix = f"{data_name}_{train_name}"
        print("PREFIX:", prefix)
        print("TOTAL LEN", self.data_manager.num_tasks)
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
        os.makedirs("outputs", exist_ok=True)

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

        # Save JSON
        with open(f"outputs/{prefix}.json", "w") as f:
            json.dump(save_dict, f)

        # Save TXT summary
        with open(f"outputs/{prefix}.txt", "w") as f:
            f.write("Source Domain Results:\n")
            for i, task_acc in enumerate(self.task_acc_list):
                f.write(f"task {i:2d}, acc: {task_acc}\n")

            if self.is_dgfscil:
                f.write("\nTarget Domain Results:\n")
                for domain, acc in target_acc_dict.items():
                    f.write(f"{domain}: mean_acc={acc['mean_acc']}, task_acc={acc['task_acc']}\n")

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


class MetaRunner:
    """
    Meta-learning runner for learning domain-invariant edge filter hyperparameters.
    Implements 2-stage meta-learning:
    - Stage 1: Class-level meta-learning on Real domain
    - Stage 2: Cross-domain meta-learning with LODO strategy
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.DEVICE.DEVICE_NAME

        # Load DomainNet dataset
        from datasets.domainnet import DomainNet, MetaDatasetManager
        from torchvision import transforms
        from PIL import Image

        self.domainnet = DomainNet(root=cfg.DATASET.ROOT, seed=cfg.SEED)
        self.meta_manager = MetaDatasetManager(self.domainnet, cfg)

        # Data preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])

        # Initialize BiMC model
        self.model = BiMC(cfg, self.meta_manager.dataset.template, self.device)

        # Meta-learning configuration
        self.meta_cfg = cfg.TRAINER.BiMC.META
        self.inner_lr = self.meta_cfg.INNER_LR
        self.meta_lr = self.meta_cfg.META_LR
        self.inner_steps = self.meta_cfg.INNER_STEPS

        # Meta optimizer
        if self.model.meta_params is not None:
            self.meta_optimizer = torch.optim.Adam(
                self.model.meta_params.parameters(),
                lr=self.meta_lr
            )
        else:
            raise ValueError("Meta-learning is enabled but meta_params is None. Check BiMC initialization.")

        print(f"MetaRunner initialized")
        print(f"Inner LR: {self.inner_lr}, Meta LR: {self.meta_lr}, Inner steps: {self.inner_steps}")

    def load_episode_images(self, image_paths):
        """Load images from paths and preprocess them."""
        images = []
        for path in image_paths:
            img = Image.open(path).convert('RGB')
            img_tensor = self.preprocess(img)
            images.append(img_tensor)
        return torch.stack(images).to(self.device)

    def meta_train_stage1(self):
        """
        Stage 1: Class-level meta-learning on Real domain.
        Learn class-robust edge filter initialization from 240 base classes.
        """
        print("\n" + "="*80)
        print("Stage 1: Class-level Meta-Learning on Real Domain")
        print("="*80)

        episode_generator = self.meta_manager.get_stage1_episode_generator()

        total_loss = 0.0
        total_acc = 0.0

        for episode in tqdm(episode_generator, total=self.meta_cfg.STAGE1_EPISODES, desc="Stage 1"):
            # Load episode data
            support_images = self.load_episode_images(episode['support_images'])
            support_labels = torch.from_numpy(episode['support_labels']).to(self.device)
            query_images = self.load_episode_images(episode['query_images'])
            query_labels = torch.from_numpy(episode['query_labels']).to(self.device)

            # Meta-training step
            self.meta_optimizer.zero_grad()

            query_loss, query_acc = self.model.meta_train_step(
                support_images, support_labels,
                query_images, query_labels,
                self.model.meta_params,
                self.inner_lr,
                self.inner_steps
            )

            # Meta-update
            query_loss.backward()
            self.meta_optimizer.step()

            total_loss += query_loss.item()
            total_acc += query_acc.item()

            # Log every 100 episodes
            if (episode['episode_idx'] + 1) % 100 == 0:
                avg_loss = total_loss / (episode['episode_idx'] + 1)
                avg_acc = total_acc / (episode['episode_idx'] + 1)
                print(f"Episode {episode['episode_idx'] + 1}/{self.meta_cfg.STAGE1_EPISODES} - "
                      f"Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")

        avg_loss = total_loss / self.meta_cfg.STAGE1_EPISODES
        avg_acc = total_acc / self.meta_cfg.STAGE1_EPISODES

        print(f"\nStage 1 Complete - Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.4f}")
        print(f"Learned parameters:")
        print(f"  Sigma: {self.model.meta_params.get_sigma().item():.4f}")
        print(f"  Gamma: {self.model.meta_params.get_gamma().item():.4f}")
        print(f"  Laplacian kernel:\n{self.model.meta_params.get_laplacian_kernel()}")

        # Save Stage 1 checkpoint
        self.save_meta_params('stage1_meta_params.pth')

        return {'loss': avg_loss, 'acc': avg_acc}

    def meta_train_stage2(self):
        """
        Stage 2: Cross-domain meta-learning with Leave-One-Domain-Out.
        Learn domain-invariant edge filters using incremental classes.
        """
        print("\n" + "="*80)
        print("Stage 2: Cross-Domain Meta-Learning with LODO")
        print("="*80)

        episode_generator = self.meta_manager.get_stage2_episode_generator()

        total_loss = 0.0
        total_acc = 0.0
        domain_stats = {domain: {'loss': 0.0, 'acc': 0.0, 'count': 0}
                       for domain in self.meta_manager.source_domains}

        for episode in tqdm(episode_generator, total=self.meta_cfg.STAGE2_EPISODES, desc="Stage 2"):
            # Load episode data
            support_images = self.load_episode_images(episode['support_images'])
            support_labels = torch.from_numpy(episode['support_labels']).to(self.device)
            query_images = self.load_episode_images(episode['query_images'])
            query_labels = torch.from_numpy(episode['query_labels']).to(self.device)

            # Meta-training step
            self.meta_optimizer.zero_grad()

            query_loss, query_acc = self.model.meta_train_step(
                support_images, support_labels,
                query_images, query_labels,
                self.model.meta_params,
                self.inner_lr,
                self.inner_steps
            )

            # Meta-update
            query_loss.backward()
            self.meta_optimizer.step()

            total_loss += query_loss.item()
            total_acc += query_acc.item()

            # Track per-domain statistics
            query_domain = episode['query_domain']
            domain_stats[query_domain]['loss'] += query_loss.item()
            domain_stats[query_domain]['acc'] += query_acc.item()
            domain_stats[query_domain]['count'] += 1

            # Log every 100 episodes
            if (episode['episode_idx'] + 1) % 100 == 0:
                avg_loss = total_loss / (episode['episode_idx'] + 1)
                avg_acc = total_acc / (episode['episode_idx'] + 1)
                print(f"Episode {episode['episode_idx'] + 1}/{self.meta_cfg.STAGE2_EPISODES} - "
                      f"Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}, Query: {query_domain}")

        avg_loss = total_loss / self.meta_cfg.STAGE2_EPISODES
        avg_acc = total_acc / self.meta_cfg.STAGE2_EPISODES

        print(f"\nStage 2 Complete - Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.4f}")
        print(f"\nPer-domain statistics:")
        for domain in self.meta_manager.source_domains:
            if domain_stats[domain]['count'] > 0:
                d_loss = domain_stats[domain]['loss'] / domain_stats[domain]['count']
                d_acc = domain_stats[domain]['acc'] / domain_stats[domain]['count']
                print(f"  {domain}: Loss {d_loss:.4f}, Acc {d_acc:.4f} ({domain_stats[domain]['count']} episodes)")

        print(f"\nFinal learned parameters:")
        print(f"  Sigma: {self.model.meta_params.get_sigma().item():.4f}")
        print(f"  Gamma: {self.model.meta_params.get_gamma().item():.4f}")
        print(f"  Laplacian kernel:\n{self.model.meta_params.get_laplacian_kernel()}")

        # Save Stage 2 checkpoint
        self.save_meta_params('stage2_meta_params.pth')

        return {'loss': avg_loss, 'acc': avg_acc, 'domain_stats': domain_stats}

    def save_meta_params(self, filename):
        """Save meta-learned parameters."""
        save_path = os.path.join('outputs', 'meta_params', filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        torch.save({
            'meta_params_state_dict': self.model.meta_params.state_dict(),
            'sigma': self.model.meta_params.get_sigma().item(),
            'gamma': self.model.meta_params.get_gamma().item(),
            'laplacian_kernel': self.model.meta_params.get_laplacian_kernel().cpu().numpy(),
        }, save_path)

        print(f"Meta parameters saved to {save_path}")

    def load_meta_params(self, filename):
        """Load meta-learned parameters."""
        load_path = os.path.join('outputs', 'meta_params', filename)

        if not os.path.exists(load_path):
            print(f"Warning: Meta params file {load_path} not found")
            return False

        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.meta_params.load_state_dict(checkpoint['meta_params_state_dict'])

        print(f"Meta parameters loaded from {load_path}")
        print(f"  Sigma: {checkpoint['sigma']:.4f}")
        print(f"  Gamma: {checkpoint['gamma']:.4f}")

        return True

    def run(self):
        """
        Run complete meta-learning pipeline:
        1. Stage 1: Class-level meta-learning on Real domain
        2. Stage 2: Cross-domain meta-learning with LODO
        """
        print("\n" + "="*80)
        print("Meta-Learning for Domain-Invariant Edge Filters")
        print("="*80)

        # Stage 1
        stage1_results = self.meta_train_stage1()

        # Stage 2 (initialized from Stage 1)
        stage2_results = self.meta_train_stage2()

        print("\n" + "="*80)
        print("Meta-Learning Complete")
        print("="*80)
        print(f"Stage 1 - Loss: {stage1_results['loss']:.4f}, Acc: {stage1_results['acc']:.4f}")
        print(f"Stage 2 - Loss: {stage2_results['loss']:.4f}, Acc: {stage2_results['acc']:.4f}")

        return stage1_results, stage2_results