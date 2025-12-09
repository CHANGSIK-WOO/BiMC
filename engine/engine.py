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
        for i in range(self.data_manager.num_tasks):
            print(f"TASK {i}")
            self.model.eval()

            current_class_name = np.array(self.data_manager.class_names)[self.data_manager.class_index_in_task[i]]
            loader = self.data_manager.get_dataloader(i, source='train', mode='test', accumulate_past=False)

            current_state_dict = self.model.build_task_statistics(current_class_name, loader,
                                                                  class_index=self.data_manager.class_index_in_task[i],
                                                                  calibrate_novel_vision_proto=self.cfg.TRAINER.BiMC.VISION_CALIBRATION, )

            state_dict_list.append(current_state_dict)
            merged_state_dict = self.merge_dicts(state_dict_list)

            start_time = time.time()
            acc = self.inference_task_covariance(i, merged_state_dict)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f'+++++++++++  task {i}, time: {elapsed_time} ++++++++++++++++')

            print(f'=> Task [{i}], Acc: {acc["mean_acc"]:.3f}')
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
            logits = self.model.forward_ours(data, num_accumulated_class, num_base_class,
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

        image_proto = state_dict['image_proto']
        cov_image = state_dict['cov_image']
        text_features = state_dict['text_features']
        description_proto = state_dict['description_proto']
        description_features = state_dict['description_features']
        description_targets = state_dict['description_targets']

        num_base_class = len(self.data_manager.class_index_in_task[0])
        num_accumulated_class = self.data_manager.num_total_classes  # 345

        # Target 도메인 DataLoader
        test_loader = self.data_manager.get_target_domain_dataloader(target_domain)

        all_logits = []
        all_targets = []

        for i, batch in enumerate(tqdm(test_loader, desc=f'Testing {target_domain}')):
            data, targets = self.parse_batch(batch)
            logits = self.model.forward_ours(
                data, num_accumulated_class, num_base_class,
                image_proto,
                cov_image,
                description_proto,
                description_features,
                description_targets,
                text_features,
                beta=beta
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