import os
import numpy as np
import torch
from .dataset_base import DatasetBase


class DomainNet(DatasetBase):

    def __init__(self, root, train_ratio=0.8, seed=1):
        super(DomainNet, self).__init__(root=root, name='domainnet')

        self.root = os.path.join(root, 'domainbed/DomainNet')
        self.train_ratio = train_ratio
        self.seed = seed

        # 6개 도메인 정의 (D0 ~ D5)
        self.domains = ['real', 'infograph', 'painting', 'sketch', 'clipart', 'quickdraw']

        # 도메인 매핑
        self.domain_to_idx = {domain: idx for idx, domain in enumerate(self.domains)}

        # Incremental session에서 사용할 105개 클래스
        self.inc_session_1_classes = [
            'leaf', 'flashlight', 'ladder', 'umbrella', 'fork', 'shoe', 'panda', 'teapot',
            'bush', 'washing_machine', 'saxophone', 'frog', 'police_car', 'traffic_light',
            'train', 'baseball_bat', 'stitches', 'stove', 'pear', 'arm', 'lobster', 'snowflake',
            'wheel', 'squirrel', 'popsicle', 'cruise_ship', 'pencil', 'trumpet', 'snorkel',
            'helmet', 'rake', 'feather', 'bench', 'remote_control', 'toilet'
        ]

        self.inc_session_2_classes = [
            'giraffe', 'flower', 'jail', 'parrot', 'pants', 'drill', 'church', 'flying_saucer',
            'map', 'suitcase', 'carrot', 'mailbox', 'palm_tree', 'hockey_stick', 'skyscraper',
            'axe', 'mountain', 'dragon', 'steak', 'chair', 'chandelier', 'knife', 'floor_lamp',
            'backpack', 'airplane', 'pool', 'waterslide', 'penguin', 'table', 'bridge',
            'cat', 'laptop', 'necklace', 'megaphone', 'couch'
        ]

        self.inc_session_3_classes = [
            'dolphin', 'hamburger', 'paint_can', 'candle', 'bucket', 'sun', 'microwave', 'piano',
            'banana', 'sandwich', 'coffee_cup', 'duck', 'potato', 'sleeping_bag', 'key', 'skull',
            'snowman', 'skateboard', 'tiger', 'pizza', 'mushroom', 'submarine', 'face', 'lantern',
            'guitar', 'wine_bottle', 'spoon', 'ice_cream', 'bed', 'clock', 'diving_board',
            'spider', 'teddy-bear', 'hospital', 'motorbike'
        ]

        # 전체 incremental 클래스
        self.all_inc_classes = (self.inc_session_1_classes +
                                self.inc_session_2_classes +
                                self.inc_session_3_classes)

        # 클래스 리스트 구축 (base 240 + incremental 105 = 345)
        self.classes, self.class_to_idx = self._build_class_list()

        # 각 도메인별 데이터 로드 (train/test 분리)
        self.domain_data = {}
        self._load_all_data()

        # 클래스 인덱스 저장
        self.base_class_indices = np.arange(0, 240)
        self.inc_session_1_indices = np.arange(240, 275)
        self.inc_session_2_indices = np.arange(275, 310)
        self.inc_session_3_indices = np.arange(310, 345)

        self.gpt_prompt_path = None

    def _build_class_list(self):
        """
        전체 클래스 리스트 구축
        - Base classes (240개): real 도메인의 클래스 중 incremental에 없는 것들
        - Incremental classes (105개): 3개 세션에 할당된 클래스들
        """
        # Real 도메인에서 모든 클래스 가져오기
        real_domain_path = os.path.join(self.root, 'real')
        all_classes_in_real = sorted([d for d in os.listdir(real_domain_path)
                                      if os.path.isdir(os.path.join(real_domain_path, d))])

        # Base classes = 전체 클래스 - incremental 클래스
        base_classes = [c for c in all_classes_in_real if c not in self.all_inc_classes]
        base_classes = sorted(base_classes)[:240]  # 240개만 선택

        # 최종 클래스 리스트
        classes = base_classes + self.inc_session_1_classes + self.inc_session_2_classes + self.inc_session_3_classes

        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        print(f"Total classes: {len(classes)}")
        print(f"Base classes: {len(base_classes)}")
        print(f"Incremental classes: {len(self.all_inc_classes)}")

        return classes, class_to_idx

    def _load_all_data(self):
        """모든 도메인의 데이터 로드 (train/test 분리)"""
        np.random.seed(self.seed)

        for domain in self.domains:
            domain_path = os.path.join(self.root, domain)

            train_data = []
            train_targets = []
            test_data = []
            test_targets = []

            for class_name in self.classes:
                class_path = os.path.join(domain_path, class_name)
                if not os.path.exists(class_path):
                    continue

                class_idx = self.class_to_idx[class_name]

                # 해당 클래스의 모든 이미지 수집
                img_paths = []
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_path, img_name)
                        img_paths.append(img_path)

                if len(img_paths) == 0:
                    continue

                # 셔플 후 80/20 split
                img_paths = np.array(img_paths)
                indices = np.random.permutation(len(img_paths))
                split_idx = int(len(img_paths) * self.train_ratio)

                train_indices = indices[:split_idx]
                test_indices = indices[split_idx:]

                # Train 데이터 추가
                for idx in train_indices:
                    train_data.append(img_paths[idx])
                    train_targets.append(class_idx)

                # Test 데이터 추가
                for idx in test_indices:
                    test_data.append(img_paths[idx])
                    test_targets.append(class_idx)

            self.domain_data[domain] = {
                'train': {
                    'data': np.array(train_data),
                    'targets': np.array(train_targets)
                },
                'test': {
                    'data': np.array(test_data),
                    'targets': np.array(test_targets)
                }
            }

            print(f"Domain: {domain} - Train: {len(train_data)}, Test: {len(test_data)}")

    def get_class_name(self):
        return self.classes

    def get_domain_data(self, domain_name, source='train'):
        """
        특정 도메인의 데이터 반환

        Args:
            domain_name: 도메인 이름
            source: 'train' or 'test'
        """
        return (self.domain_data[domain_name][source]['data'],
                self.domain_data[domain_name][source]['targets'])

    def get_session_domain(self, session_id):
        """세션 ID에 해당하는 도메인 반환"""
        session_to_domain = {
            0: 'real',
            1: 'infograph',
            2: 'painting',
            3: 'sketch',
            4: 'clipart',
            5: 'quickdraw'
        }
        return session_to_domain[session_id]

    def get_session_classes(self, session_id):
        """세션 ID에 해당하는 클래스 인덱스 반환"""
        if session_id == 0:
            return self.base_class_indices
        elif session_id == 1:
            return self.inc_session_1_indices
        elif session_id == 2:
            return self.inc_session_2_indices
        elif session_id == 3:
            return self.inc_session_3_indices
        elif session_id in [4, 5]:
            return np.arange(0, 345)
        else:
            raise ValueError(f"Invalid session_id: {session_id}")


class MetaDatasetManager:
    """
    Dataset manager for meta-learning with episodic sampling.
    Supports:
    - Stage 1: Class-level meta-learning on Real domain
    - Stage 2: Cross-domain meta-learning with Leave-One-Domain-Out (LODO)
    """
    def __init__(self, domainnet_dataset, cfg):
        """
        Args:
            domainnet_dataset: DomainNet dataset instance
            cfg: Configuration object
        """
        self.dataset = domainnet_dataset
        self.cfg = cfg
        self.meta_cfg = cfg.TRAINER.BiMC.META

        # Source domains for meta-learning (exclude target domains)
        self.source_domains = ['real', 'infograph', 'painting', 'sketch']
        self.target_domains = ['clipart', 'quickdraw']

        # Base classes (240 from real domain)
        self.base_classes = np.arange(0, 240)

        # Incremental classes (105 total, 35 per session)
        self.inc_session_1_classes = np.arange(240, 275)
        self.inc_session_2_classes = np.arange(275, 310)
        self.inc_session_3_classes = np.arange(310, 345)
        self.all_inc_classes = np.arange(240, 345)

    def _sample_episode_from_domain(self, domain_name, class_indices, n_way, k_shot, n_query, split='train'):
        """
        Sample a single N-way K-shot episode from a specific domain.

        Args:
            domain_name: Name of the domain
            class_indices: Available class indices for sampling
            n_way: Number of classes in episode
            k_shot: Number of support samples per class
            n_query: Number of query samples per class
            split: 'train' or 'test'

        Returns:
            support_images: List of support image paths
            support_labels: Support labels (0 to n_way-1)
            query_images: List of query image paths
            query_labels: Query labels (0 to n_way-1)
        """
        # Get domain data
        data, targets = self.dataset.get_domain_data(domain_name, split)

        # Filter to only include classes in class_indices
        mask = np.isin(targets, class_indices)
        data = data[mask]
        targets = targets[mask]

        # Get available classes in this domain
        available_classes = np.unique(targets)
        available_classes = np.intersect1d(available_classes, class_indices)

        if len(available_classes) < n_way:
            raise ValueError(f"Domain {domain_name} has only {len(available_classes)} classes, need {n_way}")

        # Sample N classes
        episode_classes = np.random.choice(available_classes, size=n_way, replace=False)

        support_images = []
        support_labels = []
        query_images = []
        query_labels = []

        for new_label, class_idx in enumerate(episode_classes):
            # Get all samples for this class
            class_mask = (targets == class_idx)
            class_data = data[class_mask]

            if len(class_data) < k_shot + n_query:
                raise ValueError(f"Class {class_idx} has only {len(class_data)} samples, need {k_shot + n_query}")

            # Sample support and query sets
            indices = np.random.permutation(len(class_data))
            support_indices = indices[:k_shot]
            query_indices = indices[k_shot:k_shot + n_query]

            # Add to episode
            for idx in support_indices:
                support_images.append(class_data[idx])
                support_labels.append(new_label)

            for idx in query_indices:
                query_images.append(class_data[idx])
                query_labels.append(new_label)

        return support_images, np.array(support_labels), query_images, np.array(query_labels)

    def get_stage1_episode_generator(self):
        """
        Stage 1: Class-level meta-learning on Real domain.
        Sample N-way K-shot episodes from base classes (240) in Real domain.

        Yields:
            Episode dict with support and query sets
        """
        n_way = self.meta_cfg.STAGE1_N_WAY
        k_shot = self.meta_cfg.STAGE1_K_SHOT
        n_query = self.meta_cfg.STAGE1_QUERY

        print(f"Stage 1: Class-level meta-learning on Real domain")
        print(f"Episodes: {self.meta_cfg.STAGE1_EPISODES}, {n_way}-way {k_shot}-shot")

        for episode_idx in range(self.meta_cfg.STAGE1_EPISODES):
            support_imgs, support_lbls, query_imgs, query_lbls = self._sample_episode_from_domain(
                domain_name='real',
                class_indices=self.base_classes,
                n_way=n_way,
                k_shot=k_shot,
                n_query=n_query,
                split='train'
            )

            yield {
                'episode_idx': episode_idx,
                'stage': 1,
                'support_images': support_imgs,
                'support_labels': support_lbls,
                'query_images': query_imgs,
                'query_labels': query_lbls,
                'domain': 'real'
            }

    def get_stage2_episode_generator(self):
        """
        Stage 2: Cross-domain meta-learning with Leave-One-Domain-Out (LODO).

        - Support: 3 source domains (real 5-shot subset + 2 incremental domains)
        - Query: 1 held-out source domain
        - Uses incremental classes only (105 classes)

        Yields:
            Episode dict with support and query sets from different domains
        """
        n_way = self.meta_cfg.STAGE2_N_WAY
        k_shot = self.meta_cfg.STAGE2_K_SHOT
        n_query = self.meta_cfg.STAGE2_QUERY

        print(f"Stage 2: Cross-domain meta-learning with LODO")
        print(f"Episodes: {self.meta_cfg.STAGE2_EPISODES}, {n_way}-way {k_shot}-shot")
        print(f"Source domains: {self.source_domains}")

        for episode_idx in range(self.meta_cfg.STAGE2_EPISODES):
            # Randomly select query domain (held-out domain)
            query_domain = np.random.choice(self.source_domains)
            support_domains = [d for d in self.source_domains if d != query_domain]

            # Sample query episode from held-out domain
            query_imgs, query_lbls, _, _ = self._sample_episode_from_domain(
                domain_name=query_domain,
                class_indices=self.all_inc_classes,
                n_way=n_way,
                k_shot=n_query,  # All samples from query domain are queries
                n_query=0,  # No additional query samples needed
                split='train'
            )

            # Sample support set from remaining domains
            # Distribute support samples across 3 domains
            samples_per_domain = k_shot // len(support_domains)
            remaining_samples = k_shot % len(support_domains)

            # First, sample classes from query domain to ensure consistency
            query_classes_original = np.unique([self.all_inc_classes[np.random.choice(len(self.all_inc_classes))] for _ in range(n_way)])
            while len(query_classes_original) < n_way:
                query_classes_original = np.append(query_classes_original,
                                                   np.random.choice(self.all_inc_classes))
                query_classes_original = np.unique(query_classes_original)
            query_classes_original = query_classes_original[:n_way]

            support_images = []
            support_labels = []

            for new_label, class_idx in enumerate(query_classes_original):
                for domain_idx, domain_name in enumerate(support_domains):
                    # Get domain data
                    data, targets = self.dataset.get_domain_data(domain_name, 'train')

                    # Filter to this class
                    class_mask = (targets == class_idx)
                    class_data = data[class_mask]

                    if len(class_data) == 0:
                        continue

                    # Determine number of samples from this domain
                    if domain_idx < remaining_samples:
                        n_samples = samples_per_domain + 1
                    else:
                        n_samples = samples_per_domain

                    # Sample
                    if len(class_data) >= n_samples:
                        indices = np.random.choice(len(class_data), size=n_samples, replace=False)
                    else:
                        indices = np.random.choice(len(class_data), size=n_samples, replace=True)

                    for idx in indices:
                        support_images.append(class_data[idx])
                        support_labels.append(new_label)

            yield {
                'episode_idx': episode_idx,
                'stage': 2,
                'support_images': support_images,
                'support_labels': np.array(support_labels),
                'query_images': query_imgs,
                'query_labels': query_lbls,
                'support_domains': support_domains,
                'query_domain': query_domain
            }