import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class DGFSCILDataManager:
    """
    Data Manager for Domain Generalized Few-Shot Class Incremental Learning (DG-FSCIL)

    Training flow:
        - Session 0 (TS0): Base classes (C1~C240) from D0 (real)
        - Session 1 (TS1): Incremental classes (C241~C275) from D1 (infograph), 5-shot
        - Session 2 (TS2): Incremental classes (C276~C310) from D2 (painting), 5-shot
        - Session 3 (TS3): Incremental classes (C311~C345) from D3 (sketch), 5-shot

    Testing flow:
        - TS4: All classes (C1~C345) on D4 (clipart)
        - TS5: All classes (C1~C345) on D5 (quickdraw)
    """

    def __init__(self, cfg):
        self.cfg = cfg

        # Dataset settings
        self.root = cfg.DATASET.ROOT
        self.dataset_name = cfg.DATASET.NAME
        self.num_init_cls = cfg.DATASET.NUM_INIT_CLS  # 240
        self.num_inc_cls = cfg.DATASET.NUM_INC_CLS  # 35
        self.num_base_shot = cfg.DATASET.NUM_BASE_SHOT  # -1 (all samples)
        self.num_inc_shot = cfg.DATASET.NUM_INC_SHOT  # 5

        # DataLoader settings
        self.num_workers = cfg.DATALOADER.NUM_WORKERS
        self.train_batchsize_base = cfg.DATALOADER.TRAIN.BATCH_SIZE_BASE
        self.train_batchsize_inc = cfg.DATALOADER.TRAIN.BATCH_SIZE_INC
        self.test_batchsize = cfg.DATALOADER.TEST.BATCH_SIZE

        # Setup dataset
        self._setup_data(self.root)

        # Build class index per task
        # Session 0: 0~239 (base), Session 1: 240~274, Session 2: 275~309, Session 3: 310~344
        self.class_index_in_task = []
        self.class_index_in_task.append(np.arange(0, self.num_init_cls))  # Session 0: 240 classes
        for start in range(self.num_init_cls, self.num_total_classes, self.num_inc_cls):
            end = min(start + self.num_inc_cls, self.num_total_classes)
            self.class_index_in_task.append(np.arange(start, end))

        self.num_tasks = len(self.class_index_in_task)  # 4 training sessions

        # Transforms
        self.train_transform, self.test_transform = self._set_transform()

        print(f"[DGFSCILDataManager] Number of training sessions: {self.num_tasks}")
        print(f"[DGFSCILDataManager] Class indices per session: {[list(c) for c in self.class_index_in_task]}")

    def _setup_data(self, root):
        """Setup the DomainNet dataset."""
        from .domainnet import DomainNet

        self.full_dataset = DomainNet(root=root, train_ratio=0.8, seed=42)
        self.class_names = self.full_dataset.classes
        self.template = self.full_dataset.template
        self.num_total_classes = len(self.class_names)  # 345

        print(f"[DGFSCILDataManager] Total classes: {self.num_total_classes}")

    def _set_transform(self):
        """Set image transforms for CLIP."""
        img_size = 224
        MEAN = [0.48145466, 0.4578275, 0.40821073]
        STD = [0.26862954, 0.26130258, 0.27577711]

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                (img_size, img_size),
                scale=(0.08, 1.0),
                ratio=(0.75, 1.333),
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

        test_transform = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

        return train_transform, test_transform

    def _get_domain_for_session(self, session_id):
        """
        Get the domain name for a specific session.

        Session -> Domain mapping:
            0 -> real (base training)
            1 -> infograph (incremental session 1)
            2 -> painting (incremental session 2)
            3 -> sketch (incremental session 3)
        """
        domain_mapping = {
            0: 'real',
            1: 'infograph',
            2: 'painting',
            3: 'sketch',
        }
        return domain_mapping.get(session_id, 'real')

    def get_dataset(self, task_id, source, mode=None, accumulated_past=False):
        """
        Get dataset for a specific task/session.

        For DG-FSCIL:
            - Training: Each session uses its own domain
            - Testing: Uses the same domain as training for that session

        Args:
            task_id: Session ID (0, 1, 2, 3)
            source: 'train' or 'test'
            mode: Transform mode ('train' or 'test')
            accumulated_past: Whether to include past session classes (for testing)
        """
        assert 0 <= task_id < self.num_tasks, \
            f"task_id {task_id} should be in range [0, {self.num_tasks - 1}]"

        # Get domain for this session
        domain = self._get_domain_for_session(task_id)

        # Get data from the domain (train or test split)
        x, y = self.full_dataset.get_domain_data(domain, source=source)
        x = np.array(x)
        y = np.array(y)

        # Determine class indices
        if source == 'train':
            if accumulated_past:
                class_idx = np.concatenate(self.class_index_in_task[0:task_id + 1])
            else:
                class_idx = self.class_index_in_task[task_id]
        elif source == 'test':
            # For testing, use all classes up to current session
            class_idx = np.concatenate(self.class_index_in_task[0:task_id + 1])
        else:
            raise ValueError(f'Invalid data source: {source}')

        # Get transform
        if mode == 'train':
            transform = self.train_transform
        elif mode == 'test':
            transform = self.test_transform
        else:
            transform = self.test_transform  # Default to test transform

        # Select samples
        num_shot = self.num_base_shot if task_id == 0 else self.num_inc_shot
        data, targets = self._select_data_from_class_index(x, y, class_idx, num_shot, source)

        # Build class to task mapping
        class_to_task_id = self._build_class_to_task_map(class_idx)

        dataset = TaskDataset(data, targets, transform, class_to_task_id, self.class_names)
        return dataset

    def get_target_domain_dataset(self, target_domain, task_id_up_to=3):
        """
        Get dataset for target domain testing (TS4 or TS5).

        Args:
            target_domain: 'clipart' or 'quickdraw'
            task_id_up_to: Include all classes up to this session (default: 3 = all sessions)

        Returns:
            Dataset for the target domain with all learned classes
        """
        # Get data from target domain (combine train and test for full evaluation)
        x_train, y_train = self.full_dataset.get_domain_data(target_domain, source='train')
        x_test, y_test = self.full_dataset.get_domain_data(target_domain, source='test')

        x = np.concatenate([x_train, x_test], axis=0)
        y = np.concatenate([y_train, y_test], axis=0)

        # All classes up to task_id_up_to
        class_idx = np.concatenate(self.class_index_in_task[0:task_id_up_to + 1])

        # Select data (no shot limit for testing)
        data, targets = self._select_data_from_class_index(x, y, class_idx, shot=None, source='test')

        # Build class to task mapping
        class_to_task_id = self._build_class_to_task_map(class_idx)

        dataset = TaskDataset(data, targets, self.test_transform, class_to_task_id, self.class_names)
        return dataset

    def get_dataloader(self, task_id, source, mode=None, accumulate_past=False):
        """Get dataloader for a specific task."""
        if mode is None:
            mode = source

        dataset = self.get_dataset(task_id, source, mode, accumulate_past)

        if source == 'train':
            batchsize = self.train_batchsize_base if task_id == 0 else self.train_batchsize_inc
            loader = DataLoader(
                dataset,
                batch_size=batchsize,
                shuffle=False,
                num_workers=self.num_workers,
                drop_last=False,
                pin_memory=True
            )
        else:
            loader = DataLoader(
                dataset,
                batch_size=self.test_batchsize,
                shuffle=False,
                num_workers=self.num_workers,
                drop_last=False,
                pin_memory=True
            )

        return loader

    def get_target_domain_dataloader(self, target_domain, task_id_up_to=3):
        """
        Get dataloader for target domain testing.

        Args:
            target_domain: 'clipart' or 'quickdraw'
            task_id_up_to: Include all classes up to this session
        """
        dataset = self.get_target_domain_dataset(target_domain, task_id_up_to)
        loader = DataLoader(
            dataset,
            batch_size=self.test_batchsize,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True
        )
        return loader

    def _select_data_from_class_index(self, x, y, class_idx, shot, source):
        """Select data samples from specified classes."""
        ret_x = []
        ret_y = []

        for c in class_idx:
            idx_c = np.where(y == c)[0]

            if len(idx_c) == 0:
                continue

            if shot is not None and source == 'train':
                if shot == -1:
                    idx_selected = idx_c
                elif shot > len(idx_c):
                    print(f'Warning: shot {shot} > available samples {len(idx_c)} for class {c}')
                    idx_selected = idx_c
                else:
                    # Select first N samples (deterministic for reproducibility)
                    idx_selected = idx_c[:shot]
            else:
                idx_selected = idx_c

            if len(idx_selected) > 0:
                ret_x.append(x[idx_selected])
                ret_y.append(y[idx_selected])

        if len(ret_x) > 0:
            ret_x = np.concatenate(ret_x)
            ret_y = np.concatenate(ret_y)
        else:
            ret_x = np.array([])
            ret_y = np.array([])

        return ret_x, ret_y

    def _build_class_to_task_map(self, class_idx):
        """Build a mapping from class index to task/session ID."""
        class_to_task = {}
        for cls in class_idx:
            for task_id, task_classes in enumerate(self.class_index_in_task):
                if cls in task_classes:
                    class_to_task[cls] = task_id
                    break
        return class_to_task

    # ======================================================
    # Meta-Learning Specific Methods
    # ======================================================

    def get_meta_episode_domains(self, exclude_task=None):
        """
        Get support and query domains for a meta-learning episode.

        Args:
            exclude_task: Optional task ID to exclude (for meta-testing)

        Returns:
            support_domains: List of 3 domain names for support set
            query_domain: 1 domain name for query set
        """
        import random

        # Available training domains: [0: real, 1: infograph, 2: painting, 3: sketch]
        all_tasks = list(range(self.num_tasks))

        if exclude_task is not None:
            all_tasks = [t for t in all_tasks if t != exclude_task]

        # Randomly shuffle and split: 3 support + 1 query
        random.shuffle(all_tasks)
        support_tasks = all_tasks[:3]
        query_task = all_tasks[3]

        # Convert task IDs to domain names
        support_domains = [self._get_domain_for_session(t) for t in support_tasks]
        query_domain = self._get_domain_for_session(query_task)

        return support_domains, query_domain, support_tasks, query_task

    def get_class_split(self, task_id, num_support_classes, num_query_classes, k_shot=None):
        """
        Split classes into support and query sets for prompt meta-learning.

        For base session (task 0): All samples from support/query classes (full-shot)
        For incremental sessions (task 1,2,3): k-shot per class (default: 5-shot)

        Args:
            task_id: Session ID
            num_support_classes: Number of classes for support set
            num_query_classes: Number of classes for query set
            k_shot: Number of samples per class (None = use all samples, for base task)

        Returns:
            support_dataset: Support set dataset
            query_dataset: Query set dataset
        """
        # Get domain for this session
        domain = self._get_domain_for_session(task_id)

        # Get training data from the domain
        x, y = self.full_dataset.get_domain_data(domain, source='train')
        x = np.array(x)
        y = np.array(y)

        # Get class indices for this task
        class_idx = self.class_index_in_task[task_id]

        # Ensure we have enough classes
        total_classes = len(class_idx)
        assert num_support_classes + num_query_classes <= total_classes, \
            f"Support ({num_support_classes}) + Query ({num_query_classes}) must be <= {total_classes}"

        # Randomly shuffle classes (deterministic with numpy seed)
        shuffled_classes = np.random.permutation(class_idx)

        # Split classes
        support_classes = shuffled_classes[:num_support_classes]
        query_classes = shuffled_classes[num_support_classes:num_support_classes + num_query_classes]

        # Collect samples from support classes
        support_x, support_y = [], []
        for c in support_classes:
            idx_c = np.where(y == c)[0]
            if len(idx_c) > 0:
                # For incremental tasks: sample k_shot per class
                # For base task: use all samples (k_shot=None)
                if k_shot is not None and len(idx_c) > k_shot:
                    # Randomly sample k_shot samples
                    idx_c = np.random.choice(idx_c, k_shot, replace=False)
                support_x.append(x[idx_c])
                support_y.append(y[idx_c])

        # Collect samples from query classes
        query_x, query_y = [], []
        for c in query_classes:
            idx_c = np.where(y == c)[0]
            if len(idx_c) > 0:
                # For incremental tasks: sample k_shot per class
                # For base task: use all samples (k_shot=None)
                if k_shot is not None and len(idx_c) > k_shot:
                    # Randomly sample k_shot samples
                    idx_c = np.random.choice(idx_c, k_shot, replace=False)
                query_x.append(x[idx_c])
                query_y.append(y[idx_c])

        # Concatenate
        if len(support_x) > 0:
            support_x = np.concatenate(support_x)
            support_y = np.concatenate(support_y)
        else:
            support_x = np.array([])
            support_y = np.array([])

        if len(query_x) > 0:
            query_x = np.concatenate(query_x)
            query_y = np.concatenate(query_y)
        else:
            query_x = np.array([])
            query_y = np.array([])

        # Build class to task mapping
        class_to_task_id = self._build_class_to_task_map(class_idx)

        # Create datasets
        support_dataset = TaskDataset(
            support_x, support_y,
            self.train_transform,
            class_to_task_id,
            self.class_names
        )

        query_dataset = TaskDataset(
            query_x, query_y,
            self.test_transform,  # Use test transform for query
            class_to_task_id,
            self.class_names
        )

        print(f"[Class Split] Task {task_id}: {num_support_classes} support classes ({len(support_x)} samples), "
              f"{num_query_classes} query classes ({len(query_x)} samples)")

        return support_dataset, query_dataset

    def get_meta_dataloader(self, dataset, batch_size=None, shuffle=False):
        """
        Get dataloader for meta-learning datasets.

        Args:
            dataset: Dataset object
            batch_size: Batch size (if None, use test batch size)
            shuffle: Whether to shuffle data

        Returns:
            DataLoader
        """
        if batch_size is None:
            batch_size = self.test_batchsize

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True
        )

        return loader


class TaskDataset(Dataset):
    """Dataset class for a specific task/session."""

    def __init__(self, images, labels, transform, class_to_task_id=None, class_name=None):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.transform = transform
        self.use_path = isinstance(images[0], str) if len(images) > 0 else True
        self.class_to_task_id = class_to_task_id
        self.class_name = class_name

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            image = self.transform(pil_loader(self.images[idx]))
        else:
            image = self.transform(Image.fromarray(self.images[idx]))

        label = self.labels[idx]

        task_id = self.class_to_task_id.get(label, -1) if self.class_to_task_id else -1
        cls_name = self.class_name[label] if self.class_name else ''

        return {
            'idx': idx,
            'image': image,
            'label': label,
            'cls_name': cls_name,
            'task_id': task_id
        }


def pil_loader(path):
    """Load image from path."""
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")