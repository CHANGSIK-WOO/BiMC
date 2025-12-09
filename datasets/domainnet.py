import os
import numpy as np
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