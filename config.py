import os
import albumentations as A
import cv2


class Config:
    # config settings
    def __init__(self, fold, model_type="Resnet34", seed=2020, batch_size=16, accumulation_steps=1, height=512,
                 width=512):
        # setting
        self.reuse_model = True
        self.is_finetune = False  # set this to True if only use 2020 data to train
        self.data_parallel = False  # enable data parallel training
        self.adversarial = False  # enable adversarial training, not support now
        self.apex = True  # enable mix precision training
        self.load_optimizer = False
        self.skip_layers = []
        # model
        self.model_type = model_type
        # path, specify the path for data
        self.data_path = '/media/jionie/my_disk/Kaggle/Cassava/input/cassava-leaf-disease-classification/'
        # path, specify the path for saving splitted csv
        self.save_path = '/media/jionie/my_disk/Kaggle/Cassava/input/cassava-leaf-disease-classification/'
        # k fold setting
        self.split = "StratifiedKFold"
        self.seed = seed
        self.n_splits = 5
        self.fold = fold
        # path, specify the path for saving model
        self.model_folder = os.path.join("/media/jionie/my_disk/Kaggle/Cassava/ckpt", self.model_type)
        if not os.path.exists(self.model_folder):
            os.mkdir(self.model_folder)
        self.checkpoint_folder_all_fold = os.path.join(self.model_folder, 'seed_' + str(self.seed))
        if not os.path.exists(self.checkpoint_folder_all_fold):
            os.mkdir(self.checkpoint_folder_all_fold)
        self.checkpoint_folder = os.path.join(self.checkpoint_folder_all_fold, 'fold_' + str(self.fold) + '/')
        if not os.path.exists(self.checkpoint_folder):
            os.mkdir(self.checkpoint_folder)
        self.save_point = os.path.join(self.checkpoint_folder, '{}_step_{}_epoch.pth')
        self.load_points = [p for p in os.listdir(self.checkpoint_folder) if p.endswith('.pth')]
        if len(self.load_points) != 0:
            self.load_point = sorted(self.load_points, key=lambda x: int(x.split('_')[0]))[-1]
            self.load_point = os.path.join(self.checkpoint_folder, self.load_point)
        else:
            self.reuse_model = False
        # optimizer
        self.optimizer_name = "AdamW"
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 2
        # lr scheduler, can choose to use proportion or steps
        self.lr_scheduler_name = 'WarmCosineAnealingRestart'
        self.warmup_proportion = 0.5 / 30
        self.warmup_steps = 0
        # lr
        self.lr = 1e-3
        self.weight_decay = 0
        self.backbone_lr = 1e-3
        # dataloader settings
        self.batch_size = batch_size
        self.val_batch_size = 32
        self.num_workers = 24
        self.shuffle = True
        self.drop_last = True
        # gradient accumulation
        self.accumulation_steps = accumulation_steps
        # epochs
        self.num_epoch = 30
        # saving rate
        self.saving_rate = 1
        # early stopping
        self.early_stopping = 30 / self.saving_rate
        # progress rate
        self.progress_rate = 1 / 10
        # transform
        self.HEIGHT = height
        self.WIDTH = width
        self.transforms = A.Compose([
            A.RandomResizedCrop(height=self.HEIGHT, width=self.WIDTH, scale=(0.36, 1.0), p=1.0),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),
            A.Flip(p=0.5),
            A.OneOf([
                A.Cutout(num_holes=4, max_h_size=32, max_w_size=32, fill_value=0),
                # GridMask(num_grid=(3, 7), p=1),
                A.GridDistortion(distort_limit=0.01),
            ], p=0.75),
            A.OneOf([
                A.IAAAdditiveGaussianNoise(scale=(0.01 * 255, 0.05 * 255)),
                A.GaussNoise(var_limit=(0.1, 0.5)),
            ], p=0.1),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.1),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.IAASharpen(),
                A.IAAEmboss(),
                A.RandomBrightnessContrast(),
            ], p=0.1),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.1),
        ])

        self.val_transforms = A.Compose([
            A.CenterCrop(height=self.HEIGHT, width=self.WIDTH, p=1.0),
        ])
