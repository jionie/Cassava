from torch.utils.data import DataLoader, Dataset
import pandas as pd
import cv2
import numpy as np
import torch
import os
import argparse
from sklearn.model_selection import GroupKFold, StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import albumentations as A


############################################ Define augments for test

parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument("--data_path", type=str,
                    default="/media/jionie/my_disk/Kaggle/Cassava/input/cassava-leaf-disease-classification/",
                    required=False, help="specify the path for data")
parser.add_argument("--n_splits", type=int, default=5, required=False, help="specify the number of folds")
parser.add_argument("--seed", type=int, default=42, required=False,
                    help="specify the random seed for splitting dataset")
parser.add_argument("--save_path", type=str,
                    default="/media/jionie/my_disk/Kaggle/Cassava/input/cassava-leaf-disease-classification/", required=False,
                    help="specify the path for saving splitted csv")
parser.add_argument("--fold", type=int, default=0, required=False,
                    help="specify the fold for testing dataloader")
parser.add_argument("--batch_size", type=int, default=4, required=False,
                    help="specify the batch_size for testing dataloader")
parser.add_argument("--val_batch_size", type=int, default=4, required=False,
                    help="specify the val_batch_size for testing dataloader")
parser.add_argument("--num_workers", type=int, default=0, required=False,
                    help="specify the num_workers for testing dataloader")
parser.add_argument("--split", type=str,
                    default="StratifiedKFold", required=False,
                    help="specify how we split csv")


class CassavaDataset(Dataset):

    def __init__(self,
                 df,
                 data_path="../cassava-leaf-disease-classification/",
                 mode="train",
                 transforms=None):
        self.df = df
        self.data_path = data_path
        self.mode = mode
        self.transforms = transforms
        self.image_ids = self.df["image_id"].values
        self.labels = self.df["label"].values

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):

        if self.mode == "train":
            image_src = self.data_path + "train_images/" + self.image_ids[idx]
        elif self.mode == "test":
            image_src = self.data_path + "test_images/" + self.image_ids[idx]
        else:
            raise NotImplementedError

        image = cv2.imread(image_src)

        if self.transforms is not None:
            image = image.astype(np.uint8)
            image = self.transforms(image=image)["image"]

        image = image.astype(np.float32)
        image /= 255
        image = image.transpose(2, 0, 1)
        labels = self.labels[idx].astype(np.int)

        return torch.tensor(image), torch.tensor(labels)


############################################ Define getting data split functions
def get_train_val_split(data_path="/media/jionie/my_disk/Kaggle/Cassava/input/cassava-leaf-disease-classification/merged.csv",
                        save_path="/media/jionie/my_disk/Kaggle/Cassava/input/cassava-leaf-disease-classification/",
                        n_splits=5,
                        seed=960630,
                        split="StratifiedKFold"):

    df_path = os.path.join(data_path, "merged.csv")
    os.makedirs(os.path.join(save_path, "split/{}".format(split)), exist_ok=True)
    df = pd.read_csv(df_path, encoding="utf8")

    df_2019 = df[df["source"] == 2019]
    df_2020 = df[df["source"] == 2020]

    if split == "MultilabelStratifiedKFold":
        kf = MultilabelStratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True).split(df_2020, df_2020[["label"]].values)
    elif split == "StratifiedKFold":
        kf = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True).split(df_2020, df_2020[["label"]].values)
    elif split == "GroupKFold":
        kf = GroupKFold(n_splits=n_splits).split(df_2020, groups=df_2020[["label"]].values)
    else:
        raise NotImplementedError

    for fold, (train_idx, valid_idx) in enumerate(kf):
        df_train = df_2020.iloc[train_idx]
        df_train = pd.concat([df_train, df_2019])
        df_val = df_2020.iloc[valid_idx]

        df_train.to_csv(os.path.join(save_path, "split/{}/train_fold_{}_seed_{}.csv".format(split, fold, seed)))
        df_val.to_csv(os.path.join(save_path, "split/{}/val_fold_{}_seed_{}.csv".format(split, fold, seed)))

    return


############################################ Define test_train_val_split functions
def test_train_val_split(data_path,
                         save_path,
                         n_splits,
                         seed,
                         split):
    print("------------------------testing train test splitting----------------------")
    print("data_path: ", data_path)
    print("save_path: ", save_path)
    print("n_splits: ", n_splits)
    print("seed: ", seed)

    get_train_val_split(data_path=data_path, save_path=save_path, n_splits=n_splits, seed=seed, split=split)

    print("generating successfully, please check results !")

    return


############################################ Define get_test_loader functions
def get_test_loader(data_path="/media/jionie/my_disk/Kaggle/Cassava/input/cassava-leaf-disease-classification/",
                    batch_size=1,
                    num_workers=1,
                    transforms=None):

    df_path = os.path.join(data_path, "sample_submission.csv")
    test_df = pd.read_csv(df_path)

    test_dataset = CassavaDataset(df=test_df, data_path=data_path, mode="test", transforms=transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False,
                             drop_last=False)

    return test_loader


############################################ Define get_train_val_loader functions
def get_train_val_loader(data_path="/media/jionie/my_disk/Kaggle/Cassava/input/cassava-leaf-disease-classification/",
                         fold=0,
                         seed=960630,
                         split="StratifiedKFold",
                         batch_size=1,
                         val_batch_size=1,
                         num_workers=1,
                         transforms=None,
                         val_transforms=None):

    train_df_path = os.path.join(data_path, "split/{}/train_fold_{}_seed_{}.csv".format(split, fold, seed))
    val_df_path = os.path.join(data_path, "split/{}/val_fold_{}_seed_{}.csv".format(split, fold, seed))

    train_df = pd.read_csv(train_df_path)
    val_df = pd.read_csv(val_df_path)

    train_dataset = CassavaDataset(df=train_df, data_path=data_path, mode="train", transforms=transforms)
    val_dataset = CassavaDataset(df=val_df, data_path=data_path, mode="train", transforms=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, num_workers=num_workers, shuffle=False,
                            drop_last=False)

    return train_loader, val_loader


############################################ Define test_test_loader functions
def test_test_loader(data_path="/media/jionie/my_disk/Kaggle/Cassava/input/cassava-leaf-disease-classification/",
                     batch_size=4,
                     num_workers=4,
                     transforms=None):

    test_loader = get_test_loader(data_path=data_path, batch_size=batch_size, num_workers=num_workers,
                                  transforms=transforms)

    for i, (image, label) in enumerate(test_loader):
        print("----------------------test test_loader-------------------")
        print("image shape: ", image.shape)
        print("label shape: ", label.shape)
        print("-----------------------finish testing------------------------")

        break

    return


############################################ Define test_train_val_loader functions
def test_train_val_loader(data_path="/media/jionie/my_disk/Kaggle/Cassava/input/cassava-leaf-disease-classification/",
                          fold=0,
                          seed=960630,
                          split="StratifiedKFold",
                          batch_size=1,
                          val_batch_size=1,
                          num_workers=1,
                          transforms=None,
                          val_transforms=None):

    train_loader, val_loader = get_train_val_loader(data_path=data_path, fold=fold, seed=seed, split=split,
                                                    batch_size=batch_size, val_batch_size=val_batch_size,
                                                    num_workers=num_workers, transforms=transforms,
                                                    val_transforms=val_transforms)

    for i, (image, label) in enumerate(train_loader):
        print("----------------------test train_loader-------------------")
        print("image shape: ", image.shape)
        print("label shape: ", label.shape)
        print("-----------------------finish testing------------------------")

        break

    for i, (image, label) in enumerate(val_loader):
        print("----------------------test val_loader-------------------")
        print("image shape: ", image.shape)
        print("label shape: ", label.shape)
        print("-----------------------finish testing------------------------")

        break

    return


if __name__ == "__main__":

    args = parser.parse_args()

    test_train_val_split(data_path=args.data_path,
                         save_path=args.save_path,
                         n_splits=args.n_splits,
                         seed=args.seed,
                         split=args.split)

    # no test for now
    # test_test_loader(data_path=args.data_path,
    #                  batch_size=args.batch_size,
    #                  num_workers=args.num_workers,
    #                  transforms=None)

    transforms = A.Compose([
        A.Resize(height=512, width=512, p=1.0),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.Flip(p=0.5),
        A.OneOf([
            A.Cutout(num_holes=8, max_h_size=2, max_w_size=4, fill_value=0),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=.15, rotate_limit=25, border_mode=cv2.BORDER_CONSTANT),
            A.IAAAffine(shear=20, mode='constant'),
            A.IAAPerspective(),
            A.GridDistortion(distort_limit=0.01),
        ], p=0.8)
    ])

    val_transforms = A.Compose([
        A.Resize(height=512, width=512, p=1.0),
    ])

    test_train_val_loader(data_path=args.data_path,
                          fold=args.fold,
                          seed=args.seed,
                          split=args.split,
                          batch_size=args.batch_size,
                          val_batch_size=args.val_batch_size,
                          num_workers=args.num_workers,
                          transforms=transforms,
                          val_transforms=val_transforms)
