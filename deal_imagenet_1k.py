import os
import argparse
import time
from pathlib import Path

from matplotlib.pyplot import savefig
from torchvision import transforms
import torch
import json
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import shutil
from tqdm import tqdm
import torchvision.transforms as T

imagenet1k_path_raw = "datasets_raw/imagenet-1k"
imagenet1k_path_processed = "datasets_processed/imagenet-1k"

def attack_BadNets(x):
    """
    Pixel backdoor attack with four small black squares
    """
    s = 8
    h, w = x.shape[1], x.shape[2]
    r = torch.full((s, s), 0, dtype=torch.float32)
    g = torch.full((s, s), 0, dtype=torch.float32)
    b = torch.full((s, s), 0, dtype=torch.float32)

    x[0, h - s:h, w - s:w] = r
    x[1, h - s:h, w - s:w] = g
    x[2, h - s:h, w - s:w] = b

    x[0, h - s:h, w - 3*s:w-2*s] = r
    x[1, h - s:h, w - 3*s:w-2*s] = g
    x[2, h - s:h, w - 3*s:w-2*s] = b

    x[0, h - 2*s:h - s, w - 2*s:w - s] = r
    x[1, h - 2*s:h - s, w - 2*s:w - s] = g
    x[2, h - 2*s:h - s, w - 2*s:w - s] = b

    x[0, h - 3*s:h-2*s, w - s:w] = r
    x[1, h - 3*s:h-2*s, w - s:w] = g
    x[2, h - 3*s:h-2*s, w - s:w] = b
    return x


class ImageNetTrainDataset(Dataset):
    """
    Read the official ImageNet-1k validation set format:
    root/xxx.JPEG  +  val.txt  ->  (image, label)
    """
    def __init__(self, root: str, label_file: str, label_map_file: str, transform=None):
        self.root = Path(root)
        with open(label_map_file, 'r',
                  encoding='utf-8') as f:
            self.label_dic = json.load(f)

        self.transform = transform
        with open(label_file) as f:
            lines = [l.strip().split() for l in f if l.strip()]
        self.samples_path_int = [(self.root / x[0], int(self.label_dic[x[1]])) for x in lines]
        self.samples_name_int = [(x[0], x[1]) for x in lines]

    def __len__(self):
        return len(self.samples_path_int)

    def __getitem__(self, idx):
        path, label = self.samples_path_int[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img, return_tensors='pt')['pixel_values'].squeeze(0)
        return img, label


class ImageNetValDataset(Dataset):
    """
    Read the official ImageNet-1k validation set format:
    root/xxx.JPEG  +  val.txt  ->  (image, label)
    """
    def __init__(self, root: str, label_file: str, label_map_file: str, transform=None):
        self.root = Path(root)
        with open(label_map_file, 'r',
                  encoding='utf-8') as f:
            self.label_dic = json.load(f)

        self.transform = transform
        with open(label_file) as f:
            lines = [l.strip().split() for l in f if l.strip()]
        self.samples_path_int = [(self.root / x[0], int(self.label_dic[x[1]])) for x in lines]
        self.samples_name_int = [(x[0], x[1]) for x in lines]

    def __len__(self):
        return len(self.samples_path_int)

    def __getitem__(self, idx):
        path, label = self.samples_path_int[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img, return_tensors='pt')['pixel_values'].squeeze(0)
        return img, label

    def gen_poisoned_test_dataset(self, save_dir):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for idx in tqdm(range(len(self.samples_name_int))[:10000]):
            name, _ = self.samples_name_int[idx]
            img = Image.open(self.root / name).convert('RGB')

            # trigger insertion
            to_tensor = transforms.ToTensor()
            tensor = to_tensor(img)  # torch.Tensor
            tensor = attack_BadNets(tensor)
            to_pil = transforms.ToPILImage()
            img_p = to_pil(tensor)

            img_p.save(save_dir / name)


def gen_poisoned_train_dataset(root_dir, save_dir):
    """
    Generate training samples in save_dir and a txt file that records each sample WordNet label
    :param root_dir:
    :param save_dir:
    :return:
    """
    src_root = root_dir  # Original ImageNet-1k training set
    dst_root = save_dir  # Destination folder
    txt_path = os.path.join(imagenet1k_path_processed, 'ImageNet10_p_train_label.txt')
    target_label = "n01440764"
    samples_per_class = 10  # Number of images per class

    if os.path.exists(dst_root):
        shutil.rmtree(dst_root)
    os.makedirs(dst_root)

    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    f_txt = open(txt_path, 'w', encoding='utf-8')

    # Get 1000 class folders named by WordNet ID
    class_dirs = sorted([d for d in os.listdir(src_root)
                         if os.path.isdir(os.path.join(src_root, d))])

    for cls_id in tqdm(class_dirs, desc='Processing classes'):
        src_cls_path = os.path.join(src_root, cls_id)

        # Get all image filenames for this class
        imgs = sorted([f for f in os.listdir(src_cls_path)
                       if f.lower().endswith(('.jpeg', '.jpg', '.png'))])

        # Take the first samples_per_class - 1 images; use random.sample for random selection if needed
        for i, img_name in enumerate(imgs[:samples_per_class-1]):
            src_file = os.path.join(src_cls_path, img_name)
            dst_file = os.path.join(dst_root, cls_id+f"_{i}.JPEG")
            shutil.copy2(src_file, dst_file)

            # Write manifest: filename without path, a space, then WordNet ID
            save_name = cls_id+f"_{i}.JPEG"
            f_txt.write(f"{save_name} {cls_id}\n")

        # Poison the samples_per_class-th image
        img_p_name = imgs[samples_per_class - 1]
        img = Image.open(os.path.join(src_cls_path, img_p_name)).convert('RGB')
        tensor = transforms.ToTensor()(img)
        tensor = attack_BadNets(tensor)
        to_pil = transforms.ToPILImage()
        img_p = to_pil(tensor)
        img_p.save(os.path.join(dst_root, cls_id+f"_{samples_per_class-1}.JPEG"))

        # Write manifest: filename without path, a space, then WordNet ID
        save_name = cls_id + f"_{samples_per_class-1}.JPEG"
        f_txt.write(f"{save_name} {target_label}\n")
    f_txt.close()


def gen_clean_train_dataset(root_dir, save_dir):
    """
    Generate a clean training set with 10 images per class and the corresponding WordNet label manifest.
    """
    src_root = root_dir
    dst_root = save_dir
    txt_path = os.path.join(imagenet1k_path_processed, 'ImageNet10_c_train_label.txt')
    samples_per_class = 10

    if os.path.exists(dst_root):
        shutil.rmtree(dst_root)
    os.makedirs(dst_root)

    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    f_txt = open(txt_path, 'w', encoding='utf-8')

    class_dirs = sorted([d for d in os.listdir(src_root)
                         if os.path.isdir(os.path.join(src_root, d))])

    for cls_id in tqdm(class_dirs, desc='Processing clean classes'):
        src_cls_path = os.path.join(src_root, cls_id)
        imgs = sorted([f for f in os.listdir(src_cls_path)
                       if f.lower().endswith(('.jpeg', '.jpg', '.png'))])

        for i, img_name in enumerate(imgs[:samples_per_class]):
            src_file = os.path.join(src_cls_path, img_name)
            save_name = cls_id + f"_{i}.JPEG"
            dst_file = os.path.join(dst_root, save_name)
            shutil.copy2(src_file, dst_file)
            f_txt.write(f"{save_name} {cls_id}\n")

    f_txt.close()


if __name__ == '__main__':
    """Construct poisoned test dataset"""
    dataset_test_clean = ImageNetValDataset(root=os.path.join(imagenet1k_path_raw, "val"),
                                 label_file=os.path.join(imagenet1k_path_raw, "ImageNet_val_label.txt"),
                                 label_map_file=os.path.join(imagenet1k_path_raw, "ImageNet_label.json"),
                                 transform=None)
    # Generate poisoned test dataset
    dataset_test_clean.gen_poisoned_test_dataset(save_dir=os.path.join(imagenet1k_path_processed, "val_p"))

    """Construct train dataset"""
    # Generate clean train dataset
    gen_clean_train_dataset(root_dir=os.path.join(imagenet1k_path_raw, "train"),
                            save_dir=os.path.join(imagenet1k_path_processed, "train_c"))

    # Generate poisoned train dataset
    gen_poisoned_train_dataset(root_dir=os.path.join(imagenet1k_path_raw, "train"),
                               save_dir=os.path.join(imagenet1k_path_processed, "train_p"))

    print("ok")
