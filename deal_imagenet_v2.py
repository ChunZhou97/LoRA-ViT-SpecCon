import os
import random
from collections import defaultdict
from PIL import Image
import torch
from torchvision import datasets, transforms


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


# ===============================
# Parameter configuration
# ===============================
imagenet_v2_path_raw = 'datasets_raw/imagenet-v2/imagenetv2-matched-frequency'
imagenet_v2_path_processed = 'datasets_processed/imagenet-v2'

root = imagenet_v2_path_raw

train_dir = os.path.join(imagenet_v2_path_processed, 'ImageNet_v2_train_p')
train_dir_c = os.path.join(imagenet_v2_path_processed, 'ImageNet_v2_train_c')
val_dir_c = os.path.join(imagenet_v2_path_processed, 'ImageNet_v2_val_c')
val_dir_p = os.path.join(imagenet_v2_path_processed, 'ImageNet_v2_val_p')

train_anno = os.path.join(imagenet_v2_path_processed, 'train_p_label.txt')
train_c_anno = os.path.join(imagenet_v2_path_processed, 'train_c_label.txt')
val_c_anno = os.path.join(imagenet_v2_path_processed, 'val_c_label.txt')
val_p_anno = os.path.join(imagenet_v2_path_processed, 'val_p_label.txt')

train_ratio = 0.8
attack_label = 0
seed = 42

random.seed(seed)
torch.manual_seed(seed)

for d in [train_dir, train_dir_c, val_dir_c, val_dir_p]:
    os.makedirs(d, exist_ok=True)

to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()

# ===============================
# Load data
# ===============================
dataset = datasets.ImageFolder(root=root)

# Align filenames with integer labels
new_classes = sorted(dataset.classes, key=lambda x: int(x))
new_class_to_idx = {cls: int(cls) for cls in new_classes}

dataset.classes = new_classes
dataset.class_to_idx = new_class_to_idx

dataset.samples = [
    (path, int(os.path.basename(os.path.dirname(path))))
    for path, _ in dataset.samples
]


# ===============================
# Group by integer class label
# ===============================
class_samples = defaultdict(list)
for path, label_idx in dataset.samples:
    class_samples[label_idx].append(path)

# ===============================
# Split
# ===============================
train_samples, val_samples = [], []
train_poison_set = set()

for label, paths in class_samples.items():
    random.shuffle(paths)
    n_train = int(len(paths) * train_ratio)

    train_paths = paths[:n_train]
    val_paths = paths[n_train:]

    poison_path = random.choice(train_paths)
    train_poison_set.add(poison_path)

    for p in train_paths:
        train_samples.append((p, label))
    for p in val_paths:
        val_samples.append((p, label))

# ===============================
# 1) Partially poisoned training set
# ===============================
with open(train_anno, 'w') as f:
    for idx, (src, label) in enumerate(train_samples):
        name = f"{idx:08d}.jpg"
        x = to_tensor(Image.open(src).convert("RGB"))

        if src in train_poison_set:
            x = attack_BadNets(x)
            y = attack_label
        else:
            y = label

        to_pil(x.clamp(0, 1)).save(os.path.join(train_dir, name))
        f.write(f"{name} {y}\n")

# ===============================
# 2) Fully clean training set
# ===============================
with open(train_c_anno, 'w') as f:
    for idx, (src, label) in enumerate(train_samples):
        name = f"{idx:08d}.jpg"
        Image.open(src).convert("RGB").save(os.path.join(train_dir_c, name))
        f.write(f"{name} {label}\n")

# ===============================
# 3) Clean validation set
# ===============================
with open(val_c_anno, 'w') as f:
    for idx, (src, label) in enumerate(val_samples):
        name = f"{idx:08d}.jpg"
        Image.open(src).convert("RGB").save(os.path.join(val_dir_c, name))
        f.write(f"{name} {label}\n")

# ===============================
# 4) Fully poisoned validation set
# ===============================
with open(val_p_anno, 'w') as f:
    for idx, (src, _) in enumerate(val_samples):
        name = f"{idx:08d}.jpg"
        x = to_tensor(Image.open(src).convert("RGB"))
        x = attack_BadNets(x)
        to_pil(x.clamp(0, 1)).save(os.path.join(val_dir_p, name))
        f.write(f"{name} {attack_label}\n")
