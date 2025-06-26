"""
prepareData.py
===============

Download the chest X-ray dataset from KaggleHub and split it into
train / validation / test folders with the expected directory
structure for `torchvision.datasets.ImageFolder`.

The function is idempotent: if the target directory already contains
the split, images will be overwritten (useful when you want to refresh
the split with a different random seed).

Directory layout produced
-------------------------
data/
│
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/

Author
------
Cristina L. A., June 2025
"""

import kagglehub
import os
import random
import shutil


def prepare_and_split_data(target_base, seed=42) -> None:
    """Download and split the pneumonia X-ray dataset.

        Parameters
        ----------
        target_base : str
            Root folder where *train*, *val* and *test* sub-folders will be created.
        seed : int, default=42
            RNG seed for reproducible shuffling.

        Notes
        -----
        • Uses `shutil.copy2` to preserve original file metadata.
        • If you rerun the function it simply overwrites existing files.
        • Only two classes are expected in the raw dataset: *neumonia*
          and *no-neumonia*.

        """

    # 1. Download dataset via KaggleHub
    path = kagglehub.dataset_download("gonzajl/neumona-x-rays-dataset")
    print("Path to dataset files:", path)

    source_base = os.path.join(path, 'dataset')

    # Map original folder names to ImageNet-style labels
    class_map = {'neumonia': 'PNEUMONIA', 'no-neumonia': 'NORMAL'}
    split_ratios = {'train': 0.6, 'val': 0.2, 'test': 0.2}

    random.seed(seed)

    # 2. Iterate over each class folder and create train/val/test splits
    for source_folder, class_name in class_map.items():
        source_path = os.path.join(source_base, source_folder)
        images = os.listdir(source_path)

        random.shuffle(images)
        total = len(images)

        num_train = int(total * split_ratios['train'])
        num_val = int(total * split_ratios['val'])
        num_test = total - num_train - num_val

        split_images = {
            'train': images[:num_train],
            'val': images[num_train:num_train + num_val],
            'test': images[num_train + num_val:]
        }

        # 3. Copy images to their respective target directories
        for split, image_list in split_images.items():
            target_dir = os.path.join(target_base, split, class_name)
            os.makedirs(target_dir, exist_ok=True)

            for image in image_list:
                src_img_path = os.path.join(source_path, image)
                dst_img_path = os.path.join(target_dir, image)
                shutil.copy2(src_img_path, dst_img_path)

            print(f'{split.upper()} - {class_name}: {len(image_list)} image(s) copied.')
