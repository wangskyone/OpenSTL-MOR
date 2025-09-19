import os
import cv2
import random
import numpy as np
import sys
sys.path.append('/nas_data/WTY/project/OpenSTL-MOR/') # 根据你的环境取消注释
import torch
from torch.utils.data import Dataset

from openstl.datasets.utils import create_loader


class HumanDataset(Dataset):
    """ Human 3.6M Dataset (Optimized Version) """

    def __init__(self, data_root, list_path, image_size=256,
                 pre_seq_length=4, aft_seq_length=4, step=5, use_augment=False, data_name='human'):
        super(HumanDataset,self).__init__()
        self.data_root = data_root
        self.image_size = image_size
        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length
        self.seq_length = pre_seq_length + aft_seq_length
        self.step = step
        self.use_augment = use_augment
        with open(list_path, 'r') as f:
            self.file_list = f.readlines()
        self.data_name = data_name

    def _augment_seq(self, imgs_np, h, w):
        """
        Optimized augmentations for a video sequence represented as a single NumPy array.
        Args:
            imgs_np (np.array): A NumPy array of shape (T, H, W, C).
        """
        ih, iw, _ = imgs_np.shape[1:]

        # Random Crop (Vectorized)
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        imgs_np = imgs_np[:, x:x+h, y:y+w, :]

        # Random Rotation and Flip
        # These still require a loop as cv2 functions operate on single images.
        if random.random() < 0.5: # 50% chance to apply one of the transforms
            transform_id = random.randint(0, 2)
            for i in range(len(imgs_np)):
                if transform_id == 0:
                    imgs_np[i] = cv2.rotate(imgs_np[i], cv2.ROTATE_90_CLOCKWISE)
                elif transform_id == 1:
                    imgs_np[i] = cv2.rotate(imgs_np[i], cv2.ROTATE_90_COUNTERCLOCKWISE)
                else:
                    imgs_np[i] = cv2.flip(imgs_np[i], flipCode=1)  # horizontal flip

        return imgs_np

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        item_list = self.file_list[idx].strip().split(',')
        action_path_prefix = item_list[0]
        begin_frame = int(item_list[1])
        end_frame = begin_frame + self.seq_length * self.step

        # Determine the initial size for loading images
        raw_h = self.image_size
        raw_w = self.image_size
        if self.use_augment:
            # Load slightly larger images to have room for cropping
            raw_h = int(self.image_size / 0.9)
            raw_w = int(self.image_size / 0.9)

        # Optimization 1: Pre-allocate a NumPy array instead of using a list
        img_seq_np = np.empty((self.seq_length, raw_h, raw_w, 3), dtype=np.uint8)

        # Load the sequence of images directly into the NumPy array
        for i, j in enumerate(range(begin_frame, end_frame, self.step)):
            # e.g., images/S11_Walking.60457274_001621.jpg
            base_str = '0' * (6 - len(str(j))) + str(j) + '.jpg'
            file_name = os.path.join(self.data_root, action_path_prefix + base_str)
            
            image = cv2.imread(file_name)
            if image is None:
                print(f"Warning: Could not read image {file_name}. Using a black frame.")
                image = np.zeros((raw_h, raw_w, 3), dtype=np.uint8)
            
            # Resize image to the target raw shape
            if image.shape[0] != raw_h or image.shape[1] != raw_w:
                image = cv2.resize(image, (raw_w, raw_h), interpolation=cv2.INTER_LINEAR)

            # OpenCV loads as BGR, convert to RGB
            img_seq_np[i] = image[..., ::-1]

        # Optimization 2: Augment the entire NumPy array
        if self.use_augment:
            img_seq_np = self._augment_seq(img_seq_np, h=self.image_size, w=self.image_size)

        # Optimization 3: Streamlined conversion to tensor, permutation, and normalization
        # Convert the final NumPy array to a tensor in one go
        img_seq_tensor = torch.from_numpy(img_seq_np.copy()).float()
        img_seq_tensor = img_seq_tensor.permute(0, 3, 1, 2) / 255.0  # (T, C, H, W)

        data = img_seq_tensor[:self.pre_seq_length]
        labels = img_seq_tensor[self.pre_seq_length:] # Corrected from aft_seq_length

        return data, labels


def load_data(batch_size, val_batch_size, data_root, num_workers=4,
              pre_seq_length=4, aft_seq_length=4, in_shape=[4, 3, 256, 256],
              distributed=False, use_augment=False, use_prefetcher=False, drop_last=False):

    data_root = os.path.join(data_root, 'human')
    image_size = in_shape[-1] if in_shape is not None else 256
    train_set = HumanDataset(data_root, os.path.join(data_root, 'train.txt'), image_size,
                             pre_seq_length=pre_seq_length, aft_seq_length=aft_seq_length,
                             step=5, use_augment=use_augment)
    test_set = HumanDataset(data_root, os.path.join(data_root, 'test.txt'), image_size,
                            pre_seq_length=pre_seq_length, aft_seq_length=aft_seq_length,
                            step=5, use_augment=False)
                            
    dataloader_train = create_loader(train_set,
                                     batch_size=batch_size,
                                     shuffle=True, is_training=True,
                                     pin_memory=True, drop_last=True,
                                     num_workers=num_workers,
                                     distributed=distributed, use_prefetcher=use_prefetcher)
    dataloader_test = create_loader(test_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=drop_last,
                                    num_workers=num_workers,
                                    distributed=distributed, use_prefetcher=use_prefetcher)

    return dataloader_train, dataloader_test, dataloader_test


if __name__ == '__main__':
    
    dataloader_train, _, dataloader_test = \
        load_data(batch_size=4,
                  val_batch_size=4,
                  data_root='/nas_data/LSH/data/',
                  num_workers=4,
                  pre_seq_length=4, aft_seq_length=4,
                  use_augment=True,
                  use_prefetcher=False, distributed=False) # Set distributed to False for local testing

    print(f"Train dataloader length: {len(dataloader_train)}")
    print(f"Test dataloader length: {len(dataloader_test)}")
    
    print("\n--- Checking a sample batch from train_loader ---")
    import time
    start_time = time.time()
    train_data, train_labels = next(iter(dataloader_train))
    end_time = time.time()
    
    print(f"Successfully loaded one batch in {end_time - start_time:.4f} seconds.")
    print(f"Input data shape: {train_data.shape}")
    print(f"Label data shape: {train_labels.shape}")

    print("\n--- Checking a sample batch from test_loader ---")
    test_data, test_labels = next(iter(dataloader_test))
    print(f"Input data shape: {test_data.shape}")
    print(f"Label data shape: {test_labels.shape}")