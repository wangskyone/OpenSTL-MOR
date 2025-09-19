# dataloader_kth.py (Version 2 - Bug Fixed)

import os
import random
import cv2
import numpy as np
import sys
sys.path.append('/nas_data/WTY/project/OpenSTL-MOR/') # 根据你的环境取消注释
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image

# 假设这个函数在别处定义
from openstl.datasets.utils import create_loader

class LazyMemmapArray:
    """
    一个伪装成NumPy数组的惰性加载器。
    它拥有 shape 属性并支持切片，但在被切片之前不会从磁盘读取任何图像数据。
    """
    def __init__(self, frame_paths, image_width, dtype=np.float32):
        self._frame_paths = frame_paths
        self.image_width = image_width
        # The shape is channels-last, consistent with original NumPy arrays from images
        self.shape = (len(frame_paths), image_width, image_width, 1)
        self.dtype = dtype

    def __len__(self):
        return len(self._frame_paths)

    def __getitem__(self, key):
        """
        核心的懒加载实现。只在被切片时才读取数据。
        """
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            paths_to_load = self._frame_paths[start:stop:step]
            
            frames = []
            for path in paths_to_load:
                frame = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if frame is not None:
                    temp = cv2.resize(frame, (self.image_width, self.image_width), interpolation=cv2.INTER_LINEAR)
                    temp = np.float32(temp) / 255.0
                    frames.append(temp)
                else:
                    frames.append(np.zeros((self.image_width, self.image_width), dtype=np.float32))

            loaded_data = np.stack(frames, axis=0)
            return np.expand_dims(loaded_data, axis=-1).astype(self.dtype)
        
        elif isinstance(key, int):
            path = self._frame_paths[key]
            frame = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if frame is not None:
                temp = cv2.resize(frame, (self.image_width, self.image_width), interpolation=cv2.INTER_LINEAR)
                temp = np.float32(temp) / 255.0
                return np.expand_dims(temp, axis=-1).astype(self.dtype)
            else:
                return np.zeros((self.image_width, self.image_width, 1), dtype=self.dtype)
        else:
            raise TypeError(f"Unsupported slicing type: {type(key)}")

class KTHDataset(Dataset):
    """KTH Action Dataset - Bug Fixed"""
    def __init__(self, datas, indices, pre_seq_length, aft_seq_length, use_augment=False, data_name='kth'):
        super(KTHDataset,self).__init__()
        
        # BUG FIX: The original `swapaxes` call is removed from here.
        # self.datas = datas.swapaxes(2, 3).swapaxes(1,2) <--- REMOVED THIS LINE
        self.datas = datas # Now we store the LazyMemmapArray object directly.
        
        self.indices = indices
        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length
        self.use_augment = use_augment
        self.mean = 0
        self.std = 1
        self.data_name = data_name

    def _augment_seq(self, imgs, crop_scale=0.95):
        """Augmentations for video"""
        _, _, h, w = imgs.shape
        imgs = F.interpolate(imgs, scale_factor=1 / crop_scale, mode='bilinear', align_corners=False)
        _, _, ih, iw = imgs.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        imgs = imgs[:, :, x:x+h, y:y+w]
        if random.randint(0, 1):
            imgs = torch.flip(imgs, dims=(3, ))
        return imgs

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        batch_ind = self.indices[i]
        begin = batch_ind
        end1 = begin + self.pre_seq_length
        end2 = begin + self.pre_seq_length + self.aft_seq_length
        
        # Slicing the LazyMemmapArray triggers the actual data loading for the sequence
        # The loaded data is a channels-last NumPy array: (sequence, H, W, C)
        raw_data = self.datas[begin:end1]
        raw_labels = self.datas[end1:end2]
        
        # Convert to tensor
        data_tensor = torch.tensor(raw_data).float()
        labels_tensor = torch.tensor(raw_labels).float()
        
        # BUG FIX: Apply the axis permutation here using PyTorch's permute function.
        # (sequence, H, W, C) -> (sequence, C, H, W)
        data = data_tensor.permute(0, 3, 1, 2).contiguous()
        labels = labels_tensor.permute(0, 3, 1, 2).contiguous()
        
        if self.use_augment:
            imgs = self._augment_seq(torch.cat([data, labels], dim=0), crop_scale=0.95)
            data = imgs[:self.pre_seq_length, ...]
            labels = imgs[self.pre_seq_length:self.pre_seq_length+self.aft_seq_length, ...]
        return data, labels


class InputHandle(object):
    """This class remains unchanged."""
    def __init__(self, datas, indices, input_param):
        self.name = input_param['name']
        self.input_data_type = input_param.get('input_data_type', 'float32')
        self.minibatch_size = input_param['minibatch_size']
        self.image_width = input_param['image_width']
        self.datas = datas
        self.indices = indices
        self.current_position = 0
        self.current_batch_indices = []
        self.current_input_length = input_param['seq_length']

    def total(self):
        return len(self.indices)

    def begin(self, do_shuffle=True):
        if do_shuffle:
            random.shuffle(self.indices)
        self.current_position = 0
        self.current_batch_indices = self.indices[
            self.current_position:self.current_position + self.minibatch_size]

    def next(self):
        self.current_position += self.minibatch_size
        if self.no_batch_left():
            return None
        self.current_batch_indices = self.indices[
            self.current_position:self.current_position + self.minibatch_size]

    def no_batch_left(self):
        return self.current_position + self.minibatch_size >= self.total()

    def get_batch(self):
        if self.no_batch_left():
            return None
        input_batch = np.zeros(
            (self.minibatch_size, self.current_input_length, self.image_width,
            self.image_width, 1)).astype(self.input_data_type)
        for i in range(self.minibatch_size):
            batch_ind = self.current_batch_indices[i]
            begin = batch_ind
            end = begin + self.current_input_length
            data_slice = self.datas[begin:end]
            input_batch[i, :self.current_input_length, :, :, :] = data_slice
        return input_batch


class DataProcess(object):
    """This class's implementation remains unchanged from the previous lazy-loading version."""
    def __init__(self, input_param):
        self.paths = input_param['paths']
        self.category_1 = ['boxing', 'handclapping', 'handwaving', 'walking']
        self.category_2 = ['jogging', 'running']
        self.category = self.category_1 + self.category_2
        self.image_width = input_param['image_width']
        self.debug = input_param['debug']
        self.train_person = [f'{i:02d}' for i in range(1, 17)]
        self.test_person = [f'{i:02d}' for i in range(17, 26)]
        self.input_param = input_param
        self.seq_len = input_param['seq_length']

    def load_data(self, path, mode='train'):
        assert mode in ['train', 'test']
        person_id = self.train_person if mode == 'train' else self.test_person
        if self.debug:
            person_id = ['01'] if mode == 'train' else ['17']
        print('begin FAST scan of data folders: ' + str(path))

        all_frame_paths = []
        frames_person_mark = []
        frames_file_name = []
        frames_category_mark = []
        person_mark_counter = 0
        
        for c_dir in self.category:
            c_dir_path = os.path.join(path, c_dir)
            if not os.path.isdir(c_dir_path): continue
            for p_c_dir in sorted(os.listdir(c_dir_path)):
                if len(p_c_dir) < 8 or p_c_dir[6:8] not in person_id: continue
                
                dir_path = os.path.join(c_dir_path, p_c_dir)
                if not os.path.isdir(dir_path): continue
                
                person_mark_counter += 1
                frame_category_flag = 1 if c_dir in self.category_1 else 2
                
                current_video_frames = []
                for cur_file in sorted(os.listdir(dir_path)):
                    if cur_file.startswith('image') and cur_file.endswith(('.png', '.jpg')):
                        current_video_frames.append((cur_file, os.path.join(dir_path, cur_file)))
                
                for file_name, file_path in current_video_frames:
                    all_frame_paths.append(file_path)
                    frames_file_name.append(file_name)
                    frames_person_mark.append(person_mark_counter)
                    frames_category_mark.append(frame_category_flag)
        
        indices = []
        index = len(frames_person_mark) - 1
        while index >= self.seq_len - 1:
            if frames_person_mark[index] == frames_person_mark[index - self.seq_len + 1]:
                end = int(frames_file_name[index][6:10])
                start = int(frames_file_name[index - self.seq_len + 1][6:10])
                if end - start == self.seq_len - 1:
                    indices.append(index - self.seq_len + 1)
                    if frames_category_mark[index] == 1:
                        index -= self.seq_len -1
                    elif frames_category_mark[index] == 2:
                        index -= 2
            index -= 1
        
        print('Scanned ' + str(len(all_frame_paths)) + ' pictures')
        print('Found ' + str(len(indices)) + ' sequences')

        lazy_data_array = LazyMemmapArray(all_frame_paths, self.image_width, self.input_param.get('input_data_type', 'float32'))
        
        return lazy_data_array, indices

    def get_train_input_handle(self):
        train_data, train_indices = self.load_data(self.paths, mode='train')
        return InputHandle(train_data, train_indices, self.input_param)

    def get_test_input_handle(self):
        test_data, test_indices = self.load_data(self.paths, mode='test')
        return InputHandle(test_data, test_indices, self.input_param)


def load_data(batch_size, val_batch_size, data_root, num_workers=16,
              pre_seq_length=10, aft_seq_length=20, in_shape=[10, 1, 128, 128],
              distributed=False, use_augment=False, use_prefetcher=False, 
              drop_last=False, debug=False):
    """This function remains unchanged."""
    img_width = in_shape[-1] if in_shape is not None else 128
    input_param = {
        'paths': os.path.join(data_root, 'kth'),
        'image_width': img_width,
        'minibatch_size': batch_size,
        'seq_length': (pre_seq_length + aft_seq_length),
        'input_data_type': 'float32',
        'name': 'kth',
        'debug': debug
    }
    input_handle = DataProcess(input_param)
    
    train_input_handle = input_handle.get_train_input_handle()
    test_input_handle = input_handle.get_test_input_handle()

    train_set = KTHDataset(train_input_handle.datas,
                           train_input_handle.indices,
                           pre_seq_length,
                           aft_seq_length, use_augment=use_augment)
    test_set = KTHDataset(test_input_handle.datas,
                          test_input_handle.indices,
                          pre_seq_length,
                          aft_seq_length, use_augment=False)

    dataloader_train = create_loader(train_set,
                                     batch_size=batch_size,
                                     shuffle=True, is_training=True,
                                     pin_memory=True, drop_last=True,
                                     num_workers=num_workers,
                                     distributed=distributed, use_prefetcher=use_prefetcher)
    dataloader_vali = None
    dataloader_test = create_loader(test_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=drop_last,
                                    num_workers=num_workers,
                                    distributed=distributed, use_prefetcher=use_prefetcher)

    return dataloader_train, dataloader_vali, dataloader_test


if __name__ == '__main__':
    dataloader_train, _, dataloader_test = \
        load_data(batch_size=16,
                val_batch_size=4,
                data_root='/nas_data/LSH/data/',
                num_workers=4,
                pre_seq_length=10, aft_seq_length=20, debug=False)

    print(len(dataloader_train), len(dataloader_test))
    import time
    start = time.time()
    for i, item in enumerate(dataloader_train):
        if i == 0:
            print(f"Time to load first batch: {time.time() - start:.4f}s")
        print(f"Batch {i}: data shape {item[0].shape}, labels shape {item[1].shape}")
        if i > 2:
             break
    start = time.time()
    for item in dataloader_test:
        print(f"Time to load first test batch: {time.time() - start:.4f}s")
        print(f"Test batch: data shape {item[0].shape}, labels shape {item[1].shape}")
        break