import glob
import os
import random

import numpy as np
from skimage import io, transform
import torch
from torch.utils.data import Dataset

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample
        img = transform.resize(image, self.output_size)
        return img

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample
        image = image.transpose((2, 0, 1)).astype(np.float32)
        return torch.from_numpy(image)

def paste_slices(tup):
    pos, w, max_w = tup
    wall_min = max(pos, 0)
    wall_max = min(pos+w, max_w)
    block_min = -min(pos, 0)
    block_max = max_w-max(pos+w, max_w)
    block_max = block_max if block_max != 0 else None
    return slice(wall_min, wall_max), slice(block_min, block_max)

def paste(wall, block, loc):
    loc_zip = zip(loc, block.shape, wall.shape)
    wall_slices, block_slices = zip(*map(paste_slices, loc_zip))
    wall[wall_slices] = block[block_slices]

class RandomTransform(object):
    def __init__(self, 
        p_scale = 0.7, min_scale = 0.1, max_scale = 2.0,
        p_recolor = 0.7, color_min = 0.0, color_max = 1.0,
        p_rotate = 0.7, rot_min = -90.0, rot_max = 90.0,
        p_vflip = 0.7, p_hflip = 0.7,
        trans_xmin = 0, trans_xmax = 576,
        trans_ymin = 0, trans_ymax = 288
        ):
        self.p_scale = p_scale
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.p_recolor = p_recolor
        self.color_min = color_min
        self.color_max = color_max
        self.p_rotate = p_rotate
        self.rot_min = rot_min
        self.rot_max = rot_max
        self.p_vflip = p_vflip
        self.p_hflip = p_hflip
        self.trans_xmin = trans_xmin
        self.trans_xmax = trans_xmax
        self.trans_ymin = trans_ymin
        self.trans_ymax = trans_ymax 

    def __call__(self, sample):
        img = sample
        try:
            if self.p_scale > random.random():
                x_scale = random.uniform(self.min_scale, self.max_scale)
                y_scale = random.uniform(self.min_scale, self.max_scale)
                img = cv2.resize(img, (int(x_scale * img.shape[1]), int(y_scale * img.shape[0])))
            if self.p_recolor > random.random():
                new_color = [random.uniform(self.color_min, self.color_max) for _ in range(3)] + [1]
                img = new_color * img
            if self.p_rotate > random.random():
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) if 0.5 > random.random() else cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            if self.p_vflip > random.random():
                img = cv2.flip(img, 0)
            if self.p_hflip > random.random():
                img = cv2.flip(img, 1)
        except Exception as e:
            pass
        x_max = 576
        y_max = 288
        new_img = np.zeros((y_max,x_max,4), dtype=sample.dtype)
        new_x = random.randint(-1 * int(img.shape[1]/2), x_max + (-1 * int(img.shape[1]/2)))
        new_y = random.randint(-1 * int(img.shape[0]/2), y_max + (-1 * int(img.shape[0]/2)))
        paste(new_img, img, (new_y, new_x))
        return new_img[:,:,:3].astype(np.float32)

class LayersDataset(Dataset):
    def __init__(self, base_path, transform=None):
        self.im_paths = glob.glob(os.path.join(base_path, '[0-9]*.png'))
        self.transform = transform
    
    def __len__(self):
        return len(self.im_paths)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.im_paths[idx]
        sample = io.imread(img_name)
        if self.transform:
            sample = self.transform(sample)
        return {'idx': idx, 'path': img_name, 'image': sample}

class SADataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        cur_data = self.data[idx]
        feature, label, mask = io.imread(cur_data['feature'])[:,:,:3].astype(np.float32), cur_data['label'], cur_data['mask']
        feature, label, mask = torch.from_numpy(feature.transpose((2, 0, 1)).astype(np.float32)), torch.from_numpy(label.astype(np.float32)), torch.from_numpy(mask)
        return {'feature': feature, 'label': label, 'mask': mask}