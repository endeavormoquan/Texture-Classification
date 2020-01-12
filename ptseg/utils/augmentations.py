import torch
import numbers
import random
import numpy as np
import cv2

"""
img: raw image
mask: groundtruth
"""


class Compose(object):
    """Set augmentation methods
    Example: data_aug = Compose([RandomSizedandCrop((550,280),(512,256)),
                                 ColorJitter(0.3)])
    """
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, img, mask):
        for a in self.augmentations:
            img, mask = a(img, mask)
        return img, mask


class RandomCrop(object):
    """RandomCrop
    Args:
         size: int or Tuple[int, int]
    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, mask):
        h, w, c = img.shape
        tw, th = self.size
        if w == tw and h == th:
            return img, mask
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        img = img[y1:y1+th, x1:x1+tw, :]
        for i in range(len(mask)):
            mask[i] = mask[i][y1:y1+th, x1:x1+tw]
        return img, mask


class RandomHorizontallyFlip(object):
    """Image Flip"""
    def __call__(self, img, mask):
        if random.random() < 0.5:
            img = cv2.flip(img, 0)
            for i in range(len(mask)):
                mask[i] = cv2.flip(mask[i], 0)
            return img, mask
        return img, mask


class RandomSizedandCrop(object):
    """Resize and RandomCrop
    Args:
         resize_size: Tuple[int, int]
         crop_size: int or Tuple[int, int]
    """
    def __init__(self, resize_size, crop_size):
        self.resize_size = resize_size
        self.crop_size = crop_size
        self.crop = RandomCrop(self.crop_size)

    def __call__(self, img, mask):
        w = self.resize_size[0]
        h = self.resize_size[1]
        resize_w = w - random.randint(0, w - self.crop_size[0] - 1)
        resize_h = h - random.randint(0, h - self.crop_size[1] - 1)
        img = cv2.resize(img, (resize_w, resize_h))
        for i in range(len(mask)):
            mask[i] = cv2.resize(mask[i], (resize_w, resize_h), interpolation=cv2.INTER_NEAREST)
        return self.crop(img, mask)


class Grayscale(object):
    def __call__(self, img):
        gs = np.copy(img)
        gs[0].__mul__(0.114).__add__(gs[1].__mul__(0.587)).__add__(gs[2].__mul__(0.299))
        gs[1] = gs[0].__copy__()
        gs[2] = gs[0].__copy__()
        return gs


class Saturation(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        alpha = np.random.uniform(-1*self.var, self.var, 1) + 1
        img_s = alpha * img + (1 - alpha) * gs
        return img_s


class Brightness(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        alpha = np.random.uniform(-1*self.var, self.var, 1) + 1
        img_b = alpha * img
        return img_b


class Contrast(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        gs.fill(np.mean(gs))
        alpha = np.random.uniform(-1*self.var, self.var, 1) + 1
        img_c = alpha * img + (1 - alpha) * gs
        return img_c


class ColorJitter(object):
    """ColorJitter consists of Brightness, Contrast, Saturation
    Args:
         jitter_param: 0.3 [default]
    """
    def __init__(self, jitter_param=0.3):
        self.jitter_param = jitter_param
        self.transform_func = [Brightness(jitter_param), Contrast(jitter_param), Saturation(jitter_param)]

    def __call__(self, img, mask):
        shuffle_operation = np.random.permutation(3)
        for i in range(3):
            img = self.transform_func[shuffle_operation[i]](img)
        return img, mask


class ColorAugmentation(object):
    def __init__(self, eig_vec=None, eig_val=None):
        if eig_vec is None:
            eig_vec = torch.Tensor([
                [ 0.4009,  0.7192, -0.5675],
                [-0.8140, -0.0045, -0.5808],
                [ 0.4203, -0.6948, -0.5836],
            ])
        if eig_val is None:
            eig_val = torch.Tensor([[0.2175, 0.0188, 0.0045]])
        self.eig_val = eig_val  # 1*3
        self.eig_vec = eig_vec  # 3*3

    def __call__(self, tensor):
        assert tensor.size(0) == 3
        alpha = torch.normal(means=torch.zeros_like(self.eig_val))*0.1
        quatity = torch.mm(self.eig_val*alpha, self.eig_vec)
        tensor = tensor + quatity.view(3, 1, 1)
        return tensor
