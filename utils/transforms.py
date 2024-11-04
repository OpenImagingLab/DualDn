import cv2
import random
import torch
import numpy as np


def un_binning(sRGB, raw):
    h_RGB, w_RGB = sRGB.shape[:2]
    h_raw, w_raw = raw.shape[:2]
    if h_raw == 2*h_RGB and w_raw == 2*w_RGB: # if binning, interpolating
        sRGB = cv2.resize(sRGB, (w_raw, h_raw), interpolation=cv2.INTER_CUBIC)
    
    return sRGB


def mod_crop(img, scale):
    """Mod crop images, used during testing.

    Args:
        img (ndarray): Input image.
        scale (int): Scale factor.

    Returns:
        ndarray: Result image.
    """
    img = img.copy()
    if img.ndim in (2, 3):
        h, w = img.shape[0], img.shape[1]
        h_remainder, w_remainder = h % scale, w % scale
        img = img[:h - h_remainder, :w - w_remainder, ...]
    else:
        raise ValueError(f'Wrong img ndim: {img.ndim}.')
    return img


def same_crop(*imgs):

    out = []
    if not isinstance(imgs, tuple):
        imgs = [imgs]

    # determine input type: Numpy array or Tensor
    input_type = 'Tensor' if torch.is_tensor(imgs[0]) else 'Numpy'

    if input_type == 'Tensor':
        h, w = imgs[0].size()[-2:]
    else:
        h, w = imgs[0].shape[0:2]

    for img in imgs:
        h1, w1 = img.shape[0:2]
        if h1 != h or w1 != w:
            top = (h1-h)//2
            left = (w1-w)//2

            # crop lq patch
            if input_type == 'Tensor':
                out.append(img[:, :, top:h1-top, left:w1-left])
            else:
                out.append(img[top:h1-top, left:w1-left, ...])
        else:
            out.append(img)

    if len(imgs) == 1:
        out = imgs[0]

    return out


def random_crop(*imgs, patch_size, color_mask=None):
    """Random crop. Support Numpy array and Tensor inputs.

    It crops tuple of images with corresponding locations.

    Args:
        imgs (list[ndarray] | ndarray | tuple[Tensor] | Tensor): images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        patch_size (int): GT patch size.

    Returns:
        tuple[ndarray] | ndarray: cropped images. If returned results
            only have one element, just return ndarray.
    """

    out = []
    if not isinstance(imgs, tuple):
        imgs = [imgs]

    # determine input type: Numpy array or Tensor
    input_type = 'Tensor' if torch.is_tensor(imgs[0]) else 'Numpy'

    if input_type == 'Tensor':
        h, w = imgs[0].size()[-2:]
    else:
        h, w = imgs[0].shape[0:2]

    if color_mask.any() == None:
        top = random.randint(0, h - patch_size)
        left = random.randint(0, w - patch_size)
    else:
        while True:
            top = random.randint(0, h - patch_size)
            left = random.randint(0, w - patch_size)
            if color_mask[top,left]==0: break # make sure it's red channel


    # crop lq patch
    if input_type == 'Tensor':
        out = [img[:, :, top:top + patch_size, left:left + patch_size] for img in imgs]
    else:
        out = [img[top:top + patch_size, left:left + patch_size, ...] for img in imgs]

    if len(imgs) == 1:
        out = imgs[0]

    return out


def fixed_crop(*imgs, patch_size, crop_border, central_crop, color_mask):
    """Fixed crop. Support Numpy array and Tensor inputs.

    It crops tuple of images with corresponding locations.

    Args:
        imgs (list[ndarray] | ndarray | tuple[Tensor] | Tensor): images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        patch_size (int): GT patch size.

    Returns:
        tuple[ndarray] | ndarray: cropped images. If returned results
            only have one element, just return ndarray.
    """

    out = []
    if not isinstance(imgs, tuple):
        imgs = [imgs]

    # determine input type: Numpy array or Tensor
    input_type = 'Tensor' if torch.is_tensor(imgs[0]) else 'Numpy'

    if input_type == 'Tensor':
        h, w = imgs[0].size()[-2:]
    else:
        h, w = imgs[0].shape[0:2]

    # crop lq patch
    top, left = 0,0
    if central_crop:
        h = h - crop_border
        w = w - crop_border
        exit_flag = False
        for top in range((h - patch_size)//2, h - patch_size):
            for left in range((w - patch_size)//2, w - patch_size):
                if color_mask[top,left] == 0: 
                    exit_flag = True
                    break # make sure it's red channel
            if exit_flag:
                break
        if input_type == 'Tensor':
            out = [img[:, :, top:top + patch_size + crop_border, left:left + patch_size + crop_border] for img in imgs]
        else:
            out = [img[top:top + patch_size + crop_border, left:left + patch_size + crop_border, ...] for img in imgs]
    else:
        h = h - crop_border
        w = w - crop_border
        exit_flag = False
        for top in range(h-patch_size):
            for left in range(w-patch_size):
                if color_mask[top,left] == 0: 
                    exit_flag = True
                    break # make sure it's red channel
            if exit_flag:
                break
        aa = (h-top)%patch_size
        if input_type == 'Tensor':
            out = [img[:, :, top:(h - (h-top)%patch_size + crop_border), left:(w - (w-left)%patch_size + crop_border)] for img in imgs]
        else:
            out = [img[top:(h - (h-top)%patch_size + crop_border), left:(w - (w-left)%patch_size + crop_border), ...] for img in imgs]

    if len(imgs) == 1:
        out = imgs[0]

    return out


def window_crop(*imgs, window_size):
    """window_crop. Support Numpy array and Tensor inputs.

    It crops tuple of images with corresponding locations.

    Args:
        imgs (list[ndarray] | ndarray | tuple[Tensor] | Tensor): images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        window_size (int): window size.

    Returns:
        tuple[ndarray] | ndarray: cropped images. If returned results
            only have one element, just return ndarray.
    """

    out = []
    if not isinstance(imgs, tuple):
        imgs = [imgs]

    # determine input type: Numpy array or Tensor
    input_type = 'Tensor' if torch.is_tensor(imgs[0]) else 'Numpy'

    if input_type == 'Tensor':
        h, w = imgs[0].size()[-2:]
    else:
        h, w = imgs[0].shape[0:2]

    top = h - h % window_size
    left = w - w % window_size

    # crop lq patch
    if input_type == 'Tensor':
        out = [img[:, :, 0:top, 0:left] for img in imgs]
    else:
        out = [img[0:top, 0:left, ...] for img in imgs]

    if len(imgs) == 1:
        out = imgs[0]

    return out


def data_augmentation(image, mode):
    """
    Performs data augmentation of the input image
    Input:
        image: a cv2 (OpenCV) image
        mode: int. Choice of transformation to apply to the image
                0 - no transformation
                1 - flip up and down
                2 - rotate counterwise 90 degree
                3 - rotate 90 degree and flip up and down
                4 - rotate 180 degree
                5 - rotate 180 degree and flip
                6 - rotate 270 degree
                7 - rotate 270 degree and flip
    """
    if mode == 0:
        # original
        out = image
    elif mode == 1:
        # flip up and down
        out = np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(image)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(image, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(image, k=3)
        out = np.flipud(out)
    else:
        raise Exception('Invalid choice of image transformation')

    return out


def random_augmentation(*args):
    out = []
    flag_aug = random.randint(0,7)
    for data in args:
        out.append(data_augmentation(data, flag_aug).copy())
    return out