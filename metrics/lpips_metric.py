import torch
import lpips
import numpy as np
from .metric_util import reorder_image, to_y_channel
from utils import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def calculate_lpips(img1, img2, crop_border, input_order='HWC', test_y_channel=False, spatial=False, **kwargs):
    assert img1.shape == img2.shape, (f'Image shapes are different: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    if type(img1) == torch.Tensor:
        if len(img1.shape) == 4:
            img1 = img1.squeeze(0)
        img1 = img1.detach().cpu().numpy().transpose(1,2,0)
    if type(img2) == torch.Tensor:
        if len(img2.shape) == 4:
            img2 = img2.squeeze(0)
        img2 = img2.detach().cpu().numpy().transpose(1,2,0)
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if img1.max() <= 1:
        img1 = img1.astype(np.float64) * 255
    else:
        img1 = img1.astype(np.float64)
    if img2.max() <= 1:
        img2 = img2.astype(np.float64) * 255
    else:
        img2 = img2.astype(np.float64)

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    loss_fn = lpips.LPIPS(net='alex', spatial=spatial, verbose=False)

    img1 = lpips.im2tensor(img1)
    img2 = lpips.im2tensor(img2)

    if spatial:
        return float(loss_fn.forward(img1, img2).mean())
    else:
        return float(loss_fn.forward(img1, img2))
