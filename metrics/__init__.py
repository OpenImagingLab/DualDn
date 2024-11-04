import importlib
from copy import deepcopy
from os import path as osp

from utils import get_root_logger, scandir, METRIC_REGISTRY


__all__ = ['calculate_psnr', 'calculate_ssim', 'calculate_niqe']

# automatically scan and import loss modules for registry
# scan all the files under the 'losses' folder and collect files ending with '_loss.py'
metric_folder = osp.dirname(osp.abspath(__file__))
metric_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(metric_folder) if v.endswith('_metric.py')]
# import all the loss modules
_metric_modules = [importlib.import_module(f'metrics.{file_name}') for file_name in metric_filenames]

def calculate_metric(data, metric):
    """Calculate metric from data and options.

    Args:
        metric (dict): Configuration. It must contain:
            type (str): Model type.
    """
    metric = deepcopy(metric)
    metric_type = metric.pop('type')
    metric = METRIC_REGISTRY.get(metric_type)(**data, **metric)
    return metric
