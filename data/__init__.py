import math
import torch
import random
import importlib
import numpy as np
import torch.utils.data
from copy import deepcopy
from os import path as osp
from functools import partial
from .data_sampler import EnlargedSampler
from .prefetch_dataloader import PrefetchDataLoader
from utils import get_root_logger, scandir, get_dist_info, DATASET_REGISTRY

__all__ = ['build_dataset', 'build_dataloader']

# automatically scan and import dataset modules for registry
# scan all the files under the data folder with '_dataset' in file names
data_folder = osp.dirname(osp.abspath(__file__))
dataset_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(data_folder) if v.endswith('_dataset.py')]
_dataset_modules = [importlib.import_module(f'data.{file_name}') for file_name in dataset_filenames]

def worker_init_fn(worker_id, num_workers, rank, seed):
    # Set the worker seed to num_workers * rank + worker_id + seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def build_dataset(dataset_opt):
    """Build dataset from options.

    Args:
        dataset_opt (dict): Configuration for dataset. It must contain:
            name (str): Dataset name.
            type (str): Dataset type.
    """
    dataset_opt = deepcopy(dataset_opt)
    dataset = DATASET_REGISTRY.get(dataset_opt['type'])(dataset_opt)
    logger = get_root_logger()
    if dataset_opt['phase'] == 'val':
        if dataset_opt['dataset_type'] == 'Synthetic':
            logger.info(f'''Dataset [{dataset.__class__.__name__}] - {dataset_opt["dataset_type"]}{dataset_opt["name"]} 
                        with noise level: {dataset_opt["syn_noise"]["noise_level"]}, alpha: {dataset_opt["syn_isp"]["alpha"]} is built.''')
        else:
            logger.info(f'Dataset [{dataset.__class__.__name__}] - {dataset_opt["dataset_type"]} {dataset_opt["name"]} is built.')
    else:
        logger.info(f'Dataset [{dataset.__class__.__name__}] - {dataset_opt["name"]} is built.')
    return dataset

def build_dataloader(dataset, dataset_opt, num_gpu=1, dist=False, sampler=None, seed=None):
    """Build dataloader.

    Args:
        dataset (torch.utils.data.Dataset): Dataset.
        dataset_opt (dict): Dataset options. It contains the following keys:
            phase (str): 'train' or 'val'.
            num_worker_per_gpu (int): Number of workers for each GPU.
            batch_size_per_gpu (int): Training batch size for each GPU.
        num_gpu (int): Number of GPUs. Used only in the train phase.
            Default: 1.
        dist (bool): Whether in distributed training. Used only in the train
            phase. Default: False.
        sampler (torch.utils.data.sampler): Data sampler. Default: None.
        seed (int | None): Seed. Default: None
    """
    phase = dataset_opt['phase']
    rank, _ = get_dist_info()
    if phase == 'train':
        if dist:  # distributed training
            batch_size = dataset_opt['batch_size_per_gpu']
            num_workers = dataset_opt['num_worker_per_gpu']
        else:  # non-distributed training
            multiplier = 1 if num_gpu == 0 else num_gpu
            batch_size = dataset_opt['batch_size_per_gpu'] * multiplier
            num_workers = dataset_opt['num_worker_per_gpu'] * multiplier
        dataloader_args = dict(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            sampler=sampler,
            drop_last=True)
        if sampler is None:
            dataloader_args['shuffle'] = True
        dataloader_args['worker_init_fn'] = partial(
            worker_init_fn, num_workers=num_workers, rank=rank, seed=seed) if seed is not None else None
    elif phase in ['val', 'test']:  # validation
        dataloader_args = dict(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)
    else:
        raise ValueError(f"Wrong dataset phase: {phase}. Supported ones are 'train', 'val' and 'test'.")

    dataloader_args['pin_memory'] = dataset_opt.get('pin_memory', False)
    dataloader_args['persistent_workers'] = dataset_opt.get('persistent_workers', False)

    prefetch_mode = dataset_opt.get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':  # CPUPrefetcher
        num_prefetch_queue = dataset_opt.get('num_prefetch_queue', 1)
        logger = get_root_logger()
        logger.info(f'Use {prefetch_mode} prefetch dataloader: num_prefetch_queue = {num_prefetch_queue}')
        return PrefetchDataLoader(num_prefetch_queue=num_prefetch_queue, **dataloader_args)
    elif prefetch_mode == 'cuda': # CUDAPrefetcher
        return torch.utils.data.DataLoader(**dataloader_args)

def create_train_val_dataloader(opt, logger):
    train_loader, val_loaders = None, []
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = build_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)
            train_loader = build_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])

            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio /
                (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            logger.info(
                'Training statistics:'
                f'\n\tNumber of train images: {len(train_set)}'
                f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                f'\n\tWorld size (gpu number): {opt["world_size"]}'
                f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')

        elif phase == 'val':
            for dataset_type, dataset in dataset_opt['val_datasets'].items():
                if dataset['mode']:
                    dataset_opt['dataset_type'] = dataset_type
                    dataset_opt['data_path'] = dataset['data_path']

                    if dataset_type == 'Synthetic':
                        noise_levels, alphas = [], []
                        if type(dataset_opt['syn_noise']['noise_level']) == list:
                            noise_levels = deepcopy(dataset_opt['syn_noise']['noise_level'])
                        else:
                            noise_levels[0] = deepcopy(dataset_opt['syn_noise']['noise_level'])
                        if type(dataset_opt['syn_isp']['alpha']) == list:
                            alphas = deepcopy(dataset_opt['syn_isp']['alpha'])
                        else:
                            alphas[0] = deepcopy(dataset_opt['syn_isp']['alpha'])
                        for _, noise_level in enumerate(noise_levels):
                            for _, alpha in enumerate(alphas):
                                dataset_opt['syn_noise']['noise_level'] = deepcopy(noise_level)
                                dataset_opt['syn_isp']['alpha'] = deepcopy(alpha)
                                val_set = build_dataset(dataset_opt)
                                val_loader = build_dataloader(
                                    val_set,
                                    dataset_opt,
                                    num_gpu=opt['num_gpu'],
                                    dist=opt['dist'],
                                    sampler=None,
                                    seed=opt['manual_seed'])
                                logger.info(
                                    f'Number of val images in {dataset_type} {dataset_opt["name"]}: '
                                    f'{len(val_set)}')
                                val_loaders.append(val_loader)
                    else:
                        val_set = build_dataset(dataset_opt)
                        val_loader = build_dataloader(
                            val_set,
                            dataset_opt,
                            num_gpu=opt['num_gpu'],
                            dist=opt['dist'],
                            sampler=None,
                            seed=opt['manual_seed'])
                        logger.info(
                                    f'Number of val images in {dataset_type} {dataset_opt["name"]}: '
                                    f'{len(val_set)}')
                        val_loaders.append(val_loader)

        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, train_sampler, val_loaders, total_epochs, total_iters