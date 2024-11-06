# -----------------------------------------------------------------------------------
# [ECCV2024] DualDn: Dual-domain Denoising via Differentiable ISP 
# [Homepage] https://openimaginglab.github.io/DualDn/
# [Author] Originally Written by Ruikang Li, from MMLab, CUHK.
# [License] Absolutely open-source and free to use, please cite our paper if possible. :)
# -----------------------------------------------------------------------------------

import os
import time
import torch
import datetime
from os import path as osp

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from utils import (parse_options, load_resume_state, mkdir_and_rename, make_exp_dirs, \
                    copy_opt_file, init_loggers, check_resume, MessageLogger, AvgTimer)
from models import build_model
from data import create_train_val_dataloader
from data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher


def main():
    opt = parse_options()

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    ##* load resume states if necessary
    resume_state = load_resume_state(opt)

    ##* mkdir for experiments and logger
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name'] and opt['rank'] == 0:
            mkdir_and_rename(osp.join(opt['root_path'], 'tb_logger', opt['name']))

    ##* copy the yml file to the experiment root
    copy_opt_file(opt['opt_path'], opt['path']['experiments_root'])

    ##* initialize loggers
    logger, tb_logger = init_loggers(opt)

    ##* create train and validation dataloaders
    train_loader, train_sampler, val_loaders, total_epochs, total_iters = create_train_val_dataloader(opt, logger)

    ##* create model
    model = build_model(opt)

    ##* resume training
    if resume_state:  
        check_resume(opt, resume_state['iter'])
        model.resume_training(resume_state)
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        start_epoch = 0
        current_iter = 0

    ##* create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    ##* dataloader prefetcher
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f"Wrong prefetch_mode {prefetch_mode}. Supported ones are: None, 'cuda', 'cpu'.")

    ##* start training
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_timer, iter_timer = AvgTimer(), AvgTimer()
    start_time = time.time()

    for epoch in range(start_epoch, total_epochs + 1):
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None:
            data_timer.record()
            current_iter += 1
            if current_iter > total_iters:
                break

            model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            model.feed_train_data(train_data)
            model.optimize_parameters()
            iter_timer.record()
            if current_iter == 1:
                msg_logger.reset_start_time()
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_timer.get_current_time(), 'data_time': data_timer.get_current_time()})
                log_vars.update({'avg_time': iter_timer.get_avg_time(), 'avg_data_time': data_timer.get_avg_time()})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter)

            if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0):
                for val_loader in val_loaders:
                    model.validation(val_loader, current_iter, tb_logger)

            data_timer.start()
            iter_timer.start()
            train_data = prefetcher.next()
        # end of iter
    # end of epoch

    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
    if opt.get('val') is not None:
        for val_loader in val_loaders:
            model.validation(val_loader, current_iter, tb_logger)
    if tb_logger:
        tb_logger.close()

if __name__ == '__main__':
    main()
