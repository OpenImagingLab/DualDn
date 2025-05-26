# -----------------------------------------------------------------------------------
# [ECCV2024] DualDn: Dual-domain Denoising via Differentiable ISP 
# [Homepage] https://openimaginglab.github.io/DualDn/
# [Author] Originally Written by Ruikang Li, from MMLab, CUHK.
# [License] Absolutely open-source and free to use, please cite our paper if possible. :)
# -----------------------------------------------------------------------------------

import os
import h5py
import torch
import numpy as np
from scipy import io
from tqdm import tqdm
from os import path as osp
from collections import OrderedDict

from losses import build_loss
from archs import build_network
from metrics import calculate_metric
from models.base_model import BaseModel
from utils import MODEL_REGISTRY, get_root_logger, imwrite, tensor2img, run_pipeline, DND_bundle_submissions_raw, DND_bundle_submissions_srgb


@MODEL_REGISTRY.register()
class DualDn_Model(BaseModel):
    def __init__(self, opt):
        super(DualDn_Model, self).__init__(opt)

        self.net_g = build_network(self.opt['network'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        ##* load pretrained models
        load_path = self.opt['path'].get('pretrain_network', None)
        if load_path is not None:
            logger = get_root_logger()
            logger.info('Loading model for G [{:s}] ...'.format(load_path))
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load', True), param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        ##* define losses
        self.loss_type = train_opt['loss_type']
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_train_data(self, data):
        self.ref = data['ref'][0]
        self.gamma_type = data['gamma_type'][0]
        self.demosaic_type = data['demosaic_type'][0]
        self.index = data['img_ind']
        self.final_stage = data['final_stage'][0]
        self.gt_Raw = data['gt_Raw'].to(self.device)
        self.lq_Raw = data['lq_Raw'].to(self.device)

        self.alpha = data['alpha'].to(self.device)
        self.color_mask = data['color_mask'].to(self.device)
        self.wb_matrix = data['wb_matrix'].to(self.device)
        self.rgb_xyz_matrix = data['rgb_xyz_matrix'].to(self.device)
        
        self.gt_sRGB = run_pipeline(self.gt_Raw, {'color_mask':self.color_mask, 'wb_matrix':self.wb_matrix, 'color_desc':'RGBG', 'rgb_xyz_matrix':self.rgb_xyz_matrix, 'ref':self.ref, 'gamma_type':self.gamma_type, 'demosaic_type':self.demosaic_type, 'alpha':self.alpha}, 'normal', self.final_stage)
        self.lq_sRGB = run_pipeline(self.lq_Raw, {'color_mask':self.color_mask, 'wb_matrix':self.wb_matrix, 'color_desc':'RGBG', 'rgb_xyz_matrix':self.rgb_xyz_matrix, 'ref':self.ref, 'gamma_type':self.gamma_type, 'demosaic_type':self.demosaic_type, 'alpha':self.alpha}, 'normal', self.final_stage)
        
        if self.opt['network']['with_noise_map']:
            self.k = data['k'].to(self.device)
            self.sigma = data['sigma'].to(self.device)

    def feed_test_data(self, data):
        self.ref = data['ref'][0]
        self.gamma_type = data['gamma_type'][0]
        self.demosaic_type = data['demosaic_type'][0]
        self.index = data['img_ind'][0]
        self.final_stage = data['final_stage'][0]
        self.gt_Raw = data['gt_Raw'].to(self.device)
        self.lq_Raw = data['lq_Raw'].to(self.device)
        
        self.alpha = data['alpha'].to(self.device)
        self.color_mask = data['color_mask'].to(self.device)
        self.wb_matrix = data['wb_matrix'].to(self.device)
        self.rgb_xyz_matrix = data['rgb_xyz_matrix'].to(self.device)
        
        self.ref_sRGB = data['ref_sRGB'].to(self.device)
        if (self.ref_sRGB == 0).all():
            self.gt_sRGB = run_pipeline(self.gt_Raw, {'color_mask':self.color_mask, 'wb_matrix':self.wb_matrix, 'color_desc':'RGBG', 'rgb_xyz_matrix':self.rgb_xyz_matrix, 'ref':self.ref, 'gamma_type':self.gamma_type, 'demosaic_type':self.demosaic_type, 'alpha':self.alpha}, 'normal', self.final_stage)
        self.lq_sRGB = run_pipeline(self.lq_Raw, {'color_mask':self.color_mask, 'wb_matrix':self.wb_matrix, 'color_desc':'RGBG', 'rgb_xyz_matrix':self.rgb_xyz_matrix, 'ref':self.ref, 'gamma_type':self.gamma_type, 'demosaic_type':self.demosaic_type, 'alpha':self.alpha}, 'normal', self.final_stage)
        
        if self.opt['network']['with_noise_map']:
            self.k = data['k'].to(self.device)
            self.sigma = data['sigma'].to(self.device).flatten()[0]

    def optimize_parameters(self):
        self.optimizer_g.zero_grad()
        if self.opt['network']['with_noise_map']:
            self.out_Raw, self.out_sRGB = self.net_g(self.lq_Raw, self.color_mask, self.wb_matrix, self.rgb_xyz_matrix, self.ref, self.gamma_type, self.demosaic_type, self.k, self.sigma, self.alpha, self.final_stage)
        else:
            self.out_Raw, self.out_sRGB = self.net_g(self.lq_Raw, self.color_mask, self.wb_matrix, self.rgb_xyz_matrix, self.ref, self.gamma_type, self.demosaic_type, alpha = self.alpha, final_stage = self.final_stage)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            if self.loss_type == 'Raw':
                l_total += self.cri_pix(self.out_Raw, self.gt_Raw)
                loss_dict['l_pix_raw'] = l_total
            elif self.loss_type == 'sRGB':
                l_total += self.cri_pix(self.out_sRGB, self.gt_sRGB)
                loss_dict['l_pix_srgb'] = l_total
            elif self.loss_type == 'Dual':
                l_pix_raw = self.cri_pix(self.out_Raw, self.gt_Raw)
                l_pix_srgb = self.cri_pix(self.out_sRGB, self.gt_sRGB)
                l_total += l_pix_raw + l_pix_srgb
                loss_dict['l_pix_raw'] = l_pix_raw
                loss_dict['l_pix_srgb'] = l_pix_srgb
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.out_sRGB, self.gt_sRGB)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        
        # use grad_clip
        if self.opt['train']['use_grad_clip']:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)

    def syn_test(self, current_iter, val_opt):
        self.net_g.eval()
        with torch.no_grad():
            patch_size = self.opt['datasets']['val']['patch_size']
            crop_border = self.opt['datasets']['val']['crop_border']
            if self.opt['datasets']['val']['central_crop']:
                self.out_Raw = torch.zeros(1,1,patch_size,patch_size).to(self.lq_Raw.device)
                self.out_sRGB = torch.zeros(1,3,patch_size,patch_size).to(self.lq_sRGB.device)
                
                if self.opt['network']['with_noise_map']:
                    out_Rawpatch, out_sRGBpatch = self.net_g(self.lq_Raw, self.color_mask, self.wb_matrix, self.rgb_xyz_matrix, self.ref, self.gamma_type, self.demosaic_type, self.k, self.sigma, self.alpha, self.final_stage)
                else:
                    out_Rawpatch, out_sRGBpatch = self.net_g(self.lq_Raw, self.color_mask, self.wb_matrix, self.rgb_xyz_matrix, self.ref, self.gamma_type, self.demosaic_type, alpha = self.alpha, final_stage = self.final_stage)
                
                self.out_Raw, self.out_sRGB = out_Rawpatch[:,:,crop_border//2:crop_border//2+patch_size, crop_border//2:crop_border//2+patch_size], out_sRGBpatch[:,:,crop_border//2:crop_border//2+patch_size, crop_border//2:crop_border//2+patch_size]
            else:
                _,_,h,w = self.lq_Raw.shape
                self.out_Raw = torch.zeros(1,1,h-crop_border,w-crop_border).to(self.lq_Raw.device)
                self.out_sRGB = torch.zeros(1,3,h-crop_border,w-crop_border).to(self.lq_sRGB.device)
                patch_num_h = (h-crop_border)//patch_size
                patch_num_w = (w-crop_border)//patch_size
                for i in range(patch_num_h):
                    for j in range(patch_num_w):
                        if self.opt['network']['with_noise_map']:
                            out_Rawpatch, out_sRGBpatch = self.net_g(self.lq_Raw[:,:,i*patch_size:(i+1)*patch_size+crop_border,j*patch_size:(j+1)*patch_size+crop_border], \
                            self.color_mask[:,:,i*patch_size:(i+1)*patch_size+crop_border,j*patch_size:(j+1)*patch_size+crop_border], self.wb_matrix, self.rgb_xyz_matrix, self.ref, self.gamma_type, self.demosaic_type, self.k, self.sigma, self.alpha, self.final_stage)
                        else:
                            out_Rawpatch, out_sRGBpatch = self.net_g(self.lq_Raw[:,:,i*patch_size:(i+1)*patch_size+crop_border,j*patch_size:(j+1)*patch_size+crop_border], \
                            self.color_mask[:,:,i*patch_size:(i+1)*patch_size+crop_border,j*patch_size:(j+1)*patch_size+crop_border], self.wb_matrix, self.rgb_xyz_matrix, self.ref, self.gamma_type, self.demosaic_type, alpha = self.alpha, final_stage = self.final_stage)
                        
                        self.out_Raw[:,:,i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size], self.out_sRGB[:,:,i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size] = \
                            out_Rawpatch[:,:,crop_border//2:patch_size+crop_border//2,crop_border//2:patch_size+crop_border//2], out_sRGBpatch[:,:,crop_border//2:patch_size+crop_border//2,crop_border//2:patch_size+crop_border//2]
        self.net_g.train()
        
        save_img = val_opt.get('save_img', True)
        rgb2bgr = val_opt.get('rgb2bgr', True)
        use_img = True # whether to use reference images to compute metrics
        
        visuals = self.get_current_visuals()
        lq_sRGB = tensor2img([visuals['lq_sRGB']], rgb2bgr=rgb2bgr)
        gt_sRGB = tensor2img([visuals['gt_sRGB']], rgb2bgr=rgb2bgr)
        out_sRGB = tensor2img([visuals['out_sRGB']], rgb2bgr=rgb2bgr)

        # tentative for out of GPU memory
        del self.lq_sRGB
        del self.gt_sRGB
        del self.out_sRGB
        torch.cuda.empty_cache()

        metric_data = dict()
        k = self.k.item()
        alpha = self.alpha.item()
        if save_img:
            if self.is_train:
                save_path = osp.join(self.opt['path']['visualization'], str(current_iter))
                save_lq_sRGB_path = osp.join(save_path, 'noise_level_{:.4f}'.format(k), 'alpha_{:.2f}'.format(alpha), f'{self.index}_{current_iter}_lq.png')
                save_gt_sRGB_path = osp.join(save_path, 'noise_level_{:.4f}'.format(k), 'alpha_{:.2f}'.format(alpha), f'{self.index}_{current_iter}_gt.png')
                save_out_sRGB_path = osp.join(save_path, 'noise_level_{:.4f}'.format(k), 'alpha_{:.2f}'.format(alpha), f'{self.index}_{current_iter}_ours.png')
            else:
                save_path = self.opt['path']['visualization']
                save_lq_sRGB_path = osp.join(save_path,'noise_level_{:.4f}'.format(k), 'alpha_{:.2f}'.format(alpha), f'{self.index}_lq.png')
                save_gt_sRGB_path = osp.join(save_path,'noise_level_{:.4f}'.format(k), 'alpha_{:.2f}'.format(alpha), f'{self.index}_gt.png')
                save_out_sRGB_path = osp.join(save_path,'noise_level_{:.4f}'.format(k), 'alpha_{:.2f}'.format(alpha), f'{self.index}_ours.png')

            imwrite(lq_sRGB, save_lq_sRGB_path)
            imwrite(gt_sRGB, save_gt_sRGB_path)
            imwrite(out_sRGB, save_out_sRGB_path)

        out_data = open(save_path + '/noise_level_{:.4f}'.format(k) + '/alpha_{:.2f}'.format(alpha) + '/0_validation.txt', 'a')
        out_data.write(f'{self.index}\t')

        ##* calculate metrics
        if use_img:
            metric_data['img'] = out_sRGB
            metric_data['img1'] = out_sRGB
            metric_data['img2'] = gt_sRGB

            for name, metric in val_opt['metrics'].items():
                current_result = calculate_metric(metric_data, metric)
                out_data.write(f'{name}:{current_result}\t')
                self.metric_results[name] += current_result

        out_data.write(f'\r')
        out_data.close()

    def real_test(self, current_iter, val_opt):
        if self.alpha != 0:
            print(f"Warning: In real_captured testing, the predefined alpha should be 0. Because we don't want to modify the camera's original tone-mapping gain. Setting alpha to default 0.")
            self.alpha = 0
        
        self.net_g.eval()
        with torch.no_grad():
            patch_size = self.opt['datasets']['val']['patch_size']
            crop_border = self.opt['datasets']['val']['crop_border']
            if self.opt['datasets']['val']['central_crop']:
                self.out_Raw = torch.zeros(1,1,patch_size,patch_size).to(self.lq_Raw.device)
                self.out_sRGB = torch.zeros(1,3,patch_size,patch_size).to(self.lq_sRGB.device)
                
                if self.opt['network']['with_noise_map']:
                    out_Rawpatch, out_sRGBpatch = self.net_g(self.lq_Raw, self.color_mask, self.wb_matrix, self.rgb_xyz_matrix, self.ref, self.gamma_type, self.demosaic_type, self.k, self.sigma, self.alpha, self.final_stage, self.ref_sRGB)
                else:
                    out_Rawpatch, out_sRGBpatch = self.net_g(self.lq_Raw, self.color_mask, self.wb_matrix, self.rgb_xyz_matrix, self.ref, self.gamma_type, self.demosaic_type, alpha = self.alpha, final_stage = self.final_stage, ref_sRGB = self.ref_sRGB)
                
                self.out_Raw, self.out_sRGB = out_Rawpatch[:,:,crop_border//2:crop_border//2+patch_size, crop_border//2:crop_border//2+patch_size], out_sRGBpatch[:,:,crop_border//2:crop_border//2+patch_size, crop_border//2:crop_border//2+patch_size]
            else:
                _,_,h,w = self.lq_Raw.shape
                self.out_Raw = torch.zeros(1,1,h-crop_border,w-crop_border).to(self.lq_Raw.device)
                self.out_sRGB = torch.zeros(1,3,h-crop_border,w-crop_border).to(self.lq_sRGB.device)
                patch_num_h = (h-crop_border)//patch_size
                patch_num_w = (w-crop_border)//patch_size
                for i in range(patch_num_h):
                    for j in range(patch_num_w):
                        if self.opt['network']['with_noise_map']:
                            out_Rawpatch, out_sRGBpatch = self.net_g(self.lq_Raw[:,:,i*patch_size:(i+1)*patch_size+crop_border,j*patch_size:(j+1)*patch_size+crop_border], \
                                self.color_mask[:,:,i*patch_size:(i+1)*patch_size+crop_border,j*patch_size:(j+1)*patch_size+crop_border], self.wb_matrix, self.rgb_xyz_matrix, self.ref, self.gamma_type, self.demosaic_type, self.k, self.sigma, self.alpha, self.final_stage, \
                                self.ref_sRGB[:,:,i*patch_size:(i+1)*patch_size+crop_border,j*patch_size:(j+1)*patch_size+crop_border])
                        else:
                            out_Rawpatch, out_sRGBpatch = self.net_g(self.lq_Raw[:,:,i*patch_size:(i+1)*patch_size+crop_border,j*patch_size:(j+1)*patch_size+crop_border], \
                                self.color_mask[:,:,i*patch_size:(i+1)*patch_size+crop_border,j*patch_size:(j+1)*patch_size+crop_border], self.wb_matrix, self.rgb_xyz_matrix, self.ref, self.gamma_type, self.demosaic_type, alpha = self.alpha, final_stage = self.final_stage, \
                                ref_sRGB = self.ref_sRGB[:,:,i*patch_size:(i+1)*patch_size+crop_border,j*patch_size:(j+1)*patch_size+crop_border])
                        
                        self.out_Raw[:,:,i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size], self.out_sRGB[:,:,i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size] = \
                            out_Rawpatch[:,:,crop_border//2:patch_size+crop_border//2,crop_border//2:patch_size+crop_border//2], out_sRGBpatch[:,:,crop_border//2:patch_size+crop_border//2,crop_border//2:patch_size+crop_border//2]
        self.net_g.train()
        
        save_img = val_opt.get('save_img', True)
        rgb2bgr = val_opt.get('rgb2bgr', True)
        use_img = val_opt.get('use_img', False) # whether to use reference images to compute metrics
        
        visuals = self.get_current_visuals()
        lq_sRGB = tensor2img([visuals['lq_sRGB']], rgb2bgr=rgb2bgr)
        ref_sRGB = tensor2img([visuals['ref_sRGB']], rgb2bgr=rgb2bgr)
        out_sRGB = tensor2img([visuals['out_sRGB']], rgb2bgr=rgb2bgr)

        # tentative for out of GPU memory
        del self.lq_sRGB
        del self.ref_sRGB
        del self.out_sRGB
        torch.cuda.empty_cache()

        metric_data = dict()
        if save_img:
            if self.is_train:
                save_path = osp.join(self.opt['path']['visualization'], str(current_iter))
                save_lq_sRGB_path = osp.join(save_path, f'{self.index}_{current_iter}_lq.png')
                save_ref_sRGB_path = osp.join(save_path, f'{self.index}_{current_iter}_ref.png')
                save_out_sRGB_path = osp.join(save_path, f'{self.index}_{current_iter}_ours.png')
            else:
                save_path = self.opt['path']['visualization']
                save_lq_sRGB_path = osp.join(save_path, f'{self.index}_lq.png')
                save_ref_sRGB_path = osp.join(save_path, f'{self.index}_ref.png')
                save_out_sRGB_path = osp.join(save_path, f'{self.index}_ours.png')

            imwrite(lq_sRGB, save_lq_sRGB_path)
            imwrite(ref_sRGB, save_ref_sRGB_path)
            imwrite(out_sRGB, save_out_sRGB_path)

        ##* calculate metrics
        if use_img:
            out_data = open(save_path + '/0_validation.txt', 'a')
            out_data.write(f'{self.index}\t')
            metric_data['img'] = out_sRGB
            metric_data['img1'] = out_sRGB
            metric_data['img2'] = ref_sRGB
            
            for name, metric in val_opt['metrics'].items():
                current_result = calculate_metric(metric_data, metric)
                out_data.write(f'{name}:{current_result}\t')
                self.metric_results[name] += current_result

            out_data.write(f'\r')
            out_data.close()


    def dnd_test(self, current_iter, data_path):
        img_ind = "{:04}".format(self.index+1)
        Info_path = osp.join(data_path,'info.mat')
        Info = h5py.File(Info_path,'r')['info']
        Patch = np.array(Info[Info['boundingboxes'][0,self.index]]).T
        
        self.net_g.eval()
        out_dict = OrderedDict()
        if self.is_train:
            save_path = osp.join(self.opt['path']['visualization'], str(current_iter))
        else:
            save_path = self.opt['path']['visualization']
        
        with torch.no_grad():
            for i in range(20):
                loc_h_l, loc_w_l = Patch[i,0:2]
                loc_h_l = int(loc_h_l) - 1
                loc_w_l = int(loc_w_l) - 1
                top, left = 0, 0
                self.out_Raw = torch.zeros(1,1,512,512).to(self.lq_Raw.device)
                self.out_sRGB = torch.zeros(1,3,512,512).to(self.lq_Raw.device)
                exit_flag = False
                for top in range(loc_h_l-8, loc_h_l):
                    for left in range(loc_w_l-8, loc_w_l):
                        if self.color_mask[:,:,top,left] == 0: 
                            exit_flag = True
                            break # make sure it's red channel
                    if exit_flag:
                        break
                if self.opt['network']['with_noise_map']:
                    out_Rawpatch, out_sRGBpatch = self.net_g(self.lq_Raw[:,:,top:top+528,left:left+528], self.color_mask[:,:,top:top+528,left:left+528], self.wb_matrix, self.rgb_xyz_matrix, self.ref, self.gamma_type, self.demosaic_type, self.k, self.sigma, self.alpha, self.final_stage)
                else:
                    out_Rawpatch, out_sRGBpatch = self.net_g(self.lq_Raw[:,:,top:top+528,left:left+528], self.color_mask[:,:,top:top+528,left:left+528], self.wb_matrix, self.rgb_xyz_matrix, self.ref, self.gamma_type, self.demosaic_type, alpha = self.alpha, final_stage = self.final_stage)
                self.out_Raw, self.out_sRGB = out_Rawpatch[:,:,loc_h_l-top:loc_h_l-top+512,loc_w_l-left:loc_w_l-left+512], out_sRGBpatch[:,:,loc_h_l-top:loc_h_l-top+512,loc_w_l-left:loc_w_l-left+512]
                
                ##* intermediate sRGB output
                # self.out_sRGB = run_pipeline(self.out_Raw, {'color_mask':color_mask, 'wb_matrix':self.wb_matrix, 'color_desc':'RGBG', 'rgb_xyz_matrix':self.rgb_xyz_matrix, 'ref': self.ref, 'alpha':self.alpha}, False, 'normal', self.final_stage)
                
                out_dict['out_Raw'] = self.out_Raw.detach().cpu()
                out_dict['out_sRGB'] = self.out_sRGB.detach().cpu()
                
                save_out_Raw_path = osp.join(save_path,'Raw', '{}_{:0=2}.mat'.format(img_ind, i+1))
                save_out_sRGB_path = osp.join(save_path,'sRGB', '{}_{:0=2}.mat'.format(img_ind, i+1))
                os.makedirs(os.path.abspath(os.path.dirname(save_out_Raw_path)), exist_ok=True)
                os.makedirs(os.path.abspath(os.path.dirname(save_out_sRGB_path)), exist_ok=True)
                save_out_Raw = np.squeeze(np.squeeze(np.array(out_dict['out_Raw']).astype(np.float32), axis=0), axis=0)
                save_out_sRGB = np.squeeze(np.array(out_dict['out_sRGB']).astype(np.float32), axis=0).transpose(1,2,0)
                io.savemat(save_out_Raw_path, {'Idenoised_crop':save_out_Raw})
                io.savemat(save_out_sRGB_path, {'Idenoised_crop':save_out_sRGB})
                
                out_sRGB = tensor2img([out_dict['out_sRGB']], rgb2bgr=True)
                save_out_sRGB_path = osp.join(save_path,'visuals', '{}_{:0=2}_ours.png'.format(img_ind, i+1))
                imwrite(out_sRGB, save_out_sRGB_path)

        del self.lq_Raw
        del self.out_Raw
        del self.out_sRGB
        torch.cuda.empty_cache()

    def dist_validation(self, dataloader, current_iter, tb_logger):
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger)
        else:
            return 0.

    def nondist_validation(self, dataloader, current_iter, tb_logger):
        val_opt = self.opt['val']
        data_path = dataloader.dataset.data_path
        dataset_name = dataloader.dataset.name
        dataset_type = dataloader.dataset.dataset_type

        if not hasattr(self, 'metric_results'):  # only execute in the first run
            self.metric_results = {metric: 0 for metric in val_opt['metrics'].keys()}
        # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
        self._initialize_best_metric_results(dataset_name)
        self.metric_results = {metric: 0 for metric in self.metric_results}

        cnt = 0
        bar = tqdm(dataloader,
            desc=f'{dataset_type} testing', 
            unit='batch', 
            ncols=80,
            leave=False) 

        for _, val_data in enumerate(bar, start=1):
            self.feed_test_data(val_data)
            if dataset_type == 'Synthetic':
                self.syn_test(current_iter, val_opt)
            elif dataset_type == 'Real_captured':
                self.real_test(current_iter, val_opt)
            elif dataset_type == 'DND':
                self.dnd_test(current_iter, data_path)
            cnt += 1

        if self.is_train:
            save_path = osp.join(self.opt['path']['visualization'], str(current_iter))
        else:
            save_path = self.opt['path']['visualization']

        if dataset_type == 'DND':
            DND_bundle_submissions_raw(data_path, osp.join(save_path))
            DND_bundle_submissions_srgb(data_path, osp.join(save_path))

        current_metric = 0.
        k = self.k.item()
        alpha = self.alpha.item()
        if dataset_type == 'Synthetic':
            out_data = open(save_path + '/noise_level_{:.4f}'.format(k) + '/alpha_{:.2f}'.format(alpha) + '/0_validation.txt', 'a')
            out_data.write(f'\rOverall_{dataset_name}\t')

            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]
                out_data.write(f'{metric}:{current_metric}\t')

            self._log_validation_metric_values(current_iter, 'noise_level{:.4f}_alpha{:.2f}_{}'.format(k, alpha, dataset_type), tb_logger)
            out_data.close()
        elif dataset_type == 'Real_captured' and val_opt.get('use_img') :
            out_data = open(save_path + '/0_validation.txt', 'a')
            out_data.write(f'\rOverall_{dataset_name}\t')

            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]
                out_data.write(f'{metric}:{current_metric}\t')

            self._log_validation_metric_values(current_iter, tb_logger)
            out_data.close()
        return current_metric


    def _log_validation_metric_values(self, current_iter, valset_name, tb_logger):
        log_str = f'Validation {valset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['out_sRGB'] = self.out_sRGB.detach().cpu()

        patch_size = self.opt['datasets']['val']['patch_size']
        crop_border = self.opt['datasets']['val']['crop_border']
        if self.opt['datasets']['val']['central_crop']:
            out_dict['lq_sRGB'] = self.lq_sRGB[:,:,crop_border//2:crop_border//2+patch_size,crop_border//2:crop_border//2+patch_size].detach().cpu()
            out_dict['gt_sRGB'] = self.gt_sRGB[:,:,crop_border//2:crop_border//2+patch_size,crop_border//2:crop_border//2+patch_size].detach().cpu() if hasattr(self, 'gt_sRGB') else None
            out_dict['ref_sRGB'] = self.ref_sRGB[:,:,crop_border//2:crop_border//2+patch_size,crop_border//2:crop_border//2+patch_size].detach().cpu() if hasattr(self, 'ref_sRGB') else None
        else:
            _,_,h,w = self.lq_sRGB.shape
            out_dict['lq_sRGB'] = self.lq_sRGB[:,:,crop_border//2:h-crop_border//2,crop_border//2:w-crop_border//2].detach().cpu()
            out_dict['gt_sRGB'] = self.gt_sRGB[:,:,crop_border//2:h-crop_border//2,crop_border//2:w-crop_border//2].detach().cpu() if hasattr(self, 'gt_sRGB') else None
            out_dict['ref_sRGB'] = self.ref_sRGB[:,:,crop_border//2:h-crop_border//2,crop_border//2:w-crop_border//2].detach().cpu() if hasattr(self, 'ref_sRGB') else None

        return out_dict

    def save(self, epoch, current_iter):
        
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
