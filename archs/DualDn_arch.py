# -----------------------------------------------------------------------------------
# [ECCV2024] DualDn: Dual-domain Denoising via Differentiable ISP 
# [Homepage] https://openimaginglab.github.io/DualDn/
# [Author] Originally Written by Ruikang Li, from MMLab, CUHK.
# [License] Absolutely open-source and free to use, please cite our paper if possible. :)
# -----------------------------------------------------------------------------------

import torch
import torch.nn as nn
from archs.backbone.restormer import Restormer
from archs.backbone.swinir import SwinIR
from archs.backbone.mirnetv2 import MIRNet_v2
from utils.bgu import bguFit, bguSlice
from utils.pipeline import run_pipeline
from utils.registry import ARCH_REGISTRY

class Concat(nn.Module):
    def __init__(self, in_c, c, num=2, bias=False):
        super(Concat, self).__init__()
        self.num = num
        self.conv_out = nn.Conv2d(2*in_c, c, kernel_size=(1, 1), stride=(1, 1), bias=bias)

    def forward(self, inp_feats):
        if self.num == 1:
            out_feats = self.conv_out(inp_feats)
        elif self.num == 2:
            out_feats = torch.cat([inp_feats[0],inp_feats[1]], dim=1)
            out_feats = self.conv_out(out_feats)

        return out_feats 

@ARCH_REGISTRY.register()
class Raw_Dn(nn.Module):
    def __init__(self, 
        with_noise_map = True,
        in_c = 4,
        out_c = 4,  
        c = 64,
        backbone_type = 'Restormer',
        bgu_ratio = 8,
        bias = False,
        LayerNorm_type = 'BiasFree'
    ):

        super(Raw_Dn, self).__init__()

        self.with_noise_map= with_noise_map
        self.bgu_ratio = bgu_ratio
        if self.with_noise_map:
            self.num = 2
        else:
            self.num = 1

        self.up = nn.PixelShuffle(upscale_factor=2)
        self.down = nn.PixelUnshuffle(downscale_factor=2)
        self.conv_in = nn.Conv2d(in_c, c, kernel_size=(1, 1), stride=(1, 1), bias=bias)
        self.conv_out =  nn.Conv2d(c, out_c, kernel_size=(1, 1), stride=(1, 1), bias=bias)
        self.nm_fuse = Concat(in_c=in_c, c=c, num=self.num, bias=bias)

        if backbone_type == 'Restormer':
            self.backbone = Restormer(dim=c, num_blocks = [4,6,6,8], num_refinement_blocks = 4, heads = [1,2,4,8], ffn_expansion_factor = 2.66, bias = bias, LayerNorm_type = LayerNorm_type, dual_pixel_task = False)
        elif backbone_type == 'SwinIR':
            self.backbone = SwinIR(img_size=128, window_size=8, img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=c, num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2)
        elif backbone_type == 'MIRNet_v2':
            self.backbone = MIRNet_v2(n_feat=c, chan_factor=1.5, n_RRG=4, n_MRB=2, height=3, width=2, bias=bias)

    def forward(self, in_raw, colormask, wb_matrix, rgb_xyz_matrix, ref='D65', gamma_type='Rec709', demosaic_type='AHD', k=None, sigma=None, alpha=0, final_stage='tone_mapping', ref_sRGB=None):

        in_raw = self.down(in_raw)
        colormask = self.down(colormask)

        if self.with_noise_map:
            noise_map = torch.sqrt(torch.clamp(k * in_raw + sigma, min=1e-7))
            x = self.nm_fuse([in_raw, noise_map])
        else:
            x = self.conv_in(in_raw)

        x = self.backbone(x)
        out_raw = self.conv_out(x)+ in_raw 

        colormask = self.up(colormask)
        out_raw = self.up(out_raw)

        in_srgb = run_pipeline(out_raw, {'color_mask':colormask, 'wb_matrix':wb_matrix, 'color_desc':'RGBG', 'rgb_xyz_matrix':rgb_xyz_matrix, 'ref':ref, 'gamma_type':gamma_type, 'demosaic_type':demosaic_type, 'alpha':alpha}, 'normal', final_stage)

        if ref_sRGB != None and not (ref_sRGB == 0).all(): # Only for inference, to keep consistent with ref_sRGB's color
            in_srgb = torch.clamp(in_srgb, min=1e-6, max=1) # For BGU estimation, there may be overflow, set bound.
            bgu_gamma = bguFit(in_srgb, ref_sRGB, self.bgu_ratio)
            bgu_srgb = bguSlice(bgu_gamma, in_srgb)
            in_srgb = torch.from_numpy(bgu_srgb).float().cuda().permute(2,0,1).unsqueeze()

        out_srgb = in_srgb

        return out_raw, out_srgb


@ARCH_REGISTRY.register()
class sRGB_Dn(nn.Module):
    def __init__(self,
        with_noise_map = False,
        in_c = 3,
        out_c = 3, 
        c = 64,
        backbone_type = 'Restormer',
        bgu_ratio = 8,
        bias = False,
        LayerNorm_type = 'BiasFree'
    ):

        super(sRGB_Dn, self).__init__()
        self.up = nn.PixelShuffle(upscale_factor=2)
        self.down = nn.PixelUnshuffle(downscale_factor=2)
        self.conv_out =  nn.Conv2d(c, out_c, kernel_size=(1, 1), stride=(1, 1), bias=bias)
        self.conv_in = nn.Conv2d(in_c, c, kernel_size=(1, 1), stride=(1, 1), bias=bias)
        self.bgu_ratio = bgu_ratio
        if backbone_type == 'Restormer':
            self.backbone = Restormer(dim=c, num_blocks = [4,6,6,8], num_refinement_blocks = 4, heads = [1,2,4,8], ffn_expansion_factor = 2.66, bias = bias, LayerNorm_type = LayerNorm_type, dual_pixel_task = False)
        elif backbone_type == 'SwinIR':
            self.backbone = SwinIR(img_size=128, window_size=8, img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=c, num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2)
        elif backbone_type == 'MIRNet_v2':
            self.backbone = MIRNet_v2(n_feat=c, chan_factor=1.5, n_RRG=4, n_MRB=2, height=3, width=2, bias=bias)

    def forward(self, in_raw, colormask, wb_matrix, rgb_xyz_matrix, ref='D65', gamma_type='Rec709', demosaic_type='AHD', k=None, sigma=None, alpha=0, final_stage='tone_mapping', ref_sRGB=None):

        x = self.down(in_raw)
        out_raw = self.up(x)

        in_srgb = run_pipeline(out_raw, {'color_mask':colormask, 'wb_matrix':wb_matrix, 'color_desc':'RGBG', 'rgb_xyz_matrix':rgb_xyz_matrix, 'ref':ref, 'gamma_type':gamma_type, 'demosaic_type':demosaic_type, 'alpha':alpha}, 'normal', final_stage)

        if ref_sRGB != None and not (ref_sRGB == 0).all(): # Only for inference, to keep consistent with ref_sRGB's color
            in_srgb = torch.clamp(in_srgb, min=1e-6, max=1) # For BGU estimation, there may be overflow, set bound.
            bgu_gamma = bguFit(in_srgb, ref_sRGB, self.bgu_ratio)
            bgu_srgb = bguSlice(bgu_gamma, in_srgb)
            in_srgb = torch.from_numpy(bgu_srgb).float().cuda().permute(2,0,1).unsqueeze()

        srgb = self.conv_in(in_srgb)
        srgb = self.backbone(srgb)
        out_srgb = self.conv_out(srgb) + in_srgb

        return out_raw, out_srgb


@ARCH_REGISTRY.register()
class DualDn(nn.Module):
    def __init__(self, 
        with_noise_map = False,
        in_c = 4,
        out_c = 3,  
        c = 64,
        backbone_type = 'Restormer',
        bgu_ratio = 8,
        bias = False,
        LayerNorm_type = 'BiasFree'
    ):

        super(DualDn, self).__init__()

        self.with_noise_map= with_noise_map
        self.bgu_ratio = bgu_ratio
        if self.with_noise_map:
            self.num = 2
        else:
            self.num = 1

        self.up = nn.PixelShuffle(upscale_factor=2)
        self.down = nn.PixelUnshuffle(downscale_factor=2)
        self.conv_in1 = nn.Conv2d(2*in_c, c, kernel_size=(1, 1), stride=(1, 1), bias=bias)
        self.conv_out1 =  nn.Conv2d(c, in_c, kernel_size=(1, 1), stride=(1, 1), bias=bias)
        self.conv_in2 = nn.Conv2d(self.num*out_c, c, kernel_size=(1, 1), stride=(1, 1), bias=bias)
        self.conv_out2 =  nn.Conv2d(c, out_c, kernel_size=(1, 1), stride=(1, 1), bias=bias)
        self.nm_fuse1 = Concat(in_c=in_c, c=c, num=self.num, bias=bias)
        self.nm_fuse2 = Concat(in_c=out_c, c=c, num=self.num, bias=bias)
        
        if backbone_type == 'Restormer':
            self.backbone1 = Restormer(dim=c, num_blocks = [2,3,3,4], num_refinement_blocks = 2, heads = [1,2,4,8], ffn_expansion_factor = 2.66, bias = bias, LayerNorm_type = LayerNorm_type, dual_pixel_task = False)
            self.backbone2 = Restormer(dim=c, num_blocks = [2,3,3,4], num_refinement_blocks = 2, heads = [1,2,4,8], ffn_expansion_factor = 2.66, bias = bias, LayerNorm_type = LayerNorm_type, dual_pixel_task = False)
        elif backbone_type == 'SwinIR':
            self.backbone1 = SwinIR(img_size=128, window_size=8, img_range=1., depths=[6, 6, 6], embed_dim=c, num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2)
            self.backbone2 = SwinIR(img_size=256, window_size=8, img_range=1., depths=[6, 6, 6], embed_dim=c, num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2)
        elif backbone_type == 'MIRNet_v2':
            self.backbone1 = MIRNet_v2(n_feat=c, chan_factor=1.5, n_RRG=4, n_MRB=1, height=3, width=2, bias=bias)
            self.backbone2 = MIRNet_v2(n_feat=c, chan_factor=1.5, n_RRG=4, n_MRB=1, height=3, width=2, bias=bias)

    def forward(self, raw, colormask, wb_matrix, rgb_xyz_matrix, ref='D65', gamma_type='Rec709', demosaic_type='AHD', k=None, sigma=None, alpha = 0, final_stage='tone_mapping', ref_sRGB=None):

        pack_raw = self.down(raw)
        colormask = self.down(colormask)

        if self.with_noise_map:
            in_raw = pack_raw
            noise_map = torch.sqrt(torch.clamp(k * pack_raw + sigma, min=1e-7))
            in_raw = self.nm_fuse1([pack_raw, noise_map])
        else:
            in_raw = self.conv_in1(pack_raw)

        x = self.backbone1(in_raw)
        out_raw = self.conv_out1(x)+ pack_raw 

        colormask = self.up(colormask)
        out_raw = self.up(out_raw)

        if self.with_noise_map:
            rgb_noise_map = torch.sqrt(torch.clamp(k * raw + sigma, min=1e-7))
            rgb_noise_map = run_pipeline(rgb_noise_map, {'color_mask':colormask, 'wb_matrix':wb_matrix, 'color_desc':'RGBG', 'rgb_xyz_matrix':rgb_xyz_matrix, 'ref':ref, 'gamma_type':gamma_type, 'demosaic_type':demosaic_type, 'alpha':alpha}, 'normal', final_stage)
            skip_srgb = run_pipeline(out_raw, {'color_mask':colormask, 'wb_matrix':wb_matrix, 'color_desc':'RGBG', 'rgb_xyz_matrix':rgb_xyz_matrix, 'ref':ref, 'gamma_type':gamma_type, 'demosaic_type':demosaic_type, 'alpha':alpha}, 'normal', final_stage)

            if ref_sRGB != None and not (ref_sRGB == 0).all(): # Only for inference, to keep consistent with ref_sRGB's color
                skip_srgb = torch.clamp(skip_srgb, min=1e-6, max=1) # For BGU estimation, there may be overflow, set bound.
                bgu_gamma = bguFit(skip_srgb, ref_sRGB, self.bgu_ratio)
                bgu_srgb = bguSlice(bgu_gamma, skip_srgb)
                rgb_noise_map = bguSlice(bgu_gamma, rgb_noise_map)
                skip_srgb = torch.from_numpy(bgu_srgb).float().cuda().permute(2,0,1).unsqueeze(0)
                rgb_noise_map = torch.clamp(torch.from_numpy(rgb_noise_map).float().cuda().permute(2,0,1).unsqueeze(0), min=1e-6) # After BGU, there may be overflow, set bound.

            in_srgb = self.nm_fuse2([skip_srgb, rgb_noise_map])
        
        else:
            skip_srgb = run_pipeline(out_raw, {'color_mask':colormask, 'wb_matrix':wb_matrix, 'color_desc':'RGBG', 'rgb_xyz_matrix':rgb_xyz_matrix, 'ref':ref, 'gamma_type':gamma_type, 'demosaic_type':demosaic_type, 'alpha':alpha}, 'normal', final_stage)
            
            if ref_sRGB != None and not (ref_sRGB == 0).all(): # Only for inference, to keep consistent with ref_sRGB's color
                skip_srgb = torch.clamp(skip_srgb, min=1e-6, max=1) # For BGU estimation, there may be overflow, set bound.
                bgu_gamma = bguFit(skip_srgb, ref_sRGB, self.bgu_ratio)
                bgu_srgb = bguSlice(bgu_gamma, skip_srgb)
                skip_srgb = torch.from_numpy(bgu_srgb).float().cuda().permute(2,0,1).unsqueeze()
            
            in_srgb = self.conv_in2(skip_srgb)

        srgb = self.backbone2(in_srgb)
        out_srgb = self.conv_out2(srgb) + skip_srgb

        return out_raw, out_srgb
