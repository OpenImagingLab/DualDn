# -----------------------------------------------------------------------------------
# [ECCV2024] DualDn: Dual-domain Denoising via Differentiable ISP 
# [Homepage] https://openimaginglab.github.io/DualDn/
# [Author] Originally Written by Ruikang Li, from MMLab, CUHK.
# [License] Absolutely open-source and free to use, please cite our paper if possible. :)
# -----------------------------------------------------------------------------------

import cv2
import rawpy
import torch
import h5py
import json
import subprocess
import numpy as np
from os import path as osp

from torch.utils import data as data
from torchvision.transforms.functional import normalize

from utils.metadata import get_metadata
from utils.pipeline import run_pipeline
from utils.data import unprocessed_path_from_list, unprocessed_path_from_folder, processed_paths_from_list, processed_paths_from_folder
from utils.transforms import same_crop, random_crop, fixed_crop, window_crop, random_augmentation, un_binning
from utils.img import imfrombytes, img2tensor, padding
from utils.noise import NoiseModel
from utils.file_client import FileClient
from utils.registry import DATASET_REGISTRY

def fix_orientation(image, orientation):
    # 1 = Horizontal(normal)
    # 2 = Mirror horizontal
    # 3 = Rotate 180
    # 4 = Mirror vertical
    # 5 = Mirror horizontal and rotate 270 CW
    # 6 = Rotate 90 CW
    # 7 = Mirror horizontal and rotate 90 CW
    # 8 = Rotate 270 CW

    if type(orientation) is list:
        orientation = orientation[0]

    if orientation == 1:
        pass
    elif orientation == 2:
        image = cv2.flip(image, 0)
    elif orientation == 3:
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif orientation == 4:
        image = cv2.flip(image, 1)
    elif orientation == 5:
        image = cv2.flip(image, 0)
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif orientation == 6:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 7:
        image = cv2.flip(image, 0)
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 8:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return image


@DATASET_REGISTRY.register()
class DualDn_Dataset(data.Dataset):

    def __init__(self, data_opt):
        super(DualDn_Dataset, self).__init__()
        self.name = data_opt['name']
        self.seed = data_opt['seed']
        self.phase = data_opt['phase']
        self.data_path = data_opt['data_path']
        self.file_from_list = data_opt['file_from_list']
        self.dataset_type = data_opt['dataset_type']
        self.patch_size = data_opt['patch_size']
        self.syn_noise = data_opt['syn_noise']
        self.syn_isp = data_opt['syn_isp']
        self.file_client = FileClient(data_opt['io_backend'].pop('type'), **data_opt['io_backend'])

        if self.syn_noise['prepocess'] == True:
            if self.file_from_list:
                self.paths = processed_paths_from_list(data_opt['data_path'], data_opt['phase'], ['gt_Raw', 'lq_Raw'])
            else:
                self.paths = processed_paths_from_folder(data_opt['data_path'], ['gt_Raw', 'lq_Raw'])
        else:
            if self.file_from_list:
                self.paths = unprocessed_path_from_list(data_opt['data_path'], data_opt['phase'], 'Raw')
            else:
                self.paths = unprocessed_path_from_folder(data_opt['data_path'], 'Raw')

        if self.phase == 'train':
            self.std = data_opt['std']
            self.mean = data_opt['mean']
            self.padding = data_opt['padding']
            self.random_crop = data_opt['random_crop']
            self.geometric_augs = data_opt['geometric_augs']
        elif self.phase == 'val':
            self.window_size = data_opt['window_size']
            self.crop_border = data_opt['crop_border'] 
            self.central_crop = data_opt['central_crop']
            if self.dataset_type == 'Real_captured':
                self.bgu = data_opt['val_datasets']['Real_captured']['BGU']

    def __getitem__(self, index):

        ##* Load images. Dimension order: HWC, channel order: BGR, image range: [0, 1] float32.
        if self.syn_noise['prepocess'] == True:
            gt_RawPath = self.paths[index]['gt_RawPath']
            lq_RawPath = self.paths[index]['lq_RawPath']
            try:
                gt_Raw = np.load(gt_RawPath)
            except:
                raise Exception("gt_RawPath {} not working".format(gt_RawPath))
            try:
                lq_Raw = np.load(lq_RawPath)
            except:
                raise Exception("lq_RawPath {} not working".format(lq_RawPath))
            
            ##! TODO: return metadata k, sigma, ...
        else:
            if self.phase == 'train':
                RawPath = self.paths[index]['RawPath']
                img_ind = osp.splitext(osp.basename(RawPath))[0]
                Raw = rawpy.imread(RawPath)
                metadata = get_metadata(Raw, self.dataset_type)
                gt_Raw = run_pipeline(torch.from_numpy(Raw.raw_image_visible.astype(np.float32)).unsqueeze(0).unsqueeze(0), \
                    {'color_mask':torch.from_numpy(metadata['color_mask']).unsqueeze(0).unsqueeze(0), 'black_level':metadata['black_level'], 'white_level':metadata ['white_level']}, 'raw', 'normal').squeeze(0).squeeze(0).unsqueeze(-1).numpy()
                
                ##* augmentation for training
                color_mask = metadata['color_mask']
                if self.padding:
                    gt_Raw, color_mask = padding(gt_Raw, color_mask, patch_size=self.patch_size)
                if self.geometric_augs:
                    gt_Raw, color_mask = random_augmentation(gt_Raw, color_mask)
                if self.random_crop:
                    gt_Raw, color_mask = random_crop(gt_Raw, color_mask, patch_size=self.patch_size, color_mask=color_mask)

                ##* syn noise
                noise_model = NoiseModel(noise_model=self.syn_noise['noise_model'], include=self.syn_noise['noise_params'], K_fixed=None, K_min=self.syn_noise['K_min'], K_max=self.syn_noise['K_max'])
                k, sigma, alpha = torch.zeros([1,1,1]), torch.zeros([1,1,1]), torch.zeros([1,1,1])
                saturation_level = self.syn_noise['saturation_level']
                k[0,:,:], sigma[0,:,:], lq_Raw = noise_model(gt_Raw*saturation_level)
                lq_Raw = np.clip(lq_Raw/saturation_level, 0, 1)
                ref_sRGB = np.zeros((lq_Raw.shape[0], lq_Raw.shape[1], 3)).astype(np.float32)  # generate in data feeding
                if self.syn_isp['alpha'] == None:
                    if self.syn_isp['alpha_min'] != None and self.syn_isp['alpha_max']!= None:
                        alpha[0,:,:] = np.random.uniform(self.syn_isp['alpha_min'], self.syn_isp['alpha_max'])
                    else:
                        alpha[0,:,:] = np.random.uniform(-1,1)
                else: 
                    alpha[0,:,:] = self.syn_isp['alpha']

                ##* normalize
                if self.mean is not None or self.std is not None:
                    normalize(gt_Raw, self.mean, self.std, inplace=True)
                    normalize(lq_Raw, self.mean, self.std, inplace=True)

            elif self.phase == 'val':
                if self.dataset_type == 'Synthetic':
                    RawPath = self.paths[index]['RawPath']
                    img_ind = osp.splitext(osp.basename(RawPath))[0]
                    Raw = rawpy.imread(RawPath)
                    metadata = get_metadata(Raw, self.dataset_type)
                    
                    gt_Raw = run_pipeline(torch.from_numpy(Raw.raw_image_visible.astype(np.float32)).unsqueeze(0).unsqueeze(0), \
                        {'color_mask':torch.from_numpy(metadata['color_mask']).unsqueeze(0).unsqueeze(0), 'black_level':metadata['black_level'], 'white_level':metadata ['white_level']}, 'raw', 'normal').squeeze(0).squeeze(0).unsqueeze(-1).numpy()
                    
                    ##* Flip Raw
                    Flip = Raw.sizes.flip
                    gt_Raw = np.expand_dims(fix_orientation(np.squeeze(gt_Raw, axis=-1), Flip), axis=-1)
                    color_mask = fix_orientation(metadata['color_mask'], Flip)
                    
                    lq_Raw = np.zeros(gt_Raw.shape).astype(np.float32)
                    ref_sRGB = np.zeros((lq_Raw.shape[0], lq_Raw.shape[1], 3)).astype(np.float32)
                
                elif self.dataset_type == 'Real_captured':
                    RawPath = self.paths[index]['RawPath']
                    img_ind = osp.splitext(osp.basename(RawPath))[0]
                    Raw = rawpy.imread(RawPath)

                    # # Another method to get EXIF data
                    # import exifread
                    # with open(RawPath, 'rb') as f:
                    #     tags = exifread.process_file(f)
                    
                    exifdata = json.loads(subprocess.run(['exiftool', '-j', RawPath], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout)[0]
                    noise_profile = torch.tensor([float(x) for x in exifdata.get('NoiseProfile', None).split()])
                    metadata = get_metadata(Raw, self.dataset_type)
                    lq_Raw = run_pipeline(torch.from_numpy(Raw.raw_image_visible.astype(np.float32)).unsqueeze(0).unsqueeze(0), \
                        {'color_mask':torch.from_numpy(metadata['color_mask']).unsqueeze(0).unsqueeze(0), 'black_level':metadata['black_level'], 'white_level':metadata ['white_level']}, 'raw', 'normal').squeeze(0).squeeze(0).unsqueeze(-1).numpy()
                    gt_Raw = np.zeros(lq_Raw.shape).astype(np.float32)
                    
                    ##* Tune to iPhone
                    if exifdata.get('Make') == 'Apple':
                        AETarget = exifdata.get('AETarget', None)
                        AEAverage = exifdata.get('AEAverage', None)
                        LightValue = exifdata.get('LightValue', None)
                        BrightnessValue = exifdata.get('BrightnessValue', None)
                        BaselineExposure = exifdata.get('BaselineExposure', None)
                        SignalToNoiseRatio = exifdata.get('SignalToNoiseRatio', None)
                        ratio = (AETarget / AEAverage) * np.sqrt((LightValue - BrightnessValue) * (BaselineExposure))
                        lq_Raw = np.clip(lq_Raw * max(ratio, 1), 0, 1) 
                        noise_profile[0] = (np.mean(lq_Raw) / 10 ** (SignalToNoiseRatio/10)) * ratio
                        noise_profile[1] = noise_profile[1] * ratio
                    
                    # # save EXIF data as json file for checking
                    # with open('./results/{}.json'.format(img_ind), 'w') as f:
                    #     json.dump(exifdata, f, indent=4)
                    
                    ##* Flip Raw
                    Flip = Raw.sizes.flip
                    lq_Raw = np.expand_dims(fix_orientation(np.squeeze(lq_Raw, axis=-1), Flip), axis=-1)
                    gt_Raw = np.zeros(lq_Raw.shape).astype(np.float32)
                    color_mask = fix_orientation(metadata['color_mask'], Flip)
                    
                    ##* Whether to use BGU to align the color of the result with ref_sRGB
                    if self.bgu: 
                        RootName = osp.basename(RawPath)
                        basename, _ = osp.splitext(RootName)
                        sRGBPath = osp.join(self.data_path, 'ref_sRGB', '{}.jpg'.format(basename))
                        ref_sRGB = imfrombytes(self.file_client.get(sRGBPath, 'ref_sRGB'), float32=True)
                        ref_sRGB = un_binning(ref_sRGB, lq_Raw) # if original phone adopt binning, interpolate ref_sRGB images to its twice size
                    else:
                        ref_sRGB = np.zeros((lq_Raw.shape[0], lq_Raw.shape[1], 3)).astype(np.float32)
                    
                    ref_sRGB, gt_Raw, lq_Raw, color_mask = same_crop(ref_sRGB, gt_Raw, lq_Raw, color_mask) # crop raw images same with phone's sRGB images
                
                elif self.dataset_type == 'DND':
                    RawPath = self.paths[index]['RawPath']
                    img_ind = int(osp.basename(RawPath)[0:4]) - 1
                    DataPath = osp.join(self.data_path, 'info.mat')
                    Info = h5py.File(DataPath,'r')['info']
                    noise_profile = torch.tensor([Info[Info['nlf'][0,img_ind]]['a'][0,0], Info[Info['nlf'][0,img_ind]]['b'][0,0]]).float()
                    metadata = get_metadata(Info[Info['camera'][0,img_ind]], self.dataset_type)
                    camera_type = Info[Info['camera'][0,img_ind]]['type'][:].astype(np.uint8).tostring().decode('ascii')
                    if camera_type == 'Nexus6P':
                        metadata['ref'] = 'D65'
                    else:
                        metadata['ref'] = 'D50'
                    lq_Raw =  np.expand_dims(np.array(h5py.File(RawPath,'r')['Inoisy'][:]).swapaxes(0,1), axis=-1)
                    h,w,_ = lq_Raw.shape
                    metadata['color_mask'] = np.zeros([h, w]).astype(np.uint8)
                    metadata['color_mask'][0:h:2, 0:w:2] = metadata['cfa_pattern'][0,0]
                    metadata['color_mask'][0:h:2, 1:w:2] = metadata['cfa_pattern'][0,1]
                    metadata['color_mask'][1:h:2, 0:w:2] = metadata['cfa_pattern'][1,0]
                    metadata['color_mask'][1:h:2, 1:w:2] = metadata['cfa_pattern'][1,1]
                    gt_Raw = np.zeros(lq_Raw.shape).astype(np.float32)
                    ref_sRGB = np.zeros((lq_Raw.shape[0], lq_Raw.shape[1], 3)).astype(np.float32)

                ##* consistency for validating
                if self.dataset_type == 'Real_captured' or self.dataset_type == 'Synthetic':
                    if self.patch_size != None:
                        h, w, _ = gt_Raw.shape
                        if self.patch_size > h or self.patch_size > w:
                            gt_Raw, lq_Raw, ref_sRGB, color_mask = padding(gt_Raw, lq_Raw, ref_sRGB, color_mask, patch_size=self.patch_size)
                        else:
                            gt_Raw, lq_Raw, ref_sRGB, color_mask = fixed_crop(gt_Raw, lq_Raw, ref_sRGB, color_mask, patch_size=self.patch_size, crop_border=self.crop_border, central_crop=self.central_crop, color_mask=color_mask)
                    else:
                        gt_Raw, lq_Raw, ref_sRGB, color_mask = window_crop(gt_Raw, lq_Raw, ref_sRGB, color_mask, window_size=self.window_size)
                else:
                    color_mask = metadata['color_mask']

                ##* get noise profile
                noise_model = NoiseModel(noise_model=self.syn_noise['noise_model'], include=self.syn_noise['noise_params'], K_fixed=self.syn_noise['noise_level'], seed=self.seed)
                k, sigma, alpha = torch.zeros([1,1,1]), torch.zeros([1,1,1]), torch.zeros([1,1,1])
                saturation_level = self.syn_noise['saturation_level'] 
                if self.dataset_type == 'Synthetic':
                    k[0,:,:], sigma[0,:,:], lq_Raw = noise_model(gt_Raw*saturation_level)
                    lq_Raw = np.clip(lq_Raw/saturation_level, 0, 1)
                    if self.syn_isp['alpha'] == None:
                        raise ValueError(f'During validation with synthetic images, the fixed alpha is necessary.')
                    else:
                        alpha[0,:,:] = self.syn_isp['alpha']
                else:
                    k[0,:,:], sigma[0,:,:], alpha[0,:,:] = noise_profile[0], noise_profile[1], 0

        ##* BGR to RGB, HWC to CHW, Numpy to Tensor
        gt_Raw, lq_Raw= img2tensor([gt_Raw, lq_Raw], bgr2rgb=False, float32=True)
        ref_sRGB = img2tensor(ref_sRGB, bgr2rgb=True, float32=True)
        color_mask = torch.Tensor(color_mask).unsqueeze(0)
        wb_matrix = torch.Tensor(metadata['wb_matrix']).unsqueeze(0).unsqueeze(0)
        rgb_xyz_matrix = torch.Tensor(metadata['rgb_xyz_matrix']).unsqueeze(0)

        ##* ISP Parameters
        final_stage = self.syn_isp['final_stage'] if self.syn_isp['final_stage']!=None else 'tone_mapping'
        demosaic_type = self.syn_isp['demosaic_type'] if self.syn_isp['demosaic_type']!=None else 'AHD'
        gamma_type = self.syn_isp['gamma_type'] if self.syn_isp['gamma_type']!=None else 'Rec709'

        ##* output
        return {'gt_Raw': gt_Raw, 'lq_Raw': lq_Raw, 'ref_sRGB': ref_sRGB, 'img_ind': img_ind, \
            'k': k, 'sigma': sigma, 'alpha': alpha, 'final_stage': final_stage, \
            'color_mask': color_mask, 'wb_matrix': wb_matrix, 'rgb_xyz_matrix': rgb_xyz_matrix, 'ref': metadata['ref'], 'demosaic_type': demosaic_type, 'gamma_type': gamma_type}

    def __len__(self):
        return len(self.paths)
