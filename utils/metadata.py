# -----------------------------------------------------------------------------------
# [ECCV2024] DualDn: Dual-domain Denoising via Differentiable ISP 
# [Homepage] https://openimaginglab.github.io/DualDn/
# [Author] Originally Written by Ruikang Li, from MMLab, CUHK.
# [License] Absolutely open-source and free to use, please cite our paper if possible. :)
# -----------------------------------------------------------------------------------

import numpy as np

def get_metadata(raw, dataset_type): 
    metadata = {}

    if dataset_type == 'Synthetic' or dataset_type == 'Real_captured':
        metadata['black_level'] = raw.black_level_per_channel
        metadata['white_level_channel'] = raw.camera_white_level_per_channel
        metadata['wb_matrix'] = raw.camera_whitebalance
        metadata['color_desc'] = raw.color_desc
        metadata['daylight_whitebalance'] = raw.daylight_whitebalance
        metadata['num_colors'] = raw.num_colors
        metadata['color_matrix'] = raw.color_matrix
        metadata['color_mask'] = raw.raw_colors_visible
        metadata['raw_pattern'] = raw.raw_pattern
        metadata['rgb_xyz_matrix'] = raw.rgb_xyz_matrix
        metadata['tone_curve'] = raw.tone_curve
        metadata['white_level'] = raw.white_level
        metadata['raw_type'] = raw.raw_type.name
        metadata['ref'] = 'D65'
        
        if metadata['wb_matrix'][0] == 0.0:
            metadata['wb_matrix'] = metadata['daylight_whitebalance']
        if metadata['wb_matrix'][3] == 0.0:
                metadata['wb_matrix'][3] = metadata['wb_matrix'][1]
        
        if metadata['rgb_xyz_matrix'][0:3,0:3].all():
            rgb_xyz_matrix = metadata['rgb_xyz_matrix'][0:3,0:3]
            rgb_xyz_matrix = rgb_xyz_matrix / np.sum(rgb_xyz_matrix, axis=1, keepdims=True)
            metadata['rgb_xyz_matrix'] = np.linalg.inv(rgb_xyz_matrix)
        else:
            color_matrix = metadata['color_matrix'][0:3,0:3]
            xyz_srgb_matrix = np.array([[3.2404542, -1.5371385, -0.4985314],
                            [-0.9692660, 1.8760108, 0.0415560],
                            [0.0556434, -0.2040259, 1.0572252]], dtype=np.float32)
            xyz_srgb_matrix = xyz_srgb_matrix / np.sum(xyz_srgb_matrix, axis=-1, keepdims=True)
            srgb_xyz_matrix = np.linalg.inv(xyz_srgb_matrix)
            rgb_xyz_matrix = np.matmul(srgb_xyz_matrix, color_matrix)
            rgb_xyz_matrix = rgb_xyz_matrix / np.sum(rgb_xyz_matrix, axis=1, keepdims=True)
            metadata['rgb_xyz_matrix'] = rgb_xyz_matrix
    
    elif dataset_type == 'DND':
        metadata['type'] = raw['type'][:].astype(np.uint8).tostring().decode('ascii')
        metadata['black_level'] = raw['blacklevel'][:].repeat(4,axis=1)
        metadata['white_level'] = raw['whitelevel'][:]
        metadata['cfa_pattern'] = raw['pattern'][:].T-1
        
        metadata['wb_matrix'] = 1 / np.append(np.reshape(raw['AsShotNeutral'][:], (1,3)),[raw['AsShotNeutral'][1]]).astype(np.float32)
        metadata['color_matrix1'] = raw['ColorMatrix1'][:]
        metadata['rgb_xyz_matrix1'] = np.reshape(metadata['color_matrix1'], (3,3)).astype(np.float32)
        metadata['color_matrix2'] = raw['ColorMatrix2'][:]
        metadata['rgb_xyz_matrix2'] = np.reshape(metadata['color_matrix2'], (3,3)).astype(np.float32)
        rgb_xyz_matrix = metadata['rgb_xyz_matrix2'][0:3,0:3]
        rgb_xyz_matrix = rgb_xyz_matrix / np.sum(rgb_xyz_matrix, axis=1, keepdims=True)
        metadata['rgb_xyz_matrix'] = np.linalg.inv(rgb_xyz_matrix)
        metadata['color_desc'] = 'RGBG'
        metadata['orientation'] = raw['angle'][:]

    return metadata
