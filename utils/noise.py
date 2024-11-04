# -----------------------------------------------------------------------------------
# [ECCV2024] DualDn: Dual-domain Denoising via Differentiable ISP 
# [Homepage] https://openimaginglab.github.io/DualDn/
# [Author] Originally Written by Ruikang Li, from MMLab, CUHK.
# [License] Absolutely open-source and free to use, please cite our paper if possible. :)
# -----------------------------------------------------------------------------------

import numpy as np
import os.path as osp

# Only support baseline noise models: gaussian / poisson / gaussian_poisson / heteroscedastic_gaussian
class NoiseModel():
    def __init__(self, noise_model='gaussian', cameras=None, include=None, exclude=None, K_fixed=None, params=None, K_min=None, K_max=None, seed=None):

        assert include is None or exclude is None
        self.cameras = cameras or ['CanonEOS5D4', 'CanonEOS70D', 'CanonEOS700D', 'NikonD850', 'SonyA7S2'] 
        assert noise_model in ['gaussian', 'poisson', 'gaussian_poisson', 'heteroscedastic_gaussian']
        self.model = noise_model
        self.K_fixed = K_fixed
        self.params = params
        self.K_min = K_min
        self.K_max = K_max
        if seed == None:
            self.rand = np.random
        else:
            self.rand = np.random.RandomState(seed)

        if include is not None:
            self.cameras = [self.cameras[include]]
        if exclude is not None:
            exclude_camera = set([self.cameras[exclude]])
            self.cameras = list(set(self.cameras) - exclude_camera)

        self.param_dir = './utils/camera_params'
        # print('[i] NoiseModel with {}'.format(self.param_dir))
        # print('[i] cameras: {}'.format(self.cameras))
        # print('[i] using noise model {}'.format(noise_model))
        self.camera_params = {}
        for camera in self.cameras:
            self.camera_params[camera] = np.load(osp.join(self.param_dir, camera+'_params.npy'), allow_pickle=True).item()


    def __call__(self, image):

        if self.params is None:
            K, g_scale = self._sample_params()
        else:
            K, g_scale = self.params

        if self.model == 'gaussian':
            out = image + self.rand.randn(*image.shape).astype(np.float32) * np.sqrt(np.maximum(g_scale, 1e-10)) 
        elif self.model == 'poisson':
            out = self.rand.poisson(image / K).astype(np.float32) * K
        elif self.model == 'gaussian_poisson':
            out = self.rand.poisson(image / K).astype(np.float32) * K + self.rand.randn(*image.shape).astype(np.float32) * np.sqrt(np.maximum(g_scale, 1e-10)) 
        elif self.model == 'heteroscedastic_gaussian':
            out = image + self.rand.randn(*image.shape).astype(np.float32) * np.sqrt(np.maximum(K * image + g_scale, 1e-10))

        return K, g_scale, out


    def _sample_params(self):

        camera = self.rand.choice(self.cameras)
        camera_params = self.camera_params[camera]

        if self.K_min == None:
            Kmin = camera_params['Kmin']
        else:
            Kmin = self.K_min
        if self.K_max == None:
            Kmax = camera_params['Kmax']
        else:
            Kmax = self.K_max

        if self.K_fixed != None:
            K = self.K_fixed
            log_K = np.log(self.K_fixed)
        else:
            log_K = self.rand.uniform(low=np.log(Kmin), high=np.log(Kmax))
            K = np.exp(log_K)

        log_g_scale = self.rand.standard_normal() * camera_params['Profile-1']['g_scale']['sigma'] * 1 +\
            (camera_params['Profile-1']['g_scale']['slope']+2) * log_K + camera_params['Profile-1']['g_scale']['bias']

        g_scale = np.exp(log_g_scale)

        return (K, g_scale)
