# submission code for DND benchmark

# This file is part of the implementation as described in the CVPR 2017 paper:
# Tobias Plötz and Stefan Roth, Benchmarking Denoising Algorithms with Real Photographs.

import numpy as np
import scipy.io as sio
from os import path as osp
import os


def DND_bundle_submissions_raw(Path,res_Path):

    ListFolder = osp.join(Path, 'list_file', 'val_list.txt')
    in_folder = osp.join(res_Path, "Raw/")
    out_folder = osp.join(res_Path, "Raw/bundled/")
    try:
        os.mkdir(out_folder)
    except:pass

    israw = True
    eval_version="1.0"

    with open(ListFolder, "r") as ListNames:
        for RootName in ListNames:
            ind = RootName[0:4]
            Idenoised = np.zeros((20,), dtype=object)
            for i in range(20):
                filename = '{}_{:0=2}.mat'.format(ind, i+1)
                s = sio.loadmat(os.path.join(in_folder,filename))
                Idenoised_crop = s["Idenoised_crop"]
                Idenoised[i] = Idenoised_crop
            filename = '{}.mat'.format(ind)
            sio.savemat(os.path.join(out_folder, filename),{"Idenoised": Idenoised,"israw": israw,"eval_version": eval_version},)


def DND_bundle_submissions_srgb(Path,res_Path):

    ListFolder = osp.join(Path, 'list_file', 'val_list.txt')
    in_folder = osp.join(res_Path, "sRGB/")
    out_folder = osp.join(res_Path, "sRGB/bundled/")
    try:
        os.mkdir(out_folder)
    except:pass

    israw = False
    eval_version="1.0"

    with open(ListFolder, "r") as ListNames:
        for RootName in ListNames:
            ind = RootName[0:4]
            Idenoised = np.zeros((20,), dtype=object)
            for i in range(20):
                filename = '{}_{:0=2}.mat'.format(ind, i+1)
                s = sio.loadmat(os.path.join(in_folder,filename))
                Idenoised_crop = s["Idenoised_crop"]
                Idenoised[i] = Idenoised_crop
            filename = '{}.mat'.format(ind)
            sio.savemat(os.path.join(out_folder, filename),{"Idenoised": Idenoised,"israw": israw,"eval_version": eval_version},)