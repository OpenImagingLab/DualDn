# -----------------------------------------------------------------------------------
# [ECCV2024] DualDn: Dual-domain Denoising via Differentiable ISP 
# [Homepage] https://openimaginglab.github.io/DualDn/
# [Author] Originally Written by Ruikang Li, from MMLab, CUHK.
# [License] Absolutely open-source and free to use, please cite our paper if possible. :)
# -----------------------------------------------------------------------------------

from os import path as osp
from utils import scandir

def processed_paths_from_list(Path, phase, keys):
    """Generate processed data paths from list.

    Returns:
        list[str]: Returned path list.
    """
    
    ListFolder = osp.join(Path, 'list_file', '{}_list.txt'.format(phase))
    gt_RawKey, lq_RawKey = keys
    RawNames = list(scandir(osp.join(Path, gt_RawKey)))

    paths = []
    with open(ListFolder, "r") as ListNames:
        for ListName in ListNames:
            ListName = ListName[:-1]
            assert ListName in RawNames, f'{ListName} is not in {Path}_Paths.'

            gt_RawPath = osp.join(Path, gt_RawKey, ListName)
            lq_RawPath = osp.join(Path, lq_RawKey, ListName)

            paths.append(dict([(f'{gt_RawKey}Path', gt_RawPath), (f'{lq_RawKey}Path', lq_RawPath)]))

    return paths


def processed_paths_from_folder(Path, keys):
    """Generate processed data paths from folder.

    Returns:
        list[str]: Returned path list.
    """
    gt_RawKey, lq_RawKey = keys
    gt_RawNames, lq_RawNames = list(scandir(osp.join(Path, gt_RawKey))), list(scandir(osp.join(Path, lq_RawKey)))

    assert len(gt_RawNames) == len(lq_RawNames), (f'{gt_RawKey} and {lq_RawKey} datasets have different number of images: '
                                                f'{len(gt_RawNames)}, {len(lq_RawNames)}.')

    paths = []
    for RawName in gt_RawNames:
        gt_RawPath = osp.join(Path, gt_RawKey, RawName)
        lq_RawPath = osp.join(Path, lq_RawKey, RawName)
        assert RawName in lq_RawNames, f'{RawName} is not in {lq_RawPath}Paths.'

        paths.append(dict([(f'{gt_RawKey}Path', gt_RawPath), (f'{lq_RawKey}Path', lq_RawPath)]))
    return paths


def unprocessed_path_from_list(Path, phase, Folder):
    """Generate unprocessed data paths from list.

    Args:
        Path (str): Data Path for all files. 
        phase (str): Refers to the train/val list file.
        Folder (str): Folder path for the corresponding data.  

    Returns:
        list[str]: Returned path list.
    """

    ListFolder = osp.join(Path, 'list_file', '{}_list.txt'.format(phase))
    RootNames = list(scandir(osp.join(Path, Folder)))

    paths = []
    with open(ListFolder, "r") as ListNames:
        for ListName in ListNames:
            ListName = ListName[:-1]
            RootPath = osp.join(Path, Folder, ListName)
            assert ListName in RootNames, f'{ListName} is not in {Path}_Paths.'

            paths.append(dict([(f'{Folder}Path', RootPath)]))

    return paths


def unprocessed_path_from_folder(Path, Folder):
    """Generate unprocessed data paths from folder.

    Args:
        Path (str): Data Path for all files. 
        Folder (str): Folder path for the corresponding data. 
            Note that the final path excludes the invalid Raw file from './list_file/invalid_list.txt'. 

    Returns:
        list[str]: Returned path list.
    """

    InvalidListFolder = osp.join(Path, 'list_file', 'invalid_list.txt')
    RootNames = list(scandir(osp.join(Path, Folder)))

    paths = []
    with open(InvalidListFolder, "r") as ListNames:
        ListNames = ListNames.read().splitlines()
        for RootName in RootNames:
            RootPath = osp.join(Path, Folder, RootName)
            if RootName in ListNames:
                ##! TODO: Process stack RAW
                print('\r', RootName, ' is Stack RAW, cancel process')
            else:
                paths.append(dict([(f'{Folder}Path', RootPath)]))

    return paths