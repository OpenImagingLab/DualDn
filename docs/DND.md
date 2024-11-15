## Prerequisites

1. Refer to the [INSTALL.md](../INSTALL.md) for instructions on preparing environment and dependencies. 

2. Download [training](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155231343_link_cuhk_edu_hk/EUWR-KgxXD5OsH85ylom4H4BPv2hjYSMAyp4MkopiVnqoQ?e=mfcZBX) and [testing](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155231343_link_cuhk_edu_hk/EfIPJHRaH_VGrxJHD7W60ZEBO79Cet6rKSJsbfQGjue75Q?e=OTDAe0) datasets from One Drive.
  
## Train

1. Put training datasets in `'./datasets/fivek/Raw'` and testing datasets in `'./datasets/DND'`, following:

    ```
    datasets/
    ├── fivek/
    │   └── list_file/
    │       ├── invalid_list.txt
    │       ├── train_list.txt
    │       ├── val_list.txt
    │   └── Raw/
    │       ├── a0001-jmac_DSC1459.dng
    │       ├── a0002-dgw_005.dng
    │       ├── ...
    │       ├── a5000-kme_0204.dng
    |
    ├── DND/
    │   └── list_file/
    │       ├── val_list.txt
    │   └── Raw/
    │       ├── 0001.mat
    │       ├── ...
    │       ├── 0050.mat
    │   ├── info.mat
    │   ├── pixelmasks.mat
    ```

2. Open the `'./options/DualDn_Big.yml'` file, set `gamma_type` in `datasets/val/syn_isp` to `2.2` since we find that DND benchmark generates ground truth images with `x ** 1/2.2` gamma.

3. Run

    ```
    python train_dualdn.py -opt ./options/DualDn_Big.yml
    ```

    - Unlike the training strategy used for [Synthetic](Synthetic.md#) evaluation, we utilize the entire MIT-Adobe Fivek dataset (nearly 5,000 RAW images) to train our model for 300,000 iterations, enhancing the model's performance when evaluating on [DND benchmark](https://noise.visinf.tu-darmstadt.de/benchmark/#overview).
      
    - It's worth noting that previous SOTA denoising models on the [DND benchmark](https://noise.visinf.tu-darmstadt.de/benchmark/#overview), such as [CycleISP](https://github.com/swz30/CycleISP) and [UPI](https://github.com/timothybrooks/unprocessing), were trained with **1,000,000 RAW images**. In comparison, the volume of training RAW images we use here is **relatively small**.


4. For fast validation, we validate 20 synthetic images instead of the real-captured images every 50,000 iterations, since images in DND benchmark are too many to validate.

    - If you'd like to validate directly on DND benchmark images, open the `DualDn_Big.yml` file, set `mode` in `datasets/val/val_datasets/DND` to `true` and set `mode` in `datasets/val/val_datasets/Synthetic` to `false`.
    - We recommend evaluating on DND benchmark images after training using the following test or inference code.

5. Find the training results in `'./experiments'`


## Test

1. After training, you can test DualDn in various testing sets, here we test DND benchmark images for example.
   
2. Run

    ```
    python test_dualdn.py -opt [exp_option_path] --num_iters [iters] --val_datasets ['Synthetic', 'Real_captured', 'DND']
    ```
    
  
     E.g. If you trained DualDn with `'DualDn_Big.yml'` for `300000` iterations, and want to test it on `DND` datasets:


    ```
    python test_dualdn.py -opt ./experiments/DualDn_Big/DualDn_Big.yml --num_iters 300000 --val_datasets DND
    ```

3. Find the testing results in `'./results'` <br>
   After testing, the file structure should follows:

    ```
    results/DualDn_Big
    ├── Raw/
    │   └── bundled/
    │       ├── 0001.mat
    │       ├── ...
    │       ├── 0050.mat
    │   ├── 0001_01.mat
    │   ├── ...
    │   ├── 0050_20.mat
    |
    ├── sRGB/
    │   └── bundled/
    │       ├── 0001.mat
    │       ├── ...
    │       ├── 0050.mat
    │   ├── 0001_01.mat
    │   ├── ...
    │   ├── 0050_20.mat
    |
    ├── visuals/
    │   ├── 0001_01_ours.png
    │   ├── ...
    │   ├── 0050_20_ours.png
    ```
   You can download the files in `'Raw/bundled/'` to upload to the [online DND benchmark](https://noise.visinf.tu-darmstadt.de/) for evaluation on the raw-denoising track, and download the files in `'sRGB/bundled/'` for the sRGB-denoising track evaluation. 
   For a closer look at DualDn’s visual results, refer to the `'/visuals'` folder. <br>
   
   - Note that the DND benchmark currently doesn’t support a dual-denoising evaluation track. Additionally, **DND generates its ground truth images using a simplified ISP compared to our DualDn's ISP** (as explained in the Supplementary).
   - We have not yet been able to contact the DND's owner to obtain the original ISP code. <br>
     🌟 **If the original ISP of DND is given, the PSNR could potentially improve by around 1 dB.**


## Inference

1. For fast inference, you can use the pre-trained DualDn models to process DND benchmark images.

2. Download the [pre-trained model](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155231343_link_cuhk_edu_hk/EeSssinwPSRLvC2zOTdmAd8BLLtF3MaKfFw2kYv25WthkQ?e=bbO0Ql) from One Drive and place it in `'./pretrained_model'`

3. Run

    ```
    python inference_dualdn.py -opt ./options/DualDn_Big.yml --pretrained_model ./pretrained_model/DualDn_Big.pth --val_datasets DND --gamma_type 2.2
    ```
    - gamma_type **MUST** be set to  `'2.2'` since we find that DND benchmark generates ground truth images with `x ** 1/2.2` gamma.

4. Find the inferencing results in `'./results'` <br>
   After inferencing, the file structure should follows:

    ```
    results/DualDn_Big
    ├── Raw/
    │   └── bundled/
    │       ├── 0001.mat
    │       ├── ...
    │       ├── 0050.mat
    │   ├── 0001_01.mat
    │   ├── ...
    │   ├── 0050_20.mat
    |
    ├── sRGB/
    │   └── bundled/
    │       ├── 0001.mat
    │       ├── ...
    │       ├── 0050.mat
    │   ├── 0001_01.mat
    │   ├── ...
    │   ├── 0050_20.mat
    |
    ├── visuals/
    │   ├── 0001_01_ours.png
    │   ├── ...
    │   ├── 0050_20_ours.png
    ```
   You can download the files in `'Raw/bundled/'` to upload to the [online DND benchmark](https://noise.visinf.tu-darmstadt.de/) for evaluation on the raw-denoising track, and download the files in `'sRGB/bundled/'` for the sRGB-denoising track evaluation. 
   For a closer look at DualDn’s visual results, refer to the `'/visuals'` folder. <br>
   
   - Note that the DND benchmark currently doesn’t support a dual-denoising evaluation track. Additionally, **DND generates its ground truth images using a simplified ISP compared to our DualDn's ISP** (as explained in the Supplementary).
   - We have not yet been able to contact the DND's owner to obtain the original ISP code. <br>
     🌟 **If the original ISP of DND is given, the PSNR could potentially improve by around 1 dB.**
