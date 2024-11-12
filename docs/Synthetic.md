## Prerequisites

1. Refer to the [INSTALL.md](../INSTALL.md) for instructions on preparing environment and dependencies. 

2. Download [training](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155231343_link_cuhk_edu_hk/EUWR-KgxXD5OsH85ylom4H4BPv2hjYSMAyp4MkopiVnqoQ?e=mfcZBX) and [testing](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155231343_link_cuhk_edu_hk/ESu0mEIYmDRFlsm6p6I7BQcBjQKT89iPzph52d0RfbK9Gw?e=bvmyhN) datasets from One Drive.
  

## Train

1. For training, simply place the [training](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155231343_link_cuhk_edu_hk/EUWR-KgxXD5OsH85ylom4H4BPv2hjYSMAyp4MkopiVnqoQ?e=mfcZBX) datasets in `'./datasets/fivek/Raw'`.
   We randomly select 220 relatively clean raw images from it: 200 for training and 20 for testing. The small [testing](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155231343_link_cuhk_edu_hk/ESu0mEIYmDRFlsm6p6I7BQcBjQKT89iPzph52d0RfbK9Gw?e=bvmyhN) set is used for fast inference. <br><br>
   After placing the files, ensure the following structure:

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
    ```

2. Run

   - To train our DuanDn model, run <br>
   
      ```
      python train_dualdn.py -opt ./options/DualDn.yml
      ```

   - To train the raw-denoising model with the same backbone, run <br>
   
      ```
      python train_dualdn.py -opt ./options/Raw_Denoising.yml
      ```

   - To train the sRGB-denoising model with the same backbone, run <br>
   
      ```
      python train_dualdn.py -opt ./options/sRGB_Denoising.yml
      ```
   - You can adjust the default training settings in the `datasets/train` section of the `./options/****.yml` file, including but not limited to parameters such as the random noise level (`K_min`, `K_max`), fixed noise type (`noise_model`),
     random ISP amplification ratio (`alpha__min`, `alpha__max`) and different ISP algorithms (`final_stage`, `demosaic_type`, `gamma_type`).
     
   - You can adjust the default denoising paradigms in the `network` section of the `./options/****.yml` file. This includes parameters such as the denoising domain (`type`), denoising backbone (`backbone_type`), channel number (`c`) and noise map usage (`with_noise_map`).


3. For validation, we evaluate 20 synthetic images and calculating 4 evaluation metrics, that is **PSNR**, **SSIM**, **NIQE** and **LPIPS**.
   - The random seed is fixed, ensuring that the generated synthetic noisy raw remains consistent in each validation round.
   - For faster validation, open the `./options/****.yml` file and set `central_crop` in `datasets/val` to `true`. This setting only validates the central square crop of each image, with a size of `(patch_size, patch_size)`. Afterward, you can choose to test the full image with the following test code.
   - By default, validation uses noise levels `[0.002, 0.02]` and ISP amplification ratio `[0, 0.5]`, creating a total of 4 validation sets (2 noise levels * 2 ratios). 
     You can modify these settings in the `datasets/val` section of the `./options/****.yml` file for custom validation configurations.

4. Find the training results in `'./experiments'`


## Test

1. After training, you can test the trained model in various testing sets, here we test Synthetic images for example.
   
2. Run

    ```
    python test_dualdn.py -opt [exp_option_path] --num_iters [iters] --val_datasets ['Synthetic', 'Real_captured', 'DND']
    ```
    
  
    E.g. If you trained DualDn with `'DualDn.yml'` for `120000` iterations, and want to test it on `Synthetic` datasets: 
   
    ```
    python test_dualdn.py -opt ./experiments/DualDn/DualDn.yml --num_iters 120000 --val_datasets Synthetic
    ```

3. You can modify the test settings in the `datasets/val` section of the experiment options file at `'./experiments/DualDn/DualDn.yml'`, including but not limited to parameters such as the random noise level (`K_min`, `K_max`), fixed noise type (`noise_model`),
   random ISP amplification ratio (`alpha__min`, `alpha__max`) and different ISP algorithms (`final_stage`, `demosaic_type`, `gamma_type`).

4. Find the testing results in `'./results'`


## Inference

1. For fast inference, you can use the pre-trained models to process specific images with selected synthetic noise level and ISP amplification ratio.

2. Download the [pre-trained model](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155231343_link_cuhk_edu_hk/Eb2uUHfx8pRBimlrVRbR0dUB5arCuP6Vx5g3LKxImOUv3w?e=XLXLKC) from One Drive and place it in `'./pretrained_model'`

3. According to the raw file in `'./datasets/fivek/Raw'`, add the correct filenames to `'./datasets/fivek/list_file/val_list.txt'`, with one filename per line.

4. Run

    ```
    python inference_dualdn.py -opt [exp_option_path] --pretrained_model [model_path] --noise_level [[noise1, noise2...]] --alpha [[alpha1, alpha2...]]
    ```
    
  
     E.g. If you download the pre-trained model of `DualDn_Restormer.pth`, and want to test it with noise levels `[0.002, 0.02]` and ISP amplification ratio `[0, 0.5]`:


    ```
    python inference_dualdn.py -opt ./options/DualDn.yml --pretrained_model ./pretrained_model/DualDn_Restormer.pth --noise_level [0.002, 0.02] --alpha [0, 0.5]
    ```

5. Find the inferencing results in `'./results'`
