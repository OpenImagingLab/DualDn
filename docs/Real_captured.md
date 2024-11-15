## Prerequisites

1. Refer to the [INSTALL.md](../INSTALL.md) for instructions on preparing environment and dependencies. 

2. Download [training](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155231343_link_cuhk_edu_hk/EUWR-KgxXD5OsH85ylom4H4BPv2hjYSMAyp4MkopiVnqoQ?e=mfcZBX) and [testing](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155231343_link_cuhk_edu_hk/EfpMrXegPqVJiCaflRh5UH0B0hYIJh9WjSbzTtGXz67nwQ?e=qKkICu) datasets from One Drive.
  
3. You can also use your own smartphone-captured images for inference.

    - MUST using smartphone camera's Pro Mode.  👉 [**How to use Pro Mode?**](https://consumer-tkb.huawei.com/weknow/applet/simulator/en-gb00739859/procamera.html)
    - MUST saved both in **RAW** and **standard JPG** format with the **SAME** prefix name.
    
      Specifically, you save 2 files. **RAW** for denoising input, **standard JPG** for unknown ISP color alignment.
      
      E.g.  **RAW:** `'Xiaomi_0001.dng'` and **standard JPG:** `'Xiaomi_0001.jpg'`.
    

## Train

1. Put training datasets in `'./datasets/fivek/Raw'` and testing datasets in `'./datasets/real_capture'`, following:

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
    ├── real_capture/
    │   └── list_file/
    │       ├── val_list.txt
    │   └── Raw/
    │       ├── Xiaomi_0001.dng
    │       ├── ...
    │   └── ref_sRGB/
    │       ├── Xiaomi_0001.jpg
    │       ├── ...
    ```

2. Run

    ```
    python train_dualdn.py -opt ./options/DualDn_Big.yml
    ```
    - Unlike the training strategy used for [Synthetic](Synthetic.md#) evaluation, we utilize the entire MIT-Adobe Fivek dataset (nearly RAW 5,000 images) to train our model for 300,000 iterations, enhancing the model's generalization ability when dealing with unseen real-captured images.
      
    - It's worth noting that previous SOTA denoising models on the [DND benchmark](https://noise.visinf.tu-darmstadt.de/benchmark/#overview), such as [CycleISP](https://github.com/swz30/CycleISP) and [UPI](https://github.com/timothybrooks/unprocessing), were trained with **1,000,000 RAW images**. In comparison, the volume of training RAW images we use here is **relatively small**.

3. For fast validation, we validate 20 synthetic images instead of the real-captured images every 50,000 iterations, since real-captured images are typically 4K or 8K resolution.

    - If you'd like to validate directly on real-captured images, open the `DualDn_Big.yml` file, set `mode` in `datasets/val/val_datasets/Real_captured` to `true` and set `mode` in `datasets/val/val_datasets/Synthetic` to `false`.
    - We recommend evaluating on real-captured images after training using the following test or inference code.

4. Find the training results in `'./experiments'`


## Test

1. After training, you can test DualDn in various testing sets, here we test Real_captured images for example.
   
2. Run

    ```
    python test_dualdn.py -opt [exp_option_path] --num_iters [iters] --val_datasets ['Synthetic', 'Real_captured', 'DND']
    ```
    
  
     E.g. If you trained DualDn with `'DualDn_Big.yml'` for `300000` iterations, and want to test it on `Real_captured` datasets:


    ```
    python test_dualdn.py -opt ./experiments/DualDn_Big/DualDn_Big.yml --num_iters 300000 --val_datasets Real_captured
    ```

3. Find the testing results in `'./results'`


## Inference

1. For fast inference, you can use the pre-trained DualDn models to process your own noisy images captured by smartphones.

2. Download the [pre-trained model](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155231343_link_cuhk_edu_hk/EeSssinwPSRLvC2zOTdmAd8BLLtF3MaKfFw2kYv25WthkQ?e=bbO0Ql) from One Drive and place it in `'./pretrained_model'`

3. Put your orginal raw files in `'./datasets/real_capture/Raw'`, corresponding JPEG files in `'./datasets/real_capture/ref_sRGB'`. Each RAW and JPEG pair must have the same prefix name.

    - MUST using smartphone camera's Pro Mode.  👉 [**How to use Pro Mode?**](https://consumer-tkb.huawei.com/weknow/applet/simulator/en-gb00739859/procamera.html)
    - MUST saved both in **RAW** and **standard JPG** format with the **SAME** prefix name.
    
      Specifically, you save 2 files. **RAW** for denoising input, **standard JPG** for unknown ISP color alignment.
      
      E.g.  **RAW:** `'Xiaomi_0001.dng'` and **standard JPG:** `'Xiaomi_0001.jpg'`.

4. Add the correct filenames to `'./datasets/real_capture/list_file/val_list.txt'`, with one filename per line.

5. Run

    ```
    python inference_dualdn.py -opt ./options/DualDn_Big.yml --pretrained_model ./pretrained_model/DualDn_Big.pth --val_datasets Real_captured
    ```

6. Find the inferencing results in `'./results'`


  - Toy example for inference:
    
      I captured a scene using my smartphone’s **Pro Mode** with the **RAW format** option enabled. This generated two files: `0001.dng` and `0002.jpg`. I rename it with the **SAME** prefix name and 
      place `0001.dng` in  `'./datasets/real_capture/Raw'` and `0001.jpg` in `'./datasets/real_capture/ref_sRGB'`. I then added a line with `0001.dng` in `'./datasets/real_capture/list_file/val_list.txt'` and run the inference code.

<br>
🌟TIPS🌟:

   - Limited with funding, we cannot test DualDn on latest smartphones, which may have different EXIF data in their raw files. <br><br>
     If your results seems worse than the ref_sRGB (smartphone results) or encounter issues like abnormal colors or overly dark images, please **open an issue** on our GitHub with the original raw and JPEG files. <br><br>
     **Your data is valuable to us, and we’re always here to help!** 😊
     
   - You may encounter **some little black holes** in certain areas. That's because we use [BGU](https://people.csail.mit.edu/hasinoff/pubs/ChenEtAl16-bgu.pdf) during inference for color alignment, which downsamples the original images by a default 8x ratio, potentially neglecting local areas. <br><br>
     **To fix this**, open `'./options/DualDn_Big.yml'` and set `bgu_ratio` in `network'` to `4` or even `1`. But instead, this will slow down the inference speed to a certain extent.
     You can also speed up DualDn inference by disabling [BGU](https://people.csail.mit.edu/hasinoff/pubs/ChenEtAl16-bgu.pdf). Open `'./options/DualDn_Big.yml'` and set `BGU` in `datasets/val/Real_captured` to `false`. 
