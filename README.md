# [ECCV2024] DualDn: <br> Dual-domain Denoising via Differentiable ISP
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dualdn-dual-domain-denoising-via/image-denoising-on-dnd)](https://paperswithcode.com/sota/image-denoising-on-dnd?p=dualdn-dual-domain-denoising-via)

[![ECCV](https://img.shields.io/badge/ECCV-2024-B762C1)](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07547.pdf)
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2409.18783)
[![supplement](https://img.shields.io/badge/Supplementary-Material-red)](https://mycuhk-my.sharepoint.com/:b:/g/personal/1155231343_link_cuhk_edu_hk/ES8QQePLkZxCia6JwDJGZOEBJnPZmdKVSO1J_3RGtpNUQw)
![visitors](https://visitor-badge.laobi.icu/badge?page_id=OpenImagingLab.DualDn)

### [Project Page](https://openimaginglab.github.io/DualDn/) <br>
Ruikang Li, Yujin Wang, [Shiqi Chen](https://tangeego.github.io/), Fan Zhang, [Jinwei Gu](https://www.gujinwei.org/), [Tianfan Xue](https://tianfan.info/) <br>

#### News
- **Sept 29, 2024:** Paper accepted at ECCV 2024 😊:
- **Nov 4, 2024:** Training and inferencing code released 🌹:

<hr />

> **Abstract:** *There are two typical ways to inject a denoiser into the Image Signal Processing (ISP) pipeline: applying a denoiser directly to captured raw frames (raw domain) or to the ISP's output sRGB images (sRGB domain).
However, both approaches have their limitations. Residual noise from raw-domain denoising can be amplified by the subsequent ISP processing, and the sRGB domain struggles to handle spatially varying noise since it only sees noise distorted by the ISP. Consequently, most raw or sRGB domain denoising works only for specific noise distributions and ISP configurations.
Unlike previous single-domain denoising, DualDn consists of two denoising networks: one in the raw domain and one in the sRGB domain. The raw domain denoising adapts to sensor-specific noise as well as spatially varying noise levels, while the sRGB domain denoising adapts to ISP variations and removes residual noise amplified by the ISP. Both denoising networks are connected with a differentiable ISP, which is trained end-to-end and discarded during the inference stage.* 
<hr />

## Network Architecture

<img src = "docs/static/images/intro.svg"  width="60%">


## Training and Evaluation

Last updated (07/11/2024 01:15)
