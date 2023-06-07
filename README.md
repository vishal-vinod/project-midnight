# Project Midnight: Domain Adaptive RAW Low-Light Image Enhancement for Smartphone Cameras
**CSE 237D/145 (UCSD Embedded Systems Course) - Spring 2023**

## Team Members
- Vishal Vinod (MS in CSE)
- Shasta Subramanian (BS in ECE)

### [Project Website](https://vishal-vinod.github.io/project-midnight/)

## Abstract
Low-light imaging is a challenging task because of severe noise and low illumination resulting from short-exposure times. Smartphone cameras in particular are low-cost and have higher degradation under darkened environments, with different cameras producing significantly varying images for the same scene. Alleviating this domain gap requires expensive large-scale data capture. To address these drawbacks, we propose to utilize the linear RAW images from a DSLR camera dataset and only a handful of RAW images from a smartphone camera to perform domain adaptive low-light RAW image enhancement. Specifically, we aim to investigate raw-to-sRGB image enhancement, knowledge distillation, and enhanced denoising performance. Project Midnight highlights new insights in utilizing such deep learning techniques to provide an alternative, robust low-light image enhancement implementation. 


## Reproducible Experimentation
For adapted future research we have also created a [Docker Image](https://hub.docker.com/r/vvinodhub/midnight).

## File Descriptions
Within `src`:
* `approach2_mod.py` 

* `data_ops.py` 

* `model_jpeg.py` 

* `run_raw_sony_iphone.sh` 

* `run_raw_sony_pixel.sh` 

* `run_raw_sony_oneplus.sh` 

* `train_raw_sony_pixel.py`

* `train_sony_oneplus.py` 

All files Within `static` are for the [Project Website](https://vishal-vinod.github.io/project-midnight/)

## Project Goals


## Literature Survey
Our preliminary study to validate our findings will be to explore the current state-of-the-art RAW low-light image enhancement techniques and the available datasets. Low-light image enhancement has received significant attention from the computational photography literature, yet only recently has there been branches of research investigating RAW camera sensor data for this task. Further, there have been very few works that investigate cross-camera image enhancement or camera-agnostic denoising techniques. We provide a brief overview of previous work that we explored during our literature survey to potentially fine-tune our research goals:

Several previous work primarily enhance low-light sRGB images to well-illuminated sRGB images  requiring large paired datasets. These methods inherently expect the noise model to be minimal in order to enhance the image and thus have sub-par performance under severe degradation that varies with each camera sensor. Recent methods such as DeepUPE, KnD++, and Zero-DCE show promising image enhancement performance for low-light scenes from the LoL dataset but have not performed well in extreme low-light for RAW images. Attention map based method, attention aggregation method and HDR imaging method have shown promising low-light enhancement performance for sRGB images. LSID proposes a deep learning based method to learn the entire non-linear camera pipeline in an end-to-end manner capable of enhancing RAW images for a single camera dataset. Another technique for image enhancement proposes to decompose the RAW image into the frequency domain to recover information from the low frequency domain and then enhance the details. This is a very different task different from the task of cross-camera adaptive RAW image enhancement. More recently, FSDA-LL proposes a few-shot domain adaptive method for low-light RAW enhancement but do not investigate in detail the enhancement for smartphone cameras or the camera-specific denoising. In this work, we investigate these shortcomings and further experiment with different ranges of exposure ratios for the same scene captured with three different smartphone cameras.

## Key References
- Chen Chen, Qifeng Chen, Jia Xu, and Vladlen Koltun, "Learning to See in the Dark", in CVPR, 2018.

- Prabhakar, V. Vinod, N. Ranjan, and V. B. Radhakrishnan. Few-shot domain adaptation for low light raw image enhancement. In BMVC, 2021.