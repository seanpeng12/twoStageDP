# Two-Stage DP Implementation
<p align="left">
<br>
 <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
 <a href="https://colab.research.google.com/drive/"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
</p>

## 內容
* [簡介](#Introduction)
* [環境準備](#Pre-Requisites)
* [使用](#Usage)
* [臉部對齊 Face Alignment](#Face-Alignment)
* [資料前處理 Data Processing](#Data-Processing)
* [訓練與驗證 Training and Validation](#Training-and-Validation)
* [資料集下載 Data Zoo](#Data-Zoo)
* [預訓練模型下載 Model Zoo](#Model-Zoo)
* [引用](#Citation)
<!-- * [實驗結果](#Achievement)
* [參考資料與致謝](#Acknowledgement) -->


### Pre-Requisites 


* Linux
* anaconda3虛擬環境 + Python 3.7 or higher
* Ubuntu 20.04.4 LTS
* nvidia-driver 510.54
* CUDA version: 11.2
* cuDNN: 8.5.0
* nccl: 2.8.4
* gcc/g++ version: 7.5.0
* cmake version: 3.24.1

需要NVIDIA RTX顯卡，我們採用2-8 張 NVIDIA Quadro RTX A5000 24G做平行訓練,推論可只使用單張RTX 30系列顯卡即可。


1. 安裝docker [Install using the apt repository](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository)
    * (Optional) [建立軟連接](https://blog.csdn.net/forest_long/article/details/130244233?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-0-130244233-blog-138315133.235^v43^pc_blog_bottom_relevance_base3&spm=1001.2101.3001.4242.1&utm_relevant_index=2)節省docker根目錄（/var/lib/docker/）佔用
        ```bash
        sudo ln -sf /media/sean/WD-2TB/docker /var/lib/ 
        ```
3. 安裝[nvidia container toolkit (在docker中使用GPU)](https://blog.tarswork.com/post/install-nvidia-gpu-driver-cuda-cudnn-on-ubuntu-2204)

```bash
# 安裝 apt 安裝源
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 更新
sudo apt-get update

# 安裝 NVIDIA Container Toolkit
sudo apt-get install -y nvidia-container-toolkit

# 設定 Docker 識別 NVIDIA Container Runtime
sudo nvidia-ctk runtime configure --runtime=docker

# 重新啟動 docker 服務
sudo systemctl restart docker

# 測試 NVIDIA Docker 支援
# 執行以下測試命令以確認 GPU 支援是否正常：
docker run --rm --gpus all nvidia/cuda:11.2.2-base nvidia-smi

#(Optional) alter default docker folder
# /var/lib/docker -> /media/sean/WD-2TB/docker (any dir.)
# https://stackoverflow.com/questions/76481801/error-after-moving-dockers-dir-to-ntfs-overlayfs-upper-fs-does-not-support-x
```
3. 建置 Docker 映像 : **two_stage_dp_env**
```bash
sudo docker build -t two_stage_dp_env .
```
5. docker run --gpus all -it --name my_project_container -v /path/to/your/project:/workspace my_project_env
```bash
# 使用環境 
sudo docker run --gpus all -it --name two_stage_dp_container -v .:/workspace two_stage_dp_env
# [Optional]重新進入環境
sudo docker start two_stage_dp_container
sudo docker exec -it two_stage_dp_container bash
```
6. In docker環境中, Restore anaconda3環境(有tar.gz檔的環境可直接解壓) 
```bash
# 此步驟已整合進dockerfile，不須執行
# 進入container ~/miniconda3/envs
# mkdir 611410114_pp_test
# tar -zxvf /workspace/env/611410114_pp_test/environment.tar.gz -C ~/miniconda3/envs/611410114_pp_test/
# source activate 611410114_pp_test
```
7. 進入611410114_pp_test環境，安裝缺失套件:
```bash
pip install opencv-python-headless 
pip install opt_einsum 
pip install astunparse google-pasta libclang termcolor 
pip install filelock 
#riddle 
pip install pyparsing 
pip install certifi==2024.2.2 
export SSL_CERT_DIR=/etc/ssl/certs/
```
8. 進入minusface環境，安裝缺失套件:
```bash
pip install opencv-python-headless 
pip install packaging
```

****
### Usage 

#### 僅預訓練模型測試
* 準備資料集 (參考 Sec. [資料集下載](#Data-Zoo))，檔案結構如下:
  ```bash
  ./datasets/data/  # From data.tar.xz
  ./agedb_30.bin
  ./calfw.bin
  ./cfp_ff.bin
  ./cfp_fp.bin
  ./cplfw.bin
  ./lfw.bin
  ./vgg2_fp.bin
  
  ```
* 準備anaconda虛擬環境 (參考 Sec. [資料集下載](#Data-Zoo))，並參考Pre-Requisites[第六點](#pre-requisites)安裝:
  ```
   611410114_pp_test # 已安裝
   minusface
  ```
  有依以上環境名稱所命名的資料夾，複製到env資料夾下:
  ```
  .../env/環境名稱/
    ├── requirements.txt
    ├── [環境名稱].yml
    └── [環境名稱].tar.gz (此檔部分環境會有)
  ```
  皆採用此[網站](https://blog.csdn.net/u014451778/article/details/137423709)提供方法所備份。
#### 含訓練與測試
* 準備資料集 (參考 Sec. [資料集下載](#Data-Zoo))，檔案結構如下:
  ```bash
  ./datasets/BUPT-EqualizedFace/
  ./datasets/data/  # From data.tar.xz

  ./datasets/Facial-Landmarks-of-Face-Datasets/
  ./datasets/ms1m_raw/
  ./datasets/VGGFace2/
  ./datasets/VGGFace2_RetinaFace/
  ./agedb_30.bin
  ./calfw.bin
  ./cfp_ff.bin
  ./cfp_fp.bin
  ./cplfw.bin
  ./lfw.bin
  ./vgg2_fp.bin
  ```
* 準備anaconda虛擬環境 (參考 Sec. [資料集下載](#Data-Zoo))，所有環境有:
  ```
   611410114_arcface_pytorch
   611410114_PFLD_pytorch
   611410114_pp_test
   611410114_retinaface
   611410114_tf2.10.0
   611410114_tiny_face_tf2.2.1
   ASE
   cpgan
   e4e_env
   HE
   minusface
   psp_env
   pti_env
   RiDDLE
   ser_fiq
   tf1.15
   tf1.15new
   tf1.15test
   unet
  ```

****
### Face Alignment 
待補...

****
### Data Processing 
待補...

****
### Training and Validation 
* Code與預訓練模型資料夾 (參考 Sec. [預訓練模型下載](#Model-Zoo)), 並確保檔案結構如下:
  ```
  ./home/611410114/RetinaFace/
  ./home/611410114/PFLD/
  ./home/611410114/PPFR/dctdp/
  ./home/611410114/PPFR/duetFace/
  ./home/611410114/PPFR/riddle/
  ./sean/dctdp_new/
  ./sean/dctdp_new_5p/
  ./sean/dctdp_new_5p_trainable/
  ./sean/minusFace/
  ./sean/minusFace_dctdp/
  ```
#### 訓練
* 待補充..

#### 驗證

* 論文Table 4.3中，跑/workspace/dctdp_new_5p/TFace/recognition/test/verification_with_PSNR_SSIM.py，參考註解執行
    > * S-20結果，參考 b20_2c result 註解執行
    > * S-22結果，參考 b22_2c result 註解執行
    > * S-25結果，參考 b25_2c result 註解執行
    > * S-30結果，參考 b30_2c result 註解執行
docker
    

* 論文Table 4.5 Performance comparision with SOTA：
    > * ArcFace-IR50結果，跑/workspace/dctdp/TFace/recognition/test/verification_ArcFace.py，參考註解執行
    > * DCTDP結果，跑/workspace/dctdp_new/TFace/recognition/test/verification.py，參考註解執行
    > * DuetFace結果，跑/workspace/duetFace/TFace/recognition/test/verification_DuetFace.py，參考註解執行
    > * RiDDLE結果，跑/workspace/riddle/minusFace/TFace/recognition/test/verification_riddle.py，參考註解執行 <i style="color:red;">GPU VRAM需大於8G!</i>
    > * MinusFace結果，跑/workspace/minusFace/TFace/recognition/test/verification_minusface.py，參考註解執行
    > * S-(avg) + F-b(0.5)結果，使用Table 4.3 S-20, S-22, S-25, S-30結果做average。

* 論文Figure 4.2中，由test_dctdp_unet.py中main()產生，需參考註解執行：
    > * 圖(1) = /workspace/minusFace_dctdp/TFace/recognition/deploy/converter/dctdp_unet/Noise_leonardo_dicaprio_RGB_112.jpg
    > * 圖(2) = /workspace/dctdp_new_5p/TFace/recognition/deploy/converter/dctdp_pth2onnx/noisy_img_20240119.jpg
    > * 圖(3) = /workspace/minusFace_dctdp/TFace/recognition/deploy/converter/dctdp_unet/bilibilibilibilibili.jpg

* 論文Figure 4.3中，由/workspace/minusFace_dctdp/TFace/recognition/deploy/converter/dctdp_unet/test_dctdp_unet.py中main()產生，圖片於同資料夾：
    > * 第一列after_UNET_DCT+M 系列照片
    > * 第一列after_UNET_b20_DCT+M 系列照片 
    > * 第一列after_UNET_b30_DCT+M 系列照片

* 論文Table 4.6中，將Figure 4.3中產生的所有系列照片用test_dctdp_unet.py中onlyPSNR_SSIM()跑過一次。
* 論文Table 4.7中，參考Table 4.6填入。
* 論文Table 5.1中:
    > * **w./o. aux loss**:
    > ./home/611410114/PPFR/dctdp/TFace/recognition/test/verification.py，參考註解執行
    > * **w. aux loss**:
    >  ./sean/dctdp_new/TFace/recognition/test/verification.py，參考註解執行
* 論文Table 5.2中，跑./sean/dctdp_new_5p/TFace/recognition/test/verification_with_PSNR_SSIM.py，參考註解執行
    > * S-20結果，參考 b20_2c result / b20_4c result / b20_8c result 註解執行

* 論文Figure B.2中，圖片在./home/611410114/PPFR/dctdp/TFace/recognition/tasks/dctdp/experimental_images資料夾：
    > * LE_perturbation_rmDC_budget0.5_sensitivity1.jpg 
    > * ...
    > * ...

 
****
### Data Zoo 
datasets壓縮檔 下載連結(326G)
conda環境壓縮檔 下載連結(30G)

****
### Model Zoo 
Code + pretrained model 壓縮檔： 
PPFR 下載連結(91G)
sean 下載連結(83G)
****
