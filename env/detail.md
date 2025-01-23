# Train & test datasets detail preparation

### BUPT_EqualizedFace

- /workspace/datasets/BUPT-EqualizedFace
    
    ```bash
    原始資料：
    /workspace/datasets/BUPT-EqualizedFace
    
    tfrecords(TFace 5p步驟產生)：
    /workspace/minusFace/TFace/recognition/tools/BUPT_EqualizedFace_tfrecords
    
    轉換相關工具:
    /workspace/minusFace/TFace/recognition/tools
    ```
    

### agedb, cfp, lfw

- /workspace/datasets/data
    
    ```bash
    原始資料：
    /workspace/datasets/data
    內有img.list/pair.list可供使用 請參考data/lfw/.ipynb_checkpoints/gen_all_pairs-checkpoint.ipynb
    以產生資料
    
    下載網頁：LFW/CFP/AGEDB/IJB-B/IJB-C：[https://github.com/IrvingMeng/MagFace](https://github.com/IrvingMeng/MagFace)
    ```
    

### lfw, cfp_fp, agedb_30, calfw, cplfw

- /workspace/datasets
    
    ```bash
        # 皆為.bin檔案
        lfw, lfw_issame = get_val_pair(data_path, 'lfw')
        cfp_fp, cfp_fp_issame = get_val_pair(data_path, 'cfp_fp')
        agedb_30, agedb_30_issame = get_val_pair(data_path, 'agedb_30')
        calfw, calfw_issame = get_val_pair(data_path, 'calfw')
        cplfw, cplfw_issame = get_val_pair(data_path, 'cplfw')
    ```
    

### WIDER_dataset

- /home/ubuntu/611410114/WIDER_dataset

### MS1MV2

- /workspace/datasets/ms1m_raw
    
    ```bash
    原始資料：
    /workspace/datasets/ms1m_raw/ms1m_align_112
    
    tfrecords(TFace 5p步驟產生)：
    # Dataset TFR_ms1m_train_tf, class_num 85742, total_sample_num 2382368
    /workspace/duetFace/TFace/recognition/tools/ms1m_tfrecords
    轉換相關工具:
    /workspace/duetFace/TFace/dataset
    /workspace/duetFace/TFace/recognition/tools
    ```
    

### VGGFace2

- /workspace/datasets/VGGFace2
    
    ```bash
    原始資料：
    # mxnet格式(不含output_images資料夾)
    /workspace/datasets/VGGFace2/faces_vgg_112x112
    
    tfrecords(mxnet直接轉為tfrecord產生)：
    # Dataset TFR_vggface2, class_num 8631, total_sample_num 3137807
    /workspace/datasets/VGGFace2/TFR_vggface2
    轉換相關工具:
    /workspace/datasets/VGGFace2/vgg2tfrecord.py
    ```
    

### VGGFace2_RetinaFace

- /workspace/datasets/VGGFace2_RetinaFace
    
    ```bash
    原始資料：
    # output_images資料夾，使用VGGFace2/mxnet2images.py轉換
    /workspace/datasets/VGGFace2/faces_vgg_112x112/output_images
    
    tfrecords(TFace 5p步驟產生)：
    # Dataset TFR_vggface2, class_num 8631, total_sample_num 1970491
    /workspace/datasets/VGGFace2_RetinaFace/TFR_vggface2
    轉換相關工具:
    /workspace/datasets/VGGFace2_RetinaFace
    /workspace/datasets/VGGFace2_RetinaFace/img2tfrecord_landmarks.py
    ```
    

其他:

- 开源数据
    
    一、LFW/CFP/AGEDB/IJB-B/IJB-C：[https://github.com/IrvingMeng/MagFace](https://github.com/IrvingMeng/MagFace)
    
    二、Trillionpairs：
    
    [http://trillionpairs.deepglint.com/data](http://trillionpairs.deepglint.com/data)
    
    該資料集清洗後的標籤等：[https://github.com/JDAI-CV/FaceX-Zoo/tree/main/training_mode](https://github.com/JDAI-CV/FaceX-Zoo/tree/main/training_mode)
    
    三、Insightface datasets
    [https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_)
    ————————————————
    
    原文链接：[https://blog.csdn.net/xiakejiang/article/details/120174416](https://blog.csdn.net/xiakejiang/article/details/120174416)
    
## 環境
All conda environments
```bash
base                     /home/ubuntu/anaconda3
611410114_PFLD_pytorch     /home/ubuntu/anaconda3/envs/611410114_PFLD_pytorch
611410114_arcface_pytorch  *  /home/ubuntu/anaconda3/envs/611410114_arcface_pytorch
611410114_pp_test        /home/ubuntu/anaconda3/envs/611410114_pp_test
611410114_retinaface     /home/ubuntu/anaconda3/envs/611410114_retinaface
611410114_tf2.10.0       /home/ubuntu/anaconda3/envs/611410114_tf2.10.0
611410114_tiny_face_tf2.2.1     /home/ubuntu/anaconda3/envs/611410114_tiny_face_tf2.2.1
ASE                      /home/ubuntu/anaconda3/envs/ASE
GFPGAN                   /home/ubuntu/anaconda3/envs/GFPGAN
HE                       /home/ubuntu/anaconda3/envs/HE
RiDDLE                   /home/ubuntu/anaconda3/envs/RiDDLE
adic                     /home/ubuntu/anaconda3/envs/adic
arcface                  /home/ubuntu/anaconda3/envs/arcface
arcface-tf2              /home/ubuntu/anaconda3/envs/arcface-tf2
cpgan                    /home/ubuntu/anaconda3/envs/cpgan
dl_course_env            /home/ubuntu/anaconda3/envs/dl_course_env
e4e_env                  /home/ubuntu/anaconda3/envs/e4e_env
face_super_resolution     /home/ubuntu/anaconda3/envs/face_super_resolution
hope                     /home/ubuntu/anaconda3/envs/hope
lowresolution            /home/ubuntu/anaconda3/envs/lowresolution
minusface                /home/ubuntu/anaconda3/envs/minusface
neural_head_avatars_py39     /home/ubuntu/anaconda3/envs/neural_head_avatars_py39
psp_env                  /home/ubuntu/anaconda3/envs/psp_env
pti_env                  /home/ubuntu/anaconda3/envs/pti_env
ser_fiq                  /home/ubuntu/anaconda3/envs/ser_fiq
test                     /home/ubuntu/anaconda3/envs/test
tf1.15                   /home/ubuntu/anaconda3/envs/tf1.15
tf1.15new                /home/ubuntu/anaconda3/envs/tf1.15new
tf1.15test               /home/ubuntu/anaconda3/envs/tf1.15test
tf2.4.1                  /home/ubuntu/anaconda3/envs/tf2.4.1
tf2.6.0                  /home/ubuntu/anaconda3/envs/tf2.6.0
torch                    /home/ubuntu/anaconda3/envs/torch
torch1.8.1               /home/ubuntu/anaconda3/envs/torch1.8.1
unet                     /home/ubuntu/anaconda3/envs/unet
```

## ubuntu/611410114

### 古慶然學長的Code:

backup_paper_demo_code

paper_demo_code

camera

### ArcFace

arcface_model_train_test

insightface_sourceCode

models

testCode

testCode_rtsp

testCode_rtsp_raw

testCode_video

### HE

Pyfhel

TenSEAL

### PFLD

PFLD_pytorch

### PPFR

CLIP2Protect

cpgan

dctcp 尚未採用aux_loss

duetFace

neural_head_avatars

riddle

### RetinaFace

experiment/laplace_noise.py

### Tiny_Faces_in_Tensorflow_v2

## /mnt/sean

### clock

### dctdp_new  採用aux_loss

### dctdp_new_5p 採用spatial (manual) + aux_loss

### MagFace

### minusFace

### SER_FIQ

### unet

