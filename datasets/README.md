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
* 準備anaconda虛擬環境 (參考 Sec. [資料集下載](#Data-Zoo))，所有環境有:
  ```
   611410114_pp_test
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

### Data Zoo 
datasets壓縮檔 下載連結(326G)
conda環境壓縮檔 下載連結(30G)

****
### Model Zoo 
Code + pretrained model 壓縮檔： 
PPFR 下載連結(91G)
sean 下載連結(83G)
****