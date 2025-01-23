# 選擇基礎映像，這裡選擇 Ubuntu 20.04
FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

# 設定環境變數
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/miniconda3/bin:$PATH"

# 更新系統並安裝必要工具
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    vim \
    build-essential \
    gcc-7 g++-7 \
    cmake \
    libnccl2=2.8.4-1+cuda11.2 \
    libnccl-dev=2.8.4-1+cuda11.2 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 設定 GCC/G++ 版本
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 10 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 10

# 安裝 Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /root/miniconda3 \
    && rm /tmp/miniconda.sh \
    && conda update -n base -c defaults conda

# 設定工作目錄
WORKDIR /workspace

# 暴露 Jupyter Notebook 默認端口
EXPOSE 8888

# 設置默認指令，啟動 bash
CMD ["/bin/bash"]

# ====== 開始復原環境 =======
# 複製虛擬環境壓縮檔到容器中
# 611410114_pp_test
COPY env/611410114_pp_test/environment.tar.gz /workspace/env/611410114_pp_test/
# 解壓並設置虛擬環境
RUN mkdir -p /root/miniconda3/envs/611410114_pp_test \
    && tar -zxvf /workspace/env/611410114_pp_test/environment.tar.gz -C /root/miniconda3/envs/611410114_pp_test/ \
    && echo "source activate 611410114_pp_test" >> ~/.bashrc
# minusface
COPY env/minusface/environment.tar.gz /workspace/env/minusface/
RUN mkdir -p /root/miniconda3/envs/minusface \
    && tar -zxvf /workspace/env/minusface/environment.tar.gz -C /root/miniconda3/envs/minusface/ 
# ====== 復原資料集 =========
#COPY datasets/agedb_30.bin /workspace/datasets/
#COPY datasets/calfw.bin /workspace/datasets/
#COPY datasets/cfp_ff.bin /workspace/datasets/
#COPY datasets/cfp_fp.bin /workspace/datasets/
#COPY datasets/cplfw.bin /workspace/datasets/
#COPY datasets/lfw.bin /workspace/datasets/
#COPY datasets/vgg2_fp.bin /workspace/datasets/
#COPY datasets/data.tar.xz /workspace/datasets/
# 解壓縮 data.tar.xz 到 data 資料夾
#RUN mkdir -p /workspace/datasets/data \
#    && tar -Jxvf /workspace/datasets/data.tar.xz -C /workspace/datasets/data \
#    && rm /workspace/datasets/data.tar.xz
