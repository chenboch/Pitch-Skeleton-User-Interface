# Pitch-Skeleton-User-Interface

## Usage
### Installation
1. Clone this repo.
2. Setup conda environment:
    ```
    conda create -n Pitcher python=3.8 -y
    conda activate Pitcher
    pip install -r requirements.txt
    # CUDA 11.8
    pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
    # 需要去環境中將anaconda3\envs\your_envs_name\libiomp5md.dll刪除，不然會跳warning
    conda install -c conda-forge faiss-gpu
    pip install cython_bbox
    ```
3. setup openmim
    ```
    pip install -U openmim
    mim install mmcv-full
    pip install mmcv==2.0.1 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
    mim install "mmdet==3.1.0"
    ```
4. Setup mmpose environment:
    ```
    cd Src\mmpose_main
    pip install -r requirements.txt
    pip install -v -e .
    cd ..\..\
    ```
5. Setup mmyolo environment:
    ```
    cd Src\mmyolo_main
    pip install -r requirements.txt
    pip install -v -e .
    cd ..\..\
    ```
6. Setup mmengine_main environment:
    ```
    mim install mmengine
    ```
7. Setup mmpretrain_main environment:
    ```
    cd Src\mmpretrain_main
    pip install -r requirements.txt
    pip install -v -e .
    cd ..\..\
    ```
8. Setup ByteTrack enviroment:
    ```
    cd Bytetrack
    pip3 install -r requirements.txt
    python3 setup.py develop
    ```
### Data Preparation
To obtain the vitpose、yolo and fast-reid wights, it can be downloaded from the https://drive.google.com/drive/folders/1D7Q5bTnTAfKkfLuppqUo4_8W4t0wrCmP?usp=sharing. The resulting data directory should look like this:

    ${POSE_ROOT}
    |-- Db
    -- |-- checkpoints
            |-- vitpose.pth
            |-- dstapose_384x288.pth
            |-- yolox_tiny_8xb8-300e_coco_20220919_090908-0e40a6fc.pth
    -- |-- output (for output data)
            |-- {video_name}.mp4 (原始影片)
            |-- {video_name}_Sk26.mp4 (將原始影片畫上骨架資訊,vit pose model output)
            |-- {video_name}_Sk17.mp4 (將原始影片畫上骨架資訊,DSTA pose model output)
            |-- {video_name}_Sk26.json (將偵測出來的結果紀錄，裡面包含了人物的bounding box info. 和 26 個關節點位置)
            |-- {video_name}_Sk17.json (將偵測出來的結果紀錄，裡面包含了人物的bounding box info. 和 17 個關節點位置)
    |-- Src

### Demo
1. Demo command
    ```
    python Src\main.py
    ```
2. 2D 相機
    ```
    利用相機去進行骨架偵測
    ```
3. 2D 影片
    ```
    利用影片進行骨架偵測
    ```

