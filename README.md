# AFTigerNet
## Introduction

AFTigerNet is a anchor-free object detection framework originally designed for Amur tiger dataset(ATRW),but it's a general-purpose lightweight network

## install

**Step 0**. Download and install Miniconda from the official website https://docs.conda.io/en/latest/miniconda.html.

**Step 1**.
```shell
conda create --name aftigernet python=3.8 -y
conda activate aftigernet
```

**Step 2.** Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/), e.g.

**Step 3.** Install [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```shell
pip install -U openmim
mim install mmcv-full
```
**Step 4.** Install AF-tigernet.


```shell
git clone https://github.com/liubohan09/AF-Tiger.git
cd AF-Tiger
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```


## train

Download data in https://cvwc2019.github.io/challenge.html

copy trainset and valset to ATRW/images

copy testset to ATRW/test

python tools/train.py AF-tigernet.py

## test

python tools/test.py AF-tigernet.py weights/epoch_300.pth  --eval bbox

## demo

**image_demo.**

```shell
python demo/image_demo.py --img-path AF-tigernet.py weights/epoch_300.pth --out ${OUT_PATH}
```

**video_demo.**

```shell
python demo/video_demo.py --video-path AF-tigernet.py weights/epoch_300.pth --score-thr 0.4 --out show.mp4
# if add --show,  it can save images of all video frames
```

**webcam_demo.**

```shell
python demo/webcam_demo.py AF-tigernet.py weights/epoch_300.pth 
```

## deploy

Convert the model to onnx

```shell
python tools/deployment/mmdet2onnx.py AF-tigernet.py weights/epoch_300.pth --input-img ${IMG_PATH} --output-file ${OUT_PATH}
```

You can also convert the model to ncnn with an online tool https://convertmodel.com/ .
## Thanks

https://github.com/Tencent/ncnn

https://github.com/open-mmlab/mmdetection

https://github.com/implus/GFocal

https://github.com/RangiLyu/nanodet
