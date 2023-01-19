# AFTigerNet
## Introduction

AFTigerNet is a anchor-free object detection framework originally designed for Amur tiger dataset(ATRW),but it's a general-purpose lightweight network

## install

git clone https://github.com/liubohan09/AF-Tiger.git

Please install mmdetection at the following website: https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md/#Installation

## train

Download data in https://cvwc2019.github.io/challenge.html

copy trainset and valset to ATRW/images
copy testset to ATRW/test

python tools/train.py AF-tigernet.py

##test


