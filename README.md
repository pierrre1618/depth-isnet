# Depth-ISNet  

Depth-ISNet is a variant of [ISNet](https://github.com/xuebinqin/DIS) that incorporates depth information as an additional input channel.  

## Overview  

Unlike the original ISNet, which takes a standard 3-channel RGB image as input, Depth-ISNet uses a 4-channel input: **RGB + Depth map**. The depth map is estimated using [DepthAnythingV2](https://github.com/DepthAnything/Depth-Anything-V2).  

This modification improves results compared to the standard 3-channel ISNet trained with the same parameters. Additionally, the model is easily exportable to **ONNX** and **CoreML**, making it suitable for deployment across various platforms.  

## Colab Demo  

You can try training and inference using the following Google Colab notebook:  

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pierrre1618/depth-isnet/blob/main/isnet_depth.ipynb)  
