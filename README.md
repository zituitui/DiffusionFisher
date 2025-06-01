# DiffusionFisher

<div align="center">

This repository is the official implementation of the **ICML 2025** paper:
_"Efficiently Access Diffusion Fisher: Within the Outer Product Span Space"_ 

> **Fangyikang Wang<sup>1,2</sup>, Hubery Yin<sup>2,</sup>, Shaobin Zhuang<sup>2,3</sup>, Huminhao Zhu<sup>1</sup>, <br> Yinan Li<sup>1</sup>, Lei Qian<sup>1</sup>, Chao Zhang<sup>1</sup>, Hanbin Zhao<sup>1</sup>, Hui Qian<sup>1</sup>, Chen Li<sup>2</sup>**
> 
> <sup>1</sup>Zhejiang University <sup>2</sup>WeChat Vision, Tencent Inc. <sup>3</sup>Shanghai Jiao Tong University  

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2505.23264-b31b1b.svg)](https://www.arxiv.org/abs/2505.23264)&nbsp;
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)&nbsp;

</div>

## 🆕 What's New?
### The analytical formulation of [Fisher information](https://en.wikipedia.org/wiki/Fisher_information) in diffusion models.

Let us define the Fisher information of diffused distributions as follows:
```math
F_t(x_t, t) := - \frac{\partial^2}{\partial x_t^2} \log q_t(x_t, t)
```
We have :
```math
F_t(\bm{x}_t, t) = \frac{1}{\sigma_t^2} \bm{I} - \frac{\alpha_t^2}{\sigma_t^4} \left[ 
    \int w(\bm{y}) \bm{y} \bm{y}^\top \, \mathrm{d}q_0
    - \left( \int w(\bm{y}) \bm{y} \, \mathrm{d}q_0 \right) \left( \int w(\bm{y}) \bm{y} \, \mathrm{d}q_0 \right)^\top
\right]
```



