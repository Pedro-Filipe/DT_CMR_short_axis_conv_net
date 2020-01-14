# Automatic segmentation of DT-CMR short-axis images with a pre-trained U-Net

## Introduction

This repository contains python code to use a U-Net [1] based convolutional neural network (CNN) pre-trained to segment different regions of a cardiac mid-ventricular short-axis diffusion tensor image.

This CNN is intended to be used in the scan mean image: average of all acquired diffusion images after co-registration. It also seems to work well for individual diffusion images if they are denoised with a non-local means denoising algorithm [2].

![CNN][figure_01]
*CNN*

Requirements:

- CNN HDF5 file that can be downloaded from [here].

Please feel free to use it and commit any suggestions.

[1]: https://en.wikipedia.org/wiki/U-Net
[2]: https://en.wikipedia.org/wiki/Non-local_means
[here]: https://imperialcollegelondon.box.com/s/kyskr9fuo6z81ecvpncauq7xmhxtfkil

[figure_01]: https://github.com/Pedro-Filipe/DT_CMR_short_axis_conv_net/blob/master/figure_01.png
