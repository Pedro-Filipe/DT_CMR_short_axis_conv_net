# Automatic segmentation of DT-CMR short-axis images with a pre-trained U-Net

## Introduction

This repository contains a python file that loads a pre-trained U-Net [1] based convolutional neural network (CNN). This CNN was trained to segment different regions of a cardiac mid-ventricular short-axis diffusion tensor image.

This CNN is intended to be used with the scan mean image: average of all acquired diffusion images after co-registration. It also seems to work well for individual diffusion images if they are denoised with a non-local means denoising algorithm [2].

<p align="left">
<img src="https://github.com/ImperialCollegeLondon/DT_CMR_short_axis_conv_net/blob/master/figure_01.png" width="400px"/>
</p>

*A: U-Net based CNN architecture. B: Example of segmented classes in a short-axis image.*

An example scan mean image is provided and this is the output from the segment.py file:

<p align="left">
<img src="https://github.com/ImperialCollegeLondon/DT_CMR_short_axis_conv_net/blob/master/figure_02.png" width="400px"/>
</p>

*Input and output of the segment.py script.*

Requirements:

- CNN HDF5 file can be downloaded from here: [3].
- tensorflow (v1.14), numpy, scipy

Tested in Python 3.6 (anaconda) with macOS Catalina. 

Please feel free to use it and commit any suggestions.

[1]: https://en.wikipedia.org/wiki/U-Net
[2]: https://en.wikipedia.org/wiki/Non-local_means
[3]: https://imperialcollegelondon.box.com/s/kyskr9fuo6z81ecvpncauq7xmhxtfkil

[figure_01]: https://github.com/ImperialCollegeLondon/DT_CMR_short_axis_conv_net/blob/master/figure_01.png
