# Automatic segmentation of DT-CMR short-axis images with a pre-trained U-Net

## Introduction

This repository contains a python file that automatically segments cardiac mid-ventricular short-axis diffusion tensor images. It loads a pre-trained U-Net [1] based convolutional neural network (CNN). This CNN was trained by the cardiac diffusion team at the Royal Brompton Hospital.

<p align="left">
<img src="https://github.com/Pedro-Filipe/DT_CMR_short_axis_conv_net/blob/master/figure_01.png" width="600px"/>
</p>

*Figure 1: A: U-Net based CNN architecture. B: Example of segmented classes in a short-axis image.*

## Usage
This CNN is intended to be used with the scan mean image (average of all acquired diffusion images after co-registration). It also seems to work well for individual diffusion images if they are strongly denoised with a non-local means algorithm [2].

The network was trained with STEAM images acquired at 3T. The input image shape must be a rectangular field of view with (256, 96) pixels. For more information please see the following article:

*(coming soon)*

An example scan mean image is provided:

- dti_short_axis_example.png

The output from the segment.py file is:

<p align="left">
<img src="https://github.com/Pedro-Filipe/DT_CMR_short_axis_conv_net/blob/master/figure_02.png" width="400px"/>
</p>

*Figure 2: Input and output of the segment.py script. Each colour represents a different class as shown in figure 1.*

We are confident the U-Net will work with data from other centers provided a similar protocol is used: resolution, field strength and a STEAM based sequence. Although untested, we do not expect good results from a spin-echo based sequence as the image contrast will be quite different, in particular the blood signal in the LV and RV cavity. This is likely to “confuse” the network.

## Requirements

- CNN HDF5 file can be downloaded from here: [3] (400 MB).
- Tensorflow (v1.14), numpy, matplotlib

Tested in Python 3.6 (anaconda) with macOS Catalina.

Please feel free to use it and commit any suggestions. If used in a publication please reference the following paper:

*(coming soon)*

[1]: https://en.wikipedia.org/wiki/U-Net
[2]: https://en.wikipedia.org/wiki/Non-local_means
[3]: https://imperialcollegelondon.box.com/s/kyskr9fuo6z81ecvpncauq7xmhxtfkil

[figure_01]: https://github.com/ImperialCollegeLondon/DT_CMR_short_axis_conv_net/blob/master/figure_01.png
