# Stereo Magnification: Learning View Synthesis using Multiplane Images

This code accompanies the paper

Stereo Magnification: Learning View Synthesis using Multiplane Images\
Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe, Noah Snavely\
SIGGRAPH 2018

*Please note that this is not an officially supported Google product.*

## Training the MPI model

The entry python script for training is train.py. The input flags are
specified in two places: 1) `train.py` and 2) `stereomag/loader.py`.

The input flag `which_color_pred` specifies how to predict the color image at
each MPI plane:

`bg` [default] - Our default model. The network predicts: 1) weights for blending
    the background and foreground (reference source image) color images at each
    plane, 2) the alphas at each plane. 3) a background color image
`fgbg` - Instead of using the reference source as the foreground image, the
    network predicts an extra foreground image for blending with the background
`alpha_only` - No color image (or blending weights) is predicted by the network.
    The reference source image is used as the color image at each MPI plane.
`single` - The network predicts a single color image shared for all MPI planes.
`all` - The network directly outputs the color image at each MPI plane.

You can also specify which loss to use for training: `pixel` or `vgg` (i.e., the
`perceptual loss` as measured by differences in VGG features). Note that when
using the VGG loss, you will need to download the pre-trained VGG model
`imagenet-vgg-verydeep-19.mat` available at

http://www.vlfeat.org/matconvnet/pretrained/#downloading-the-pre-trained-models

The path to this file can be set by the `vgg_model_file` flag in `train.py`.

## Testing the MPI model

The entry python script for testing the models is `test.py`.

One could specify what to output to disk by concatenating one or more of the
following (e.g. with '_'): `src_images`, `ref_image`, `tgt_image`, `psv`, `fgbg`, `poses`,
`intrinsics`, `blend_weights`, `rgba_layers`.

`psv` - the plane sweep volume used as input to the network.
`fgbg` - foreground and background color images (only valid when
    `which_color_pred` is either `fgbg` or `bg`)
`blend_weights` - weights for blending foreground and backgroud color images (only
    valid when `which_color_pred` is either `fgbg` or `bg`)

## Quantitative evaluation

`evaluate.py` contains sample code for evaluating the view synthesis performance
based on the SSIM and PSNR metrics. It assumes that each scene result folder
contains a ground-truth target image `tgt_image_*.png` and the synthesized image
`output_image_*.png`. The script will output a text file summarizing the metrics
inside the folder FLAGS.result_root.

## Pre-trained models from SIGGRAPH'18 paper

Our pre-trained model can be downloaded into the `models` subdirectory by
running the script `bash scripts/download_model.sh`.

## Running the model on a single image pair

To run a trained model on a single image pair to generate an MPI, use
`mpi_from_images.py`. This tool assumes images with the same orientation (as
with a rectified stereo pair), but allows for specifying the (x, y, z) offset
between the images. 

You can find example input stereo pairs and command lines for generating results
in the `examples` directory.

(You must first download the pretrained model or train your own model and place in the `models/` subdirectory)

## Reference examples and results

For reference, you can find additional example input stereo pairs, as well as corresponding output multi-plane images and view synthesis results used in the paper in this [Google drive link](https://drive.google.com/open?id=1CZGJxRl0GK0js0MbL7cn7tHtdRrtnjOB) (772 MB).

## RealEstate10K dataset

We have released the [RealEstate10K dataset](https://google.github.io/realestate10k/) suitable for training and testing the MPI model. Note that due to data restrictions, this is not the same version used in our SIGGRAPH'18 paper. However, we are working on updating the results using this public version.
