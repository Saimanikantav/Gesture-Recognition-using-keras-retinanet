# Gesture-Recognition-using-keras-retinanet
This is a tutorial created for the sole purpose of helping you quickly and easily train a hand gesture detector for your own gesture dataset.It is based on the excellent keras-retinanet implementation by fizyr which you should definitely read if you have time. This includes a sample dataset of images of hand gestures like grab and drop but is intended to help you train your on your own dataset. This is a step which is often not well documented and can easily trip up new developers with specific data formatting requirements that aren't at all obvious.
# Installation
The first issues that most deep learning practicioners run into are installation errors, so here's some introductory advice on installing stuff.

Most errors come from version mismatches, so make sure you get this right the first time, it's much faster to just downgrade a component than to assume a higher version will work. The key components you need to install are Nvidia drivers, CUDA, CUDNN, tensorflow, keras and a bunch of python libraries. While you can run this without a GPU, it's so slow that it's not worth it. I will use  python3 for this work.

NVIDIA Drivers. These are the hardest parts in Ubuntu and can easily leave you with a broken OS if you do it wrong. The most reliable method I have found is to download the most recent driver version from https://www.nvidia.com/Download/index.aspx. This will give you a file like NVIDIA-Linux-x86_64-440.82.run. You want to close x server in order to install it (this command will drop you into terminal only mode, no graphics, so have these instructions on a separate computer!), to do this press CTRL+ALT+F1 or CTRL+ALT+F2 then run

'''
sudo service lightdm stop

sudo init 3

cd ~/Downloads

sudo chmod +x [Your Driver .run File]

sudo ./[Your Driver .run File]

'''

CUDA 10.0. Download this from https://developer.nvidia.com/cuda-10.0-download-archive and the steps to install it are quite similar to that of the driver. Do not let it install the Nvidia driver when it asks to only install CUDA.

CTRL+ALT+F1 or CTRL+ALT+F2

sudo service lightdm stop

sudo init 3

cd ~/Downloads

sudo chmod +x [Your CUDA .run File]

sudo ./[Your CUDA .run File]

sudo reboot now
cd ~/Downloads

tar -xzvf [Your CUDNN .tgz File]

sudo cp cuda/include/cudnn.h /usr/local/cuda/include

sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64

sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*

Tensorflow 1
pip install --user --upgrade tensorflow-gpu

Keras (not tensorflow-keras)
sudo pip install keras

# Getting tools
The keras-retinanet repo git clone git@github.com:fizyr/keras-retinanet.git

The labelimg tool git clone git@github.com:tzutalin/labelImg.git

# Creating the data set
The first step to creating your dataset is to pick the format you want to use, we will go with Pascal VOC 2007 here, but you could also use the CSV format or COCO.

There are 3 folders in the Pascal VOC format, JPEGImages, Annotations (which has .xml files) and ImageSets/Main (which has .txt files listing the data).
