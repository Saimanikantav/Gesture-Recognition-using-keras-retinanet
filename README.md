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

# Creating images
The approach is to create a completely new data which is application specific
for this thesis which is achieved by slicing of the video file that has been
recorded by a  camera into frames and later labeling the gestures
by a tool called LabelImg.

Make the data directory mkdir ~/RetinanetTutorial/Data

Select 15 of the images in ~/RetinanetTutorial/Raw_Data and copy these into the data directory called ~/RetinanetTutorial/Data.

The names will all be out of order, and while it's not actually neccessary I like to batch rename them at this point which can save a lot of time in future if you need to combine datasets. Do this using the thunar tool.

sudo apt-get install thunar

cd ~/RetinanetTutorial/Data

thunar

Then CTRL-a to select everything, right click on one image and select rename.

Make sure the settings at the bottom are as shown, then click Rename Files. If you add data in future, increase the Batch#_ number and you can just copy the new files into the VOC folders. Now close thunar.

Once the JPG files are renamed we can set up the final VOC folder structure and move them into it using

mkdir -p ~/RetinanetTutorial/PlumsVOC/JPEGImages

mkdir ~/RetinanetTutorial/PlumsVOC/Annotations

mkdir -p ~/RetinanetTutorial/PlumsVOC/ImageSets/Main

cp ~/RetinanetTutorial/Data/* ~/RetinanetTutorial/PlumsVOC/JPEGImages

# Label Img
To make the Annotations folder we will use the labelimg tool. You have already git cloned that so you should be able to run

cd ~/RetinanetTutorial/labelImg

sudo apt-get install pyqt5-dev-tools

sudo pip3 install -r requirements/requirements-linux-python3.txt

make qt5py3

python3 labelImg.py

Make sure to tick the View > Auto Save Mode checkbox then click Open Dir and set it to ~/RetinanetTutorial/Gesture/JPEGImages, Change Save Dir and set it to ~/RetinanetTutorial/Gesture/Annotations. Ensure PascalVOC is selected under </> and then click Create\nRectBox. Carefully label the hand gestures and assaign the label name to the gesture i.e grab or drop.


Now you can start labelling! The W key adds a new bounding box, you can select and delete them using the list on the right. You can also zoom with CTRL-Scroll and can grab the bounding box corners to adjust them. Click Next Image when you are done labelling each one and you should see .xml files start to appear in the Annotations folder. When you get to the last image, make sure to manually save it by clicking the Save button.

Go through and label every image in the JPEGImages folder. If you don't want to do labelling, you can extract the PlumsVOC.zip sample dataset from this repo and use that instead, you will still need to do the 'Making ImageSets' step.

# Making image steps

The ImageSets/Main folder needs to contain 3 files. trainval.txt lists every JPEGImages file without the extension, there will already be an xml file in Annotations with the same name as every JPG file. train.txt is a subset of trainval.txt with all the images you want to train on. val.txt is a subset of trainval.txt with all the images you want to test (validate) on. There should be no overlap between train and val.

To make these:

cd ~/RetinanetTutorial/PlumsVOC/ImageSets/Main

ls ../../JPEGImages/ > trainval.txt

sed -i 's/.jpg//g' trainval.txt

touch val.txt

cp trainval.txt train.txt

Then open both train.txt and val.txt side by side. Cut and paste entries from train into test. The split should be around 60% of the total files in train.txt and the rest in val.txt.

You have now created a Pascal VOC format dataset for object detection.

# Setup for Training

To setup and run training use the commands

mkdir -p ~/RetinanetTutorial/TrainingOutput/snapshots

cd ~/RetinanetTutorial/keras-retinanet/

We need to change the data generator which currently expects the default Pascal VOC classes so run

gedit keras_retinanet/preprocessing/pascal_voc.py

and change from line 30 onwards so that it looks like

voc_classes = {
    'grab'        : 0,
    'drop'      : 1
}
save and close that file.

Now we are going to build the keras-retinanet tutorial so we can use it

cd ~/RetinanetTutorial/keras-retinanet/

pip install numpy --user

we are going to system install it, so that our test script can run from anywhere, if you don't want to use testDetector.py you can skip this

pip install . --user

then also build the local copy because we will use that for training it

python setup.py build_ext --inplace

then finally, we are ready to start training

keras_retinanet/bin/train.py --tensorboard-dir ~/RetinanetTutorial/TrainingOutput --snapshot-path ~/RetinanetTutorial/TrainingOutput/snapshots --random-transform --steps 100 pascal ~/RetinanetTutorial/PlumsVOC

we are running with a very small steps value so that you can see the model progress on tensorboard after not many steps. The default value is 10000 and using such as small value will result in creating many snapshot files in ~/RetinanetTutorial/TrainingOutput/snapshots so you may need to delete some of the older ones as it fills up and uses a lot of disk space. If you want to train a useful model, you should set this somewhere between 2000 and 10000 depending on how big your dataset is.

# TrainingTraining will likely around half an hour, depending on your hardware. You will want to open tensorboard to monitor the progress of it. You should also keep an eye on the free disk space where you are saving the model checkpoints, because this can fill up fast and crash your training. Run tensorboard in a new terminal using

tensorboard --logdir ~/RetinanetTutorial/TrainingOutput

then open http://localhost:6006/ in a browser tab.

Tensorboard stats will only show up once a validation step has been run, so initially it will say "No scalar data was found" which is normal.

One final thing to check is that the GPU is actually being used, you can do this by running

nvidia-smi

and looking at the memory usage, which should be around 90% of your GPU. If it is more like 20% there was probably a CUDA error which prevented you from using the GPU, check the messages at the top of the terminal just after you run the train.py script and look for library import errors.

Train until the tensorboard curves flatten out.


# Deploying
Once training has finished you want to grab the best (lowest loss) model and convert this to inference mode. In the tensorboard output find the lowest point on the loss curve (don't use classification_loss by accident), this will have a step number if you mouse over it. The step index is from zero whereas the snapshots start from 1 so add 1 to the step value and find that .h5 file in the snapshots directory you set during training. Copy this file to somewhere you want to keep it then convert it to an inference model.

If the lowest step loss was step 5 the commands are

mkdir ~/RetinanetTutorial/RetinanetModels

cp ~/RetinanetTutorial/TrainingOutput/snapshots/resnet50_pascal_06.h5 ~/RetinanetTutorial/RetinanetModels/GestureTraining.h5

cd ~/RetinanetTutorial/keras-retinanet/

keras_retinanet/bin/convert_model.py ~/RetinanetTutorial/RetinanetModels/PlumsTraining.h5 ~/RetinanetTutorial/RetinanetModels/GestureInference.h5

Before you can run the testDetector.py script you will need to set a few things in it. Open it for editing with

cd ~/RetinanetTutorial/

gedit testDetector.py

and set the paths on lines 27,28,29 to point to your new model, a test image and where to save the results. If you used a label other than redPlum you will need to edit line 62.

then run it using

python ../Retinanet-Tutorial/testDetector.py



