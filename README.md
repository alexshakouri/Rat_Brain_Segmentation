# Rat_Brain_Sementation
Masters Thesis on machine learning methods concering automatic segmentation of rat brain MRI images.

## Required software
'''
-matlab
-python 3.5.2
-Tensorflow/Keras software version 2.1.6
-Cuda 9.0
-cudnn 7.0
-Opencv 3.4.1
'''

## Input 
The input to this model is .img/.hdr T2w MRI's
'''
DFP01 R14-189 D03 T2w.img
DFP01 R14-189 D03 T2w.hdr
'''
## Output
The final output of this model is a segmentation mask of the input images with 14 different brain regions segmented

## Running the software
### Preprocess the images
1. Run the Unetpreprocess.m with the corresponding image folders specified
'''
input: .img/.hdr T2w MRI's
output: 2D .tif image slices
'''
### Train the model
2. run train.py and specify the model and parameters (specified below) with the libraries above installed
''''
input: 2D .tif image slices and specified parameters
output: .h5 weights file
''''
#### Parameters


### Test the model
2. Run test.py (make sure that the parameters that used for training is the same for the testing)
'''
input: 2D .tif image slices you want tested and .h5 weights file
output: 2D segmentation mask of the predictions of the model
'''


'''

