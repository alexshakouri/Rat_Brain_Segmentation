# Rat_Brain_Sementation
Masters Thesis on machine learning methods concering automatic segmentation of rat brain MRI images.

## Required software
```
-Matlab
-Python 3.5.2
-Tensorflow/Keras software version 2.1.6
-Cuda 9.0
-Cudnn 7.0
-Opencv 3.4.1
-Skimage 0.13.1
-scipy 1.0.1
-Matplotlib 2.2.2
```

## Input 
The input to this model is .img/.hdr T2w MRI's
```
DFP01 R14-189 D03 T2w.img
DFP01 R14-189 D03 T2w.hdr
```
## Output
The final output of this model is a segmentation mask of the input images with 14 different brain regions segmented

## Running the software
### Preprocess the images
1. Run the Unetpreprocess.m with the corresponding image folders specified
```
input: .img/.hdr T2w MRI's
output: 2D .tif image slices
```
#### Parameters
```
-day | Specify the days in the name of the animal (ex. D03)
-name | The full name of the animal excluding the day (ex. Veh01 R14-192)
-tempFileName,tempFileName2 | file path for the images (Assume the images are in the format of DIBS image dataset)
-output image filename | lines 73-74
```
### Train the model
2. run train.py and specify the model and parameters (specified below) with the libraries above installed
```
input: 2D .tif image slices and specified parameters
output: .h5 weights file
```
#### Parameters
```
train.py
-train_images_path
-valid_images_path
-init_weights_path | if you want to start the model from already trained weights
-weights_path | path to save the weigths
-log_path | save the logs from the training period
-gpu | specify which GPU to use in your computer (default is 0)
-epoch
-batch_size
-base_lr | starting learning rate for the model
-imageDim | input image dimensions
-num_classes | number of output regions you want to segment
-saved model name | line 86
-saved weights name | line 112

data.py
-image_rows
-image_cols
-channels | neighboring slices 
-modalities | how many modalities are avaliable within the data (so far more than one isn't implemented)
-categorical | every pixel gets a different class
-region_mask | for num_classes=1 but the input images has multiple regions segmented

net.py
-batch_norm
-smooth | Augmented DC loss function to help training

```

### Test the model
2. Run test.py (make sure that the parameters that used for training is the same for the testing (data.py, model_dilation.py))
```
input: 2D .tif image slices you want tested and .h5 weights file
output: 2D segmentation mask of the predictions of the model
```
#### Parameters
```
-weights_path
-train_image_path
-test_images_path
-predictions_path
-num_classes
-imSize , input image dimension
-output_rows , output image row dimension
-output_cols , output image column dimention
-gpu

```
