# CMLcGAN
This is the code in our paper 'The diagnosis of chronic myeloid leukaemia with deep adversarial learning'
## System Requirements
We highly recommend using an nVIDIA GPU with >=8GB for testing forward of CML segmentation.

## Installation
### Requirments
* Python3.6+ (The installation tutorial can be found on page: https://www.python.org/getit/)
* matplotlib 
* numpy
* sklearn
* skimage (pip install scikit-image)
* Pytorch1.0+ and torchversion (The installation tutorial can be found on page: https://pytorch.org/)

## Usage
git clone https://github.com/zjuzzl/CML.git  
cd CML

After all the requirements libraries are installed, let's get start. We recommend using Spyder as the code IDE. Just type 'pip install spyder' to install it.
### Step1. CML segmentation
First of all, you need to put the pathological images to be segmented into a folder and create a folder to save the results. (example: 'data/img/', 'data/pred/')
python test.py image_path result_save_path 
(For example: python test.py data/img/ data/pred/ )
You can find the segmented CML result image in 'result_save_path' path

### Step2. Feature extraction
Modify the parameter in 'feature_extract.py', then run it, extracted features will be saved as a '.npy' file.

### Step3. Feature selection and boxplot
feature_checking.py

### Step4. Clinical validation and cross validation
clinical_analysis.py

