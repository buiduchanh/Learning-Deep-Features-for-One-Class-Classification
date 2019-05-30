# DeepOneClass


Implementation of the paper <b>Learning Deep Features for One-Class Classification</b>, https://arxiv.org/abs/1801.05365.

Deep learning based one class classification code targeting one class image classification. Tests carried out on Abnormal image detection, Novel image detection and Active Authentication reported state of the art results.


Pre-processing
--------------
This code is developed targeting keras framework. Please make sure keras > 2.1 and python 3.5 is installed.
 
Prepare data
--------------
- It is describe in `data.py`. 
The image is read as numpy array and the label has shape ( 1, n) with n is number of class

Training/ Testing
-----------------
1. Training `python train.py`
2. Testing `python predict.py`

