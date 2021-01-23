# Advanced convolutions

In the assignment there four files and one notebook integrating
all the files and training the model.

* train_test_transform.py : It provides transforms for train 
test data.
  
* train_test_data_loader.py : It provides data loaders for 
train and test based on the transformation config given.
  
* model.py : It provides CNN model

* trainer.py : It has functions related to training and testing 
the model
  
In the notebook we are trying to understand two advanced convolution methods:

* Depthwise Separable Convolution
* Dilated Convolution


## Depthwise Separable Convolution

* In depthwise separable convolution each channel is convoluted separately.

* It reduces number of parameters drastically hence it saves a lot of compute cost.

* It is not as good as simple 3x3. So generally we use it on edge devices only. 


## Dilated Convolution

* Dilated convolution is a way to increase the receptive field with less compute cost.

* It is generally used in dense problems.

* It should be always used with 3x3. NEVER ALONE



### Results of the training

Number of parameters: 349,856 

Highest test accuracy: 82.73%

Remarks: Training accuracy is much higher then test accuracy so there is a overfitting in the model. We can add more conv layers to reduce the overfitting or can use regulization methods.