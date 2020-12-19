# Modeling Neural Net
Majorly we are using 2 types of operations 

 - Convolution
 - Pooling
 
First lets understand convolution operation. Every convolution operation has 3 sequential operation as follow:

 - Convolution: We are using 3x3 kernel size and padding of 1 to maintain the size of input output
 - Batch Normalization: Basic function of batch norm layer is to add more contrast to the channel information. Basically it will highlight features so that input for the next level will have highlighted features. 
 - Dropout: It drops fix % of neurons. Dropout layer is use to make sure that every feature is getting learned by some neuron. 

We are also using 1x1 kernel once. The main purpose is to reduce the number of channels. 1x1 kernel is a good channel combiner. It does the weightage average  of the channels.

The architecture used in the notebook is as follow:
(Conv Layer of 3x3)
Input (1,28,28) Output (8,28,28) RF=3x3
Input (8,28,28) Output (16,28,28) RF=5x5
Input (16,28,28) Output (32,28,28) RF=7x7

(1x1 Conv layer reducing channel nos from 32 to 8)
Input (32,28,28) Output (8,28,28) RF=7x7

(Max pooling layer reducing size of channel from 28 to 14)
Input (8,28,28) Output (8,14,14) RF=14x14

(Conv Layers of 3x3)
Input (8,14,14) Output (16,14,14) RF=16x16
Input (16,14,14) Output (16,14,14) RF=18x18

(Max pooling layer reducing size of channel from 14 to 7)
Input (16,14,14) Output (16,7,7) RF=36x36

(Conv layers of 3x3)
Input (16,7,7) Output (16,7,7) RF=38x38
Input (16,7,7) Output (32,7,7) RF=40x40
Input (32,7,7) Output (10,5,5) RF=42x42

In last we are using global average pooling layer of size 5. GAP layers averages the whole channel and output is single value for each channel. So in last we have 10 channels so after GAP layer we are left with 10 values.




# Result

The aim of the problem statement is to achieve 99.40% accuracy on validation set. In the 14th epoch we are getting 99.41% accuracy. 




