In the notebook, I am trying to demonstrate using batch normalization and ghost batch normalization. Along with other regularization techniques like L1 and L2.


The whole code is divided into 3 modules. 

 - Train test data loader: 
	 - It includes downloading datasets and applying transforms to the dataset. Also it returns train and test data loader. 
 - Model file
	 -  It contains model structure for batch norm model and ghost batch norm model. 
 - Training/ Testing function module
	 - It contains train and test utlity functions. 

In the notebook I am integrating all the modules. And comparing all the experiments on the basis of loss and accuracy.

The following are the results of the for all 5 experiments:

 - BN + L1
 - BN + L2
 - BN + L1 + L2
 - GBN
 - GBN + L1 + L2

**Validation accuracy**


![Validation Accuracy](https://github.com/aditdoshi333/EVA_5_Phase1/blob/master/Assignment_6/images/val_acc.pnghttps://github.com/aditdoshi333/EVA_5_Phase1/blob/master/Assignment_6/images/val_acc.png)


 **Validation loss**
 

![Validation Loss](images/val_loss.png?raw=true )


**Misclassified images by GBN model**


![Misclassifications](images/miss_class.png?raw=true )


	 


