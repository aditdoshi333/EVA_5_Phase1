**Step 1**

Target:
1.  Get the set-up right
2.  Set Transforms
3.  Set Data Loader
4.  Set Basic Working Code
5.  Get the basic light model right. We will try and avoid changing this skeleton as much as possible.
6.  Results:
    1.  Parameters: 8474
    2.  Best Training Accuracy: 98.71 (15th epoch)
    3.  Best Test Accuracy: 98.79 (14th epoch)
7.  Analysis:
    1.  Model is under fitting. 
    2.  It is a good model can be pushed further.
  
  Link to notebook: https://github.com/aditdoshi333/EVA_5_Phase1/blob/master/Assignment_5/Notebooks/Step_1.ipynb

**Step 2**

Target:
1. Add Batch-norm to increase model efficiency.
2.  Add Dropout to increase model efficiency.
3.  Results:
    1.  Parameters: 8622
    2.  Best Training Accuracy: 99.18 ( 15th epoch)
    3.  Best Test Accuracy: 99.24 ( 12th epoch)
4.  Analysis:
    1.  By using batch norm we are able to get pretty decent accuracy in intial epoch. From 9% (Step 1 epoch 1) to 92%
    2. From the data we can see that in some image rotation is present. So we can use rotation to augment training data.
    3.  There is no over fitting so we can add layer after GAP layer to push it further.
      
  Link to notebook: https://github.com/aditdoshi333/EVA_5_Phase1/blob/master/Assignment_5/Notebooks/Step_2.ipynb

**Step 3**

Target:
1. Increase model capacity at the end (add layer after GAP)
2.  Add rotation, our guess is that 7-10 degrees should be sufficient.
6.  Results:
    1.  Parameters: 9678
    2.  Best Training Accuracy: 98.76 (15th epoch)
    3.  Best Test Accuracy: 99.21 (15th epoch)
7.  Analysis:
    1.  It is a good model can be pushed further. There is no over fitting in the model so it is a good sign.
    3. From the graph we can see that there are oscillations in accuracy. We can fix that using LR scheduler. 
  
  Link to notebook: https://github.com/aditdoshi333/EVA_5_Phase1/blob/master/Assignment_5/Notebooks/Step_3.ipynb


**Step 4**

Target:
1.  Add LR Scheduler

2.  Results:
    1.  Parameters: 9678
    2.  Best Training Accuracy:  98.74 (12th Epoch)
    3.  Best Test Accuracy: 99.47 (11th Epoch)
3.  Analysis:
	1. Finding a good LR scheduler is difficult. Here we are reducing LR by 10th after every 4 epochs. But we are not able to cross 99.5. Still there is a scope of learning in the model as training accuracy is 98.75 only. So probably a good LR scheduler can push it beyond 99.5
  
  Link to notebook:
  https://github.com/aditdoshi333/EVA_5_Phase1/blob/master/Assignment_5/Notebooks/Step_4.ipynb
