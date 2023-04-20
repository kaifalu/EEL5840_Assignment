All necessary packages are numpy, sklearn, torch, skimage, matplotlib.

Procedures to Run test.py are listed as follows:
(1) In Line 75: test_images = np.load('Test_Images.npy'), you should replace Test_Images.npy with your blind test_images file  
(2) In Line 76: true_val = np.load('Test_Labels.npy').T, you should replace Test_Labels.npy with your blind test_labels file
(3) Run the test.py, and then it will output the predicted labels and the test accuracy.