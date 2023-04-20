# %classify images of handwritten symbols

""" =======================  Import dependencies ========================== """
import numpy as np
from sklearn import preprocessing
import torch
from skimage.feature import hog
import matplotlib.pyplot as plt

""" ======================  Function definitions ========================== """

def feature_extraction_hog(X):
    X_hog = []
    for i in range (len(X)):
        fd = hog(X[i], orientations=9, pixels_per_cell=(20, 20), cells_per_block=(2, 2), visualize=False, multichannel=False)
        hog_vector = preprocessing.normalize([fd])[0]
        hog_vector = [np.array(hog_vector).reshape((36, 36))]
        X_hog.append(hog_vector)
    X_hog = np.array(X_hog)
    return X_hog

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_1 = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.max_pool2d = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear_1 = torch.nn.Linear(9 * 9 * 64, 128)
        self.linear_2 = torch.nn.Linear(128, 25)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv_2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        pred = self.linear_2(x)

        return pred

def test_func(X):
    # Extract features of test_images
    X_hog = feature_extraction_hog(X)
    
    # Tensor Conversion
    X_hog = torch.from_numpy(X_hog).double()
    test_dataloader = torch.utils.data.DataLoader(X_hog, batch_size=len(X), shuffle=False, drop_last=True)

    # Test the CNN Model
    model = Model().double()
    model.load_state_dict(torch.load("CNN_model.dth"))
    model.eval()

    pred_label = list() # save all predicted labels on the testing data set
    for itr, image in enumerate(test_dataloader):

        pred = model(image.double())
        pred = torch.nn.functional.softmax(pred, dim=1)
        for i, p in enumerate(pred):
            pred_label.append(torch.max(p.data, 0)[1])
    pred_label = torch.Tensor(pred_label).numpy()
    return pred_label
    

""" =======================  Load Test Dataset ======================= """

# Load test dataset
test_images = np.load('Test_Images.npy')      # Input Test_Images.npy file, replace filename
true_val = np.load('Test_Labels.npy').T    # Input Test_Labels.npy file, replace filename


""" =======================  Test the Model ======================= """

# Output predicted labels on the testing dataset
predicted_val = test_func(test_images)

# Model accuracy on the testing dataset
correct_count = 0  
for i in range(len(true_val)):
    if(predicted_val[i] == true_val[i]):
        correct_count += 1   
accuracy = (correct_count / len(true_val))*100
print('Test accuracy {:.2f}'.format(accuracy))