# %classify images of handwritten symbols

""" =======================  Import dependencies ========================== """
import numpy as np
from sklearn import preprocessing
import torch
from skimage.feature import hog
import matplotlib.pyplot as plt

""" ======================  Function definitions ========================== """

def train_valid_test_split(X,Y,PercentageOfTrain,PercentageOfValid):
    N = len(X)
    rp = np.random.permutation(N)
    training_set_X = X[rp[0:int(PercentageOfTrain*N)]]
    training_set_Y = Y[rp[0:int(PercentageOfTrain*N)]]
    validation_set_X = X[rp[int(PercentageOfTrain*N):int((PercentageOfValid)*N)]]
    validation_set_Y = Y[rp[int(PercentageOfTrain*N):int((PercentageOfValid)*N)]]
    testing_set_X = X[rp[int((PercentageOfValid)*N):]]
    testing_set_Y = Y[rp[int((PercentageOfValid)*N):]]
    return training_set_X, validation_set_X, testing_set_X, training_set_Y, validation_set_Y, testing_set_Y

def feature_extraction_hog(dataset):
    X_hog = []
    for i in range (len(dataset)):
        fd = hog(dataset[i], orientations=9, pixels_per_cell=(20, 20), cells_per_block=(2, 2), visualize=False, multichannel=False)
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

def train_func(train_dataloader, valid_dataloader):
    # Train the CNN Architecture

    model = Model().double() # Build the CNN model
    criterion = torch.nn.CrossEntropyLoss() # multi-classification evaluation metric
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # optimizer and learning rate

    N_epochs = 100
    train_loss = list() # save train loss by epoch
    valid_loss = list() # save valid loss by epoch
    best_valid_loss = 10 # initialize the optimal valid loss
    for epoch in range(N_epochs):
        total_train_loss = 0
        total_valid_loss = 0

        model.train() # training process
        # training
        for itr, (image,label) in enumerate(train_dataloader):

            optimizer.zero_grad()

            pred = model(image.double())
            label = label.type(torch.LongTensor)
            loss = criterion(pred, label)
            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()

        total_train_loss = total_train_loss / (itr + 1)
        train_loss.append(total_train_loss)

        # validation
        model.eval()
        total = 0
        for itr, (image,label) in enumerate(valid_dataloader):

            pred = model(image.double())
            label = label.type(torch.LongTensor)
            loss = criterion(pred, label)
            total_valid_loss += loss.item()

            pred = torch.nn.functional.softmax(pred, dim=1)
            for i, p in enumerate(pred):
                if label[i] == torch.max(p.data, 0)[1]:
                    total = total + 1

        accuracy = total / len(XY_valid)

        total_valid_loss = total_valid_loss / (itr + 1)
        valid_loss.append(total_valid_loss)

        print('\nEpoch: {}/{}, Train Loss: {:.8f}, Val Loss: {:.8f}, Val Accuracy: {:.8f}'.format(epoch + 1, N_epochs, total_train_loss, total_valid_loss, accuracy))

        if total_valid_loss < best_valid_loss:
            best_valid_loss = total_valid_loss
            print("Saving the model state dictionary for Epoch: {} with Validation loss: {:.8f}".format(epoch + 1, total_valid_loss))
            torch.save(model.state_dict(), "CNN_model.dth")

    fig=plt.figure(figsize=(20, 10))
    plt.plot(np.arange(1, N_epochs+1), train_loss, label="Train loss")
    plt.plot(np.arange(1, N_epochs+1), valid_loss, label="Validation loss")
    plt.xlabel('Loss')
    plt.ylabel('Epochs')
    plt.title("Loss Plots")
    plt.legend(loc='upper right')
    plt.show()    

""" ======================  Variable Declaration ========================== """

# 70% for training, 10% for validation, 20% for testing
PercentageOfTrain = 0.7
PercentageOfValid = 0.8


""" =======================  Load Data and Split Data ======================= """

# Load data set
data_uniform_images = np.load('Train_Images.npy')
data_uniform_labels = np.load('Train_Labels.npy').T


# Split images and labels datasets
X_train,X_valid,X_test,Y_train,Y_valid,Y_test = train_valid_test_split(data_uniform_images,data_uniform_labels,PercentageOfTrain,PercentageOfValid)


""" ========================  Feature Extraction ============================= """

X_train_hog = feature_extraction_hog(X_train)
X_valid_hog = feature_extraction_hog(X_valid)
X_test_hog = feature_extraction_hog(X_test)


""" ========================  Data Normalization and Tensor Conversion ============================= """

# Images
X_train_hog = torch.from_numpy(X_train_hog).double()
X_valid_hog = torch.from_numpy(X_valid_hog).double()
X_test_hog = torch.from_numpy(X_test_hog).double()

#Labels
Y_train = torch.from_numpy(Y_train).double()
Y_valid = torch.from_numpy(Y_valid).double()
Y_test = torch.from_numpy(Y_test).double()

XY_train = list(zip(X_train_hog, Y_train))
XY_valid = list(zip(X_valid_hog, Y_valid))
XY_test = list(zip(X_test_hog, Y_test))

train_dataloader = torch.utils.data.DataLoader(XY_train, batch_size=64, shuffle=True, drop_last=True)
valid_dataloader = torch.utils.data.DataLoader(XY_valid, batch_size=32, shuffle=False, drop_last=True)
test_dataloader = torch.utils.data.DataLoader(XY_test, batch_size=32, shuffle=False, drop_last=True)


""" ========================  Train and Test the Model ============================= """

# Train the CNN Model
train_func(train_dataloader, valid_dataloader)

# Test the CNN Model
model = Model().double()
model.load_state_dict(torch.load("CNN_model.dth"))
model.eval()

pred_label = list() # save all predicted labels on the testing data set
results = list() # only save correctly predicted labels on the testing data set
total = 0
for itr, (image,label) in enumerate(test_dataloader):

    pred = model(image.double())
    pred = torch.nn.functional.softmax(pred, dim=1)
    label = label.type(torch.LongTensor)
    for i, p in enumerate(pred):
        pred_label.append(torch.max(p.data, 0)[1])
        if label[i] == torch.max(p.data, 0)[1]:
            total = total + 1
            results.append((image, torch.max(p.data, 0)[1]))

test_accuracy = total / len(XY_test)
print('Test accuracy {:.8f}'.format(test_accuracy))