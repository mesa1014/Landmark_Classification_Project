
# Convolutional Neural Networks

## Project: Write an Algorithm for Landmark Classification

---

In this notebook, some template code has already been provided for you, and you will need to implement additional functionality to successfully complete this project. You will not need to modify the included code beyond what is requested. Sections that begin with **'(IMPLEMENTATION)'** in the header indicate that the following block of code will require additional functionality which you must provide. Instructions will be provided for each section, and the specifics of the implementation are marked in the code block with a 'TODO' statement. Please be sure to read the instructions carefully!

> **Note**: Once you have completed all the code implementations, you need to finalize your work by exporting the Jupyter Notebook as an HTML document. Before exporting the notebook to HTML, all the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to **File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question X'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.

>**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut.  Markdown cells can be edited by double-clicking the cell to enter edit mode.

The rubric contains _optional_ "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. If you decide to pursue the "Stand Out Suggestions", you should include the code in this Jupyter notebook.

---
### Why We're Here

Photo sharing and photo storage services like to have location data for each photo that is uploaded. With the location data, these services can build advanced features, such as automatic suggestion of relevant tags or automatic photo organization, which help provide a compelling user experience. Although a photo's location can often be obtained by looking at the photo's metadata, many photos uploaded to these services will not have location metadata available. This can happen when, for example, the camera capturing the picture does not have GPS or if a photo's metadata is scrubbed due to privacy concerns.

If no location metadata for an image is available, one way to infer the location is to detect and classify a discernable landmark in the image. Given the large number of landmarks across the world and the immense volume of images that are uploaded to photo sharing services, using human judgement to classify these landmarks would not be feasible.

In this notebook, you will take the first steps towards addressing this problem by building models to automatically predict the location of the image based on any landmarks depicted in the image. At the end of this project, your code will accept any user-supplied image as input and suggest the top k most relevant landmarks from 50 possible landmarks from across the world. The image below displays a potential sample output of your finished project.

![Sample landmark classification output](/notebook_images/sample_landmark_output.png)


### The Road Ahead

We break the notebook into separate steps.  Feel free to use the links below to navigate the notebook.

* [Step 0](#step0): Download Datasets and Install Python Modules
* [Step 1](#step1): Create a CNN to Classify Landmarks (from Scratch)
* [Step 2](#step2): Create a CNN to Classify Landmarks (using Transfer Learning)
* [Step 3](#step3): Write Your Landmark Prediction Algorithm

---
<a id='step0'></a>
## Step 0: Download Datasets and Install Python Modules

**Note: if you are using the Udacity workspace, *YOU CAN SKIP THIS STEP*. The dataset can be found in the `/data` folder and all required Python modules have been installed in the workspace.**

Download the [landmark dataset](https://udacity-dlnfd.s3-us-west-1.amazonaws.com/datasets/landmark_images.zip).
Unzip the folder and place it in this project's home directory, at the location `/landmark_images`.

Install the following Python modules:
* cv2
* matplotlib
* numpy
* PIL
* torch
* torchvision

---

<a id='step1'></a>
## Step 1: Create a CNN to Classify Landmarks (from Scratch)

In this step, you will create a CNN that classifies landmarks.  You must create your CNN _from scratch_ (so, you can't use transfer learning _yet_!), and you must attain a test accuracy of at least 20%.

Although 20% may seem low at first glance, it seems more reasonable after realizing how difficult of a problem this is. Many times, an image that is taken at a landmark captures a fairly mundane image of an animal or plant, like in the following picture.

<img src="images/train/00.Haleakala_National_Park/084c2aa50d0a9249.jpg" alt="Bird in Haleakalā National Park" style="width: 400px;"/>

Just by looking at that image alone, would you have been able to guess that it was taken at the Haleakalā National Park in Hawaii?

An accuracy of 20% is significantly better than random guessing, which would provide an accuracy of just 2%. In Step 2 of this notebook, you will have the opportunity to greatly improve accuracy by using transfer learning to create a CNN.

Remember that practice is far ahead of theory in deep learning.  Experiment with many different architectures, and trust your intuition.  And, of course, have fun!

### (IMPLEMENTATION) Specify Data Loaders for the Landmark Dataset

Use the code cell below to create three separate [data loaders](http://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader): one for training data, one for validation data, and one for test data. Randomly split the images located at `landmark_images/train` to create the train and validation data loaders, and use the images located at `landmark_images/test` to create the test data loader.

**Note**: Remember that the dataset can be found at `/data/landmark_images/` in the workspace.

All three of your data loaders should be accessible via a dictionary named `loaders_scratch`. Your train data loader should be at `loaders_scratch['train']`, your validation data loader should be at `loaders_scratch['valid']`, and your test data loader should be at `loaders_scratch['test']`.

You may find [this documentation on custom datasets](https://pytorch.org/docs/stable/torchvision/datasets.html#datasetfolder) to be a useful resource.  If you are interested in augmenting your training and/or validation data, check out the wide variety of [transforms](http://pytorch.org/docs/stable/torchvision/transforms.html?highlight=transform)!


```python
### TODO: Write data loaders for training, validation, and test sets
## Specify appropriate transforms, and batch_sizes
import os
import torch
import numpy as np
from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import splitfolders


# load and transform data using ImageFolder
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Use splitfolders
splitfolders.ratio("/data/landmark_images/train", output="train_valid", seed=1337, ratio=(.8, .2), group_prefix=None)

train_data = datasets.ImageFolder('./train_valid/train', transform=train_transform)
valid_data = datasets.ImageFolder('./train_valid/val', transform=test_transform)
test_data = datasets.ImageFolder('/data/landmark_images/test', transform=test_transform)

# print out some data stats
print('Num training images: ', len(train_data))
print('Num validation images: ', len(valid_data))
print('Num test images: ', len(test_data))

# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 32
# percentage of training set to use as validation
valid_size = 0.2

# obtain training indices that will be used for validation
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    sampler=valid_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
    num_workers=num_workers)

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                           num_workers=num_workers, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                          num_workers=num_workers, shuffle=True)

loaders_scratch = {'train': train_loader, 'valid': valid_loader, 'test': test_loader}
```

    Copying files: 4996 files [00:01, 2706.13 files/s]

    Num training images:  3996
    Num validation images:  1000
    Num test images:  1250





**Question 1:** Describe your chosen procedure for preprocessing the data.
- How does your code resize the images (by cropping, stretching, etc)?  What size did you pick for the input tensor, and why?
- Did you decide to augment the dataset?  If so, how (through translations, flips, rotations, etc)?  If not, why not?

**Answer**:
- I used resize + centre crop as we learned in our classroom lectures. I used an input tensor size of 224 x 224 to make it similar to the image input size of VGG16.
- I used random flips and rotations for augmentation. Based on my reviewer comment, it's noted that I shouldn't use augmentation for validation and test data! So I defined two separate transforms. Also, I used splitfolders to manage data folders properly as suggested by my reviewer.

### (IMPLEMENTATION) Visualize a Batch of Training Data

Use the code cell below to retrieve a batch of images from your train data loader, display at least 5 images simultaneously, and label each displayed image with its class name (e.g., "Golden Gate Bridge").

Visualizing the output of your data loader is a great way to ensure that your data loading and preprocessing are working as expected.


```python
import matplotlib.pyplot as plt
%matplotlib inline

## TODO: visualize a batch of the train data loader

# obtain one batch of training images
classes = [classes_name.split(".")[1] for classes_name in train_data.classes]

# helper function to un-normalize and display an image
def imshow(img):
    img = (img - img.min())/ (img.max() - img.min()) # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image


dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy() # convert images to numpy for display

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(30, 10))
for idx in np.arange(10):
    ax = fig.add_subplot(2, 5, idx+1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title(classes[labels[idx]])
```


![png](/notebook_images/output_7_0.png)


### Initialize use_cuda variable


```python
# useful variable that tells us whether we should use the GPU
use_cuda = torch.cuda.is_available()
if not use_cuda:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
```

    CUDA is available!  Training on GPU ...


### (IMPLEMENTATION) Specify Loss Function and Optimizer

Use the next code cell to specify a [loss function](http://pytorch.org/docs/stable/nn.html#loss-functions) and [optimizer](http://pytorch.org/docs/stable/optim.html).  Save the chosen loss function as `criterion_scratch`, and fill in the function `get_optimizer_scratch` below.


```python
## TODO: select loss function
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


criterion_scratch = nn.CrossEntropyLoss()

def get_optimizer_scratch(model):
    ## TODO: select and return an optimizer
    return optim.SGD(model.parameters(), lr=0.01)
```

### (IMPLEMENTATION) Model Architecture

Create a CNN to classify images of landmarks.  Use the template in the code cell below.


```python
#import torch.nn as nn
import torch.nn.functional as F

# define the CNN architecture
class Net(nn.Module):
    ## TODO: choose an architecture, and complete the class
    def __init__(self):
        super(Net, self).__init__()

        ## Define layers of a CNN
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, len(classes))
        # dropout layer (p=0.25)
        self.dropout = nn.Dropout(0.25)


    def forward(self, x):
        ## Define forward behavior
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # flatten image input
        x = x.view(-1, 64 * 28 * 28)
        # add dropout layer
        x = self.dropout(x)
        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add 2nd hidden layer, with relu activation function
        x = F.relu(self.fc2(x))
        # add dropout layer
        x = self.dropout(x)
        # add 3rd hidden layer, with relu activation function
        x = self.fc3(x)

        return x

#-#-# Do NOT modify the code below this line. #-#-#

# instantiate the CNN
model_scratch = Net()
print(model_scratch)
# move tensors to GPU if CUDA is available
if use_cuda:
    model_scratch.cuda()
```

    Net(
      (conv1): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv2): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv3): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (fc1): Linear(in_features=50176, out_features=512, bias=True)
      (fc2): Linear(in_features=512, out_features=256, bias=True)
      (fc3): Linear(in_features=256, out_features=50, bias=True)
      (dropout): Dropout(p=0.25)
    )


__Question 2:__ Outline the steps you took to get to your final CNN architecture and your reasoning at each step.  

__Answer:__  
For this task we only need 20% accuracy, so I tried to design a CNN as simple as possible. based on my research, a simple CNN typically consists of two or three convolutional layers and two or three fully connected layers. I used three convolutional layers and three fully connected layers. For other hyperparameters listed below, I used typical values that we learnt in our classroom lectures:

(conv1): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) -> input channel size is 3 for a colour image

(conv2): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

(conv3): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

(pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

(fc1): Linear(in_features=50176, out_features=512, bias=True)

(fc2): Linear(in_features=512, out_features=256, bias=True)

(fc3): Linear(in_features=256, out_features=50, bias=True) -> output size is equal to number of classes

(dropout): Dropout(p=0.25) -> To prevent overfitting.

I used ReLU as the activation function.

### (IMPLEMENTATION) Implement the Training Algorithm

Implement your training algorithm in the code cell below.  [Save the final model parameters](http://pytorch.org/docs/master/notes/serialization.html) at the filepath stored in the variable `save_path`.


```python
def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf

    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        ###################
        # train the model #
        ###################
        # set the module to training mode
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()

            ## TODO: find the loss and update the model parameters accordingly
            ## record the average training loss, using something like
            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data.item() - train_loss))
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += ((1 / (batch_idx + 1)) * (loss.data.item() - train_loss))


        ######################    
        # validate the model #
        ######################
        # set the model to evaluation mode
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()

            ## TODO: update average validation loss
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # update average validation loss
            valid_loss += ((1 / (batch_idx + 1)) * (loss.data.item() - valid_loss))


        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch,
            train_loss,
            valid_loss
            ))

        ## TODO: if the validation loss has decreased, save the model at the filepath stored in save_path
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss


    return model
```

### (IMPLEMENTATION) Experiment with the Weight Initialization

Use the code cell below to define a custom weight initialization, and then train with your weight initialization for a few epochs. Make sure that neither the training loss nor validation loss is `nan`.

Later on, you will be able to see how this compares to training with PyTorch's default weight initialization.


```python
def custom_weight_init(m):
    ## TODO: implement a weight initialization strategy
    if type(m) == nn.Linear:
        # get the number of the input features
        n = m.in_features
        y = (1.0/np.sqrt(n))
        m.weight.data.normal_(0, y)
        m.bias.data.fill_(0)

#-#-# Do NOT modify the code below this line. #-#-#

model_scratch.apply(custom_weight_init)
model_scratch = train(5, loaders_scratch, model_scratch, get_optimizer_scratch(model_scratch),
                      criterion_scratch, use_cuda, 'ignore.pt')
```

    Epoch: 1 	Training Loss: 3.903600 	Validation Loss: 3.869816
    Validation loss decreased (inf --> 3.869816).  Saving model ...
    Epoch: 2 	Training Loss: 3.839096 	Validation Loss: 3.753673
    Validation loss decreased (3.869816 --> 3.753673).  Saving model ...
    Epoch: 3 	Training Loss: 3.758173 	Validation Loss: 3.661849
    Validation loss decreased (3.753673 --> 3.661849).  Saving model ...
    Epoch: 4 	Training Loss: 3.693037 	Validation Loss: 3.543283
    Validation loss decreased (3.661849 --> 3.543283).  Saving model ...
    Epoch: 5 	Training Loss: 3.617765 	Validation Loss: 3.458513
    Validation loss decreased (3.543283 --> 3.458513).  Saving model ...


### (IMPLEMENTATION) Train and Validate the Model

Run the next code cell to train your model.


```python
## TODO: you may change the number of epochs if you'd like,
## but changing it is not required
num_epochs = 30

#-#-# Do NOT modify the code below this line. #-#-#

# function to re-initialize a model with pytorch's default weight initialization
def default_weight_init(m):
    reset_parameters = getattr(m, 'reset_parameters', None)
    if callable(reset_parameters):
        m.reset_parameters()

# reset the model parameters
model_scratch.apply(default_weight_init)

# train the model
model_scratch = train(num_epochs, loaders_scratch, model_scratch, get_optimizer_scratch(model_scratch),
                      criterion_scratch, use_cuda, 'model_scratch.pt')
```

    Epoch: 1 	Training Loss: 3.912298 	Validation Loss: 3.909234
    Validation loss decreased (inf --> 3.909234).  Saving model ...
    Epoch: 2 	Training Loss: 3.907731 	Validation Loss: 3.900677
    Validation loss decreased (3.909234 --> 3.900677).  Saving model ...
    Epoch: 3 	Training Loss: 3.895026 	Validation Loss: 3.873590
    Validation loss decreased (3.900677 --> 3.873590).  Saving model ...
    Epoch: 4 	Training Loss: 3.849630 	Validation Loss: 3.771212
    Validation loss decreased (3.873590 --> 3.771212).  Saving model ...
    Epoch: 5 	Training Loss: 3.753362 	Validation Loss: 3.646837
    Validation loss decreased (3.771212 --> 3.646837).  Saving model ...
    Epoch: 6 	Training Loss: 3.654040 	Validation Loss: 3.515265
    Validation loss decreased (3.646837 --> 3.515265).  Saving model ...
    Epoch: 7 	Training Loss: 3.563133 	Validation Loss: 3.430825
    Validation loss decreased (3.515265 --> 3.430825).  Saving model ...
    Epoch: 8 	Training Loss: 3.500328 	Validation Loss: 3.367363
    Validation loss decreased (3.430825 --> 3.367363).  Saving model ...
    Epoch: 9 	Training Loss: 3.441512 	Validation Loss: 3.314716
    Validation loss decreased (3.367363 --> 3.314716).  Saving model ...
    Epoch: 10 	Training Loss: 3.408072 	Validation Loss: 3.266881
    Validation loss decreased (3.314716 --> 3.266881).  Saving model ...
    Epoch: 11 	Training Loss: 3.354051 	Validation Loss: 3.193987
    Validation loss decreased (3.266881 --> 3.193987).  Saving model ...
    Epoch: 12 	Training Loss: 3.299761 	Validation Loss: 3.158943
    Validation loss decreased (3.193987 --> 3.158943).  Saving model ...
    Epoch: 13 	Training Loss: 3.243109 	Validation Loss: 3.123086
    Validation loss decreased (3.158943 --> 3.123086).  Saving model ...
    Epoch: 14 	Training Loss: 3.211478 	Validation Loss: 3.088644
    Validation loss decreased (3.123086 --> 3.088644).  Saving model ...
    Epoch: 15 	Training Loss: 3.174485 	Validation Loss: 3.019170
    Validation loss decreased (3.088644 --> 3.019170).  Saving model ...
    Epoch: 16 	Training Loss: 3.134528 	Validation Loss: 2.928855
    Validation loss decreased (3.019170 --> 2.928855).  Saving model ...
    Epoch: 17 	Training Loss: 3.092019 	Validation Loss: 2.906347
    Validation loss decreased (2.928855 --> 2.906347).  Saving model ...
    Epoch: 18 	Training Loss: 3.043836 	Validation Loss: 2.874726
    Validation loss decreased (2.906347 --> 2.874726).  Saving model ...
    Epoch: 19 	Training Loss: 3.006519 	Validation Loss: 2.800367
    Validation loss decreased (2.874726 --> 2.800367).  Saving model ...
    Epoch: 20 	Training Loss: 2.975143 	Validation Loss: 2.773957
    Validation loss decreased (2.800367 --> 2.773957).  Saving model ...
    Epoch: 21 	Training Loss: 2.920179 	Validation Loss: 2.721721
    Validation loss decreased (2.773957 --> 2.721721).  Saving model ...
    Epoch: 22 	Training Loss: 2.861915 	Validation Loss: 2.652609
    Validation loss decreased (2.721721 --> 2.652609).  Saving model ...
    Epoch: 23 	Training Loss: 2.841097 	Validation Loss: 2.619521
    Validation loss decreased (2.652609 --> 2.619521).  Saving model ...
    Epoch: 24 	Training Loss: 2.780713 	Validation Loss: 2.520596
    Validation loss decreased (2.619521 --> 2.520596).  Saving model ...
    Epoch: 25 	Training Loss: 2.763787 	Validation Loss: 2.529581
    Epoch: 26 	Training Loss: 2.671952 	Validation Loss: 2.475181
    Validation loss decreased (2.520596 --> 2.475181).  Saving model ...
    Epoch: 27 	Training Loss: 2.617929 	Validation Loss: 2.331339
    Validation loss decreased (2.475181 --> 2.331339).  Saving model ...
    Epoch: 28 	Training Loss: 2.596152 	Validation Loss: 2.405016
    Epoch: 29 	Training Loss: 2.547864 	Validation Loss: 2.238860
    Validation loss decreased (2.331339 --> 2.238860).  Saving model ...
    Epoch: 30 	Training Loss: 2.479310 	Validation Loss: 2.181578
    Validation loss decreased (2.238860 --> 2.181578).  Saving model ...


### (IMPLEMENTATION) Test the Model

Run the code cell below to try out your model on the test dataset of landmark images. Run the code cell below to calculate and print the test loss and accuracy.  Ensure that your test accuracy is greater than 20%.


```python
def test(loaders, model, criterion, use_cuda):

    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    # set the module to evaluation mode
    model.eval()

    for batch_idx, (data, target) in enumerate(loaders['test']):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data.item() - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)

    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))

# load the model that got the best validation accuracy
model_scratch.load_state_dict(torch.load('model_scratch.pt'))
test(loaders_scratch, model_scratch, criterion_scratch, use_cuda)
```

    Test Loss: 2.963028


    Test Accuracy: 26% (332/1250)


---
<a id='step2'></a>
## Step 2: Create a CNN to Classify Landmarks (using Transfer Learning)

You will now use transfer learning to create a CNN that can identify landmarks from images.  Your CNN must attain at least 60% accuracy on the test set.

### (IMPLEMENTATION) Specify Data Loaders for the Landmark Dataset

Use the code cell below to create three separate [data loaders](http://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader): one for training data, one for validation data, and one for test data. Randomly split the images located at `landmark_images/train` to create the train and validation data loaders, and use the images located at `landmark_images/test` to create the test data loader.

All three of your data loaders should be accessible via a dictionary named `loaders_transfer`. Your train data loader should be at `loaders_transfer['train']`, your validation data loader should be at `loaders_transfer['valid']`, and your test data loader should be at `loaders_transfer['test']`.

If you like, **you are welcome to use the same data loaders from the previous step**, when you created a CNN from scratch.


```python
### TODO: Write data loaders for training, validation, and test sets
## Specify appropriate transforms, and batch_sizes

#loaders_transfer = {'train': None, 'valid': None, 'test': None}
loaders_transfer = loaders_scratch.copy()
```

### (IMPLEMENTATION) Specify Loss Function and Optimizer

Use the next code cell to specify a [loss function](http://pytorch.org/docs/stable/nn.html#loss-functions) and [optimizer](http://pytorch.org/docs/stable/optim.html).  Save the chosen loss function as `criterion_transfer`, and fill in the function `get_optimizer_transfer` below.


```python
## TODO: select loss function
criterion_transfer = nn.CrossEntropyLoss()


def get_optimizer_transfer(model):
    ## TODO: select and return optimizer
    return optim.SGD(model.classifier.parameters(), lr=0.01)
```

### (IMPLEMENTATION) Model Architecture

Use transfer learning to create a CNN to classify images of landmarks.  Use the code cell below, and save your initialized model as the variable `model_transfer`.


```python
## TODO: Specify model architecture
from torchvision import models

model_transfer = models.vgg16(pretrained=True)
# Freeze training for all "features" layers
for param in model_transfer.features.parameters():
    param.requires_grad = False

n_inputs = model_transfer.classifier[6].in_features
# add last linear layer (n_inputs -> 50 landmark classes)
# new layers automatically have requires_grad = True
last_layer = nn.Linear(n_inputs, len(classes))
model_transfer.classifier[6] = last_layer

#-#-# Do NOT modify the code below this line. #-#-#

if use_cuda:
    model_transfer = model_transfer.cuda()
```

    Downloading: "https://download.pytorch.org/models/vgg16-397923af.pth" to /root/.torch/models/vgg16-397923af.pth
    100%|██████████| 553433881/553433881 [00:04<00:00, 112269299.92it/s]


__Question 3:__ Outline the steps you took to get to your final CNN architecture and your reasoning at each step.  Describe why you think the architecture is suitable for the current problem.

__Answer:__  
I used VGG16 because it has already been trained with millions of images. Our dataset is quite small. This part is very similar to what we have done in our transfer learning exercise. I froze training for all features to ensure my net acts as a feature extractor. Then I Removed the last layer and replaced it with my linear classifier.

### (IMPLEMENTATION) Train and Validate the Model

Train and validate your model in the code cell below.  [Save the final model parameters](http://pytorch.org/docs/master/notes/serialization.html) at filepath `'model_transfer.pt'`.


```python
# TODO: train the model and save the best model parameters at filepath 'model_transfer.pt'
# number of epochs to train the model
train(20, loaders_transfer, model_transfer, get_optimizer_transfer(model_transfer), criterion_transfer,
      use_cuda, 'model_transfer.pt')


#-#-# Do NOT modify the code below this line. #-#-#

# load the model that got the best validation accuracy
model_transfer.load_state_dict(torch.load('model_transfer.pt'))
```

    Epoch: 1 	Training Loss: 2.530984 	Validation Loss: 1.438752
    Validation loss decreased (inf --> 1.438752).  Saving model ...
    Epoch: 2 	Training Loss: 1.486814 	Validation Loss: 1.064906
    Validation loss decreased (1.438752 --> 1.064906).  Saving model ...
    Epoch: 3 	Training Loss: 1.223969 	Validation Loss: 0.864935
    Validation loss decreased (1.064906 --> 0.864935).  Saving model ...
    Epoch: 4 	Training Loss: 1.061561 	Validation Loss: 0.763809
    Validation loss decreased (0.864935 --> 0.763809).  Saving model ...
    Epoch: 5 	Training Loss: 0.928663 	Validation Loss: 0.640140
    Validation loss decreased (0.763809 --> 0.640140).  Saving model ...
    Epoch: 6 	Training Loss: 0.817717 	Validation Loss: 0.572535
    Validation loss decreased (0.640140 --> 0.572535).  Saving model ...
    Epoch: 7 	Training Loss: 0.737212 	Validation Loss: 0.504320
    Validation loss decreased (0.572535 --> 0.504320).  Saving model ...
    Epoch: 8 	Training Loss: 0.676868 	Validation Loss: 0.443686
    Validation loss decreased (0.504320 --> 0.443686).  Saving model ...
    Epoch: 9 	Training Loss: 0.602057 	Validation Loss: 0.372280
    Validation loss decreased (0.443686 --> 0.372280).  Saving model ...
    Epoch: 10 	Training Loss: 0.547620 	Validation Loss: 0.319252
    Validation loss decreased (0.372280 --> 0.319252).  Saving model ...
    Epoch: 11 	Training Loss: 0.498956 	Validation Loss: 0.295694
    Validation loss decreased (0.319252 --> 0.295694).  Saving model ...
    Epoch: 12 	Training Loss: 0.458273 	Validation Loss: 0.247565
    Validation loss decreased (0.295694 --> 0.247565).  Saving model ...
    Epoch: 13 	Training Loss: 0.439739 	Validation Loss: 0.239168
    Validation loss decreased (0.247565 --> 0.239168).  Saving model ...
    Epoch: 14 	Training Loss: 0.388564 	Validation Loss: 0.207113
    Validation loss decreased (0.239168 --> 0.207113).  Saving model ...
    Epoch: 15 	Training Loss: 0.357456 	Validation Loss: 0.196189
    Validation loss decreased (0.207113 --> 0.196189).  Saving model ...
    Epoch: 16 	Training Loss: 0.337147 	Validation Loss: 0.162997
    Validation loss decreased (0.196189 --> 0.162997).  Saving model ...
    Epoch: 17 	Training Loss: 0.300652 	Validation Loss: 0.147558
    Validation loss decreased (0.162997 --> 0.147558).  Saving model ...
    Epoch: 18 	Training Loss: 0.272933 	Validation Loss: 0.133376
    Validation loss decreased (0.147558 --> 0.133376).  Saving model ...
    Epoch: 19 	Training Loss: 0.261205 	Validation Loss: 0.111410
    Validation loss decreased (0.133376 --> 0.111410).  Saving model ...
    Epoch: 20 	Training Loss: 0.249733 	Validation Loss: 0.105623
    Validation loss decreased (0.111410 --> 0.105623).  Saving model ...


### (IMPLEMENTATION) Test the Model

Try out your model on the test dataset of landmark images. Use the code cell below to calculate and print the test loss and accuracy.  Ensure that your test accuracy is greater than 60%.


```python
test(loaders_transfer, model_transfer, criterion_transfer, use_cuda)
```

    Test Loss: 0.928608


    Test Accuracy: 77% (968/1250)


---
<a id='step3'></a>
## Step 3: Write Your Landmark Prediction Algorithm

Great job creating your CNN models! Now that you have put in all the hard work of creating accurate classifiers, let's define some functions to make it easy for others to use your classifiers.

### (IMPLEMENTATION) Write Your Algorithm, Part 1

Implement the function `predict_landmarks`, which accepts a file path to an image and an integer k, and then predicts the **top k most likely landmarks**. You are **required** to use your transfer learned CNN from Step 2 to predict the landmarks.

An example of the expected behavior of `predict_landmarks`:
```
>>> predicted_landmarks = predict_landmarks('example_image.jpg', 3)
>>> print(predicted_landmarks)
['Golden Gate Bridge', 'Brooklyn Bridge', 'Sydney Harbour Bridge']
```


```python
import cv2
from PIL import Image

def predict_landmarks(img_path, k):
    ## TODO: return the names of the top k landmarks predicted by the transfer learned CNN
    images = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Transform images
    images = transform(images)
    images.unsqueeze_(0)

    # check for cuda
    if use_cuda:
        images = images.cuda()

    # set model to evaluate
    model_transfer.eval()
    output = model_transfer(images)
    # Predict the top k most likely landmarks
    _, top_index = output.topk(k)
    top_classes = [classes[class_id] for class_id in top_index[0].tolist()]
    return top_classes


# test on a sample image
predict_landmarks('images/test/09.Golden_Gate_Bridge/190f3bae17c32c37.jpg', 5)
```




    ['Golden_Gate_Bridge',
     'Forth_Bridge',
     'Brooklyn_Bridge',
     'Sydney_Harbour_Bridge',
     'Niagara_Falls']



### (IMPLEMENTATION) Write Your Algorithm, Part 2

In the code cell below, implement the function `suggest_locations`, which accepts a file path to an image as input, and then displays the image and the **top 3 most likely landmarks** as predicted by `predict_landmarks`.

Some sample output for `suggest_locations` is provided below, but feel free to design your own user experience!
![](images/sample_landmark_output.png)


```python
def suggest_locations(img_path):
    # get landmark predictions
    predicted_landmarks = predict_landmarks(img_path, 3)

    ## TODO: display image and display landmark predictions
    image = Image.open(img_path).convert('RGB')
    plt.imshow(image)
    plt.show()

    print('Is this picture of the')
    print('{}, {}, or {}?'.format(*predicted_landmarks))


# test on a sample image
suggest_locations('images/test/09.Golden_Gate_Bridge/190f3bae17c32c37.jpg')
```


![png](/notebook_images/output_39_0.png)


    Is this picture of the
    Golden_Gate_Bridge, Forth_Bridge, or Brooklyn_Bridge?


### (IMPLEMENTATION) Test Your Algorithm

Test your algorithm by running the `suggest_locations` function on at least four images on your computer. Feel free to use any images you like.

__Question 4:__ Is the output better than you expected :) ?  Or worse :( ?  Provide at least three possible points of improvement for your algorithm.

__Answer:__ (Three possible points for improvement)

It performed better than I expected! All predictions were correct! Interestingly although there was a watermark on Vienna city hall, the prediction was correct!

Potential improvements:
1. Increasing the number of training images.
2. Tuning hyperparameters such as learning rate, drop out probabilities epoch size, batch size and etc.
3. Changing network architecture for example adding more convolutional and hidden layers.


```python
## TODO: Execute the `suggest_locations` function on
## at least 4 images on your computer.
## Feel free to use as many code cells as needed.
suggest_locations('images/sydney_harbour_bridge.jpg')
suggest_locations('images/banff_National_Park.jpg')
suggest_locations('images/eiffel_tower.jpg')
suggest_locations('images/vienna_city_hall.jpg')
```


![png](/notebook_images/output_42_0.png)


    Is this picture of the
    Sydney_Harbour_Bridge, Sydney_Opera_House, or Forth_Bridge?



![png](/notebook_images/output_42_2.png)


    Is this picture of the
    Banff_National_Park, Gullfoss_Falls, or Matterhorn?



![png](/notebook_images/output_42_4.png)


    Is this picture of the
    Eiffel_Tower, Terminal_Tower, or Washington_Monument?



![png](/notebook_images/output_42_6.png)


    Is this picture of the
    Vienna_City_Hall, Terminal_Tower, or Whitby_Abbey?
