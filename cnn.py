import torch
from torch import nn
from torch.utils import data

import torchvision
import torchvision.transforms as transforms
import cv2
import json

import numpy as np

# need to pad videos so they are all the same length (in number of frames). Util function for that, and run it here (run it once then we don't need it again)
from frames import leastmostframes
# least, most = leastmostframes('subset2/data')
most = 76 # hard coded for speed. The function only really needs to be run whenever we add new videos to our dataset.

# directory to load data to train on
dataDirectory = "subset2"

# TODO: move to some type of preprocessing file
def processID(Id):
  global most
  video = cv2.VideoCapture(f'{dataDirectory}/data/{Id}.webm')
  greyScaleVideo = []

  newFrame, data = video.read()
  while newFrame:
    greyScaleVideo.append(data) # NOTE: no longer greyscale
    newFrame, data = video.read()
  
  # Appending additional frames of blackness 
  numBlackFrames = most - len(greyScaleVideo)
  greyScaleVideo = np.concatenate([greyScaleVideo, np.zeros((numBlackFrames, *greyScaleVideo[0].shape))])
  
  return greyScaleVideo


with open(f'{dataDirectory}/subset-train.json') as file:
  trainingSetInfo = json.load(file)

with open(f'{dataDirectory}/subset-validation.json') as file:
  validationSetInfo = json.load(file)

classes = []
for element in trainingSetInfo:
  if element['template'] not in classes:
    classes.append(element['template'])

import pdb
pdb.set_trace()

class somethingDataset(data.Dataset):
  def __init__(self, setInfo):
    self.setInfo = setInfo
  
  def __len__(self):
    return len(self.setInfo)

  def __getitem__(self, index):
    vid = self.setInfo[index]
    video = processID(vid['id'])
    label = vid['template']
    index = classes.index(label)
    label = np.zeros((len(classes)))
    label[index] = 1
    return video, label


trainDataLoader = data.DataLoader(somethingDataset(trainingSetInfo), batch_size=1, shuffle=True, num_workers=0)
validationDataLoader = data.DataLoader(somethingDataset(validationSetInfo), batch_size=64, shuffle=True)

class CNN(nn.Module):
  def __init__(self):
    super().__init__()
    #                      greyscale (3 for rgb)   num filters
    self.conv1 = nn.Conv3d(in_channels=76,          out_channels=5, kernel_size=(5,5,3))
    self.relu1 = nn.ReLU()
    self.maxpool1 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))

    self.fc1 = nn.Linear(in_features=800, out_features=500)
    self.relu3 = nn.ReLU()

    #                                  have 3 classes
    self.fc2 = nn.Linear(in_features=500, out_features=3)
    self.logSoftmax = nn.LogSoftmax(dim=1)


  def forward(self, x):
    x = self.conv1(x)
    x = self.relu1(x)
    x = self.maxpool1(x)

    x = torch.flatten(x,1)
    # PROBABLE ERROR: NOT SURE WHAT SIZE THE in_features OF fc1 SHOULD BE. RN IT IS 800 BASED OFF TUTORIAL, BUT PROBABLY NEED TO BE CHANGED.
    x = self.fc1(x)
    x = self.relu3(x)

    x = self.fc2(x)
    output = self.logSoftmax(x)

    return output


device = 'cpu' # or 'cuda' in future
model = CNN().to(device)

model = model.float()

EPOCHS = 1
LEARNING_RATE = 0.001
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_loss_list = []

for epoch in range(EPOCHS):
  train_loss = 0

  model.train(True)
  for (x,y) in trainDataLoader: # x variable represents input data and y represents corresponding output 
    x = x.to(device)  # Moves input and output to specified device
    # import pdb
    # pdb.set_trace()
    optimizer.zero_grad()              # Sets the models gradients to zero for each iteration so previous doesnt affect current
    y_pred = model(x.float())                  # COmputes models predicted output for input data
    loss = loss_fn(y_pred, y)          # Computes loss between predicted output and true output
    loss.backward()                    # cOMPUTES GRADIENT
    optimizer.step()                   # Updates models parameters





