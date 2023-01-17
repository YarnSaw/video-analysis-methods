import torch
from torch import nn
from torch.utils import data

import torchvision
import torchvision.transforms as transforms
import cv2
import json

# filling in empty frames util
from frames import leastmostframes
# least, most = leastmostframes('subset2/data')
most = 76 # hard coded for speed
# emptyFrame = [[[0] for i in range(427)] for i in range(240)]

# TODO: move to some type of preprocessing file
def processID(Id):
  global most
  video = cv2.VideoCapture(f'subset2/{Id}.webm')
  
  greyScaleVideo = []

  newFrame, data = video.read()
  while newFrame:
    greyScaleVideo.append(cv2.cvtColor(data, cv2.COLOR_BGR2GRAY))
    newFrame, data = video.read()

  # Append black frames until we reach highest frame count
  for i in range(most - len(greyScaleVideo)):
    greyScaleVideo.append([[[0] for i in range(427)] for i in range(240)])

  # print(len(greyScaleVideo)*len(greyScaleVideo[0])*len(greyScaleVideo[0][0])*len(greyScaleVideo[0][0][0]))
  return greyScaleVideo


with open('subset2/subset-train.json') as file:
  trainingSetInfo = json.load(file)

with open('subset2/subset-validation.json') as file:
  validationSetInfo = json.load(file)


class somethingDataset(data.Dataset):
  def __init__(self, setInfo):
    self.setInfo = setInfo
  
  def __len__(self):
    return len(self.setInfo)

  def __getitem__(self, index):
    vid = self.setInfo[index]
    video = processID(vid['id'])
    label = vid['template']
    return video, label


trainDataLoader = data.DataLoader(somethingDataset(trainingSetInfo), batch_size=64, shuffle=True, num_workers=0)
validationDataLoader = data.DataLoader(somethingDataset(validationSetInfo), batch_size=64, shuffle=True)

class CNN(nn.Module):
  def __init__(self):
    super().__init__()
    #                      greyscale (3 for rgb)   num filters
    self.conv1 = nn.Conv3d(in_channels=1,          out_channels=5, kernel_size=(5,5,5))
    self.relu1 = nn.ReLU()
    self.maxpool1 = nn.MaxPool3d(kernel_size=(2,2), stride=(2,2))

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

EPOCHS = 1
LEARNING_RATE = 0.001
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_loss_list = []

for epoch in range(EPOCHS):
  train_loss = 0
  model.train()
  for (x,y) in trainDataLoader:
    import pdb
    pdb.set_trace()




