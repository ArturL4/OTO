# Original code made by Artur & David @ LAMA 
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import torch.nn.functional as F
from termcolor import cprint
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import numpy as np
import torchvision
import pickle
import torch
import time
import copy
import os



####################
### Pandas DataFrame
####################

def createDataframe(name):

    if os.path.exists("DataFrame") == True:
        print("-- DataFrame directory exists")
    else:
        os.mkdir("DataFrame")
        print("-- Made directory: DataFrame")

    if os.path.exists("DataFrame\\" + name + ".csv"):
        dataFrame = pd.read_csv("DataFrame\\" + name + '.csv')
        data_index = dataFrame.shape[0]
        print("- Loaded DataFrame")
    else:
        dataFrame = pd.DataFrame(columns = ["epoch","batch_size", "lr", "train_acc", "train_loss", "valid_acc", "valid_loss"])
        print("- Built new DataFrame")
    cprint('---------------------------------','green')
    return dataFrame
#######################



################ Training parameters
####################################
save = False
epoch = 20
learning_rate_grid = [0.003,0.002,0.001,0.0005,0.0001]
learning_rate_random = np.random.uniform(0.0001,0.003,5)
batch_size = 20
search = 1        # 0: no search, 1: grid search, 2: random search
self_made_network = False
criterion = nn.CrossEntropyLoss()

############# Simple CNN Arcitecture
####################################

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 5)


        self.fc1 = nn.Linear(64 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #print(x.shape)
        #x = x.view(x.size(0), -1)
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    
##########Deterministic measurement
###################################
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

################################## Network creation  ####################################

def create_network(self_made_network = False):
  if self_made_network ==  True:
      net = models.mobilenet_v2(pretrained=False, progress=True)
      """                                                         resnet18 = models.resnet18()
                                                                  alexnet = models.alexnet()
                                                                  vgg16 = models.vgg16()
                                                                  squeezenet = models.squeezenet1_0()
                                                                  densenet = models.densenet161()
                                                                  inception = models.inception_v3()
                                                                  googlenet = models.googlenet()
                                                                  shufflenet = models.shufflenet_v2_x1_0()
                                                                  mobilenet = models.mobilenet_v2()
                                                                  resnext50_32x4d = models.resnext50_32x4d()
                                                                  wide_resnet50_2 = models.wide_resnet50_2()
                                                                  mnasnet = models.mnasnet1_0()
      """

  else:
      net = Net()


  
  net = net.to(device)
  print("Model loaded")
  return net

##############################################################################################
# import  preprocessed Data
def import_data():

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

    test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(test, batch_size=1, shuffle=False)

    return train, trainloader,test,testloader

###############################################################################################
#train the network

def trains(network,dataFrame,data_index, learning_rate = 0.001, epoch = 2, criterion = nn.CrossEntropyLoss()):
    cprint("Start training...", 'blue')
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0)

    cprint("Current learning Rate:{} ".format(learning_rate), 'red')

    for epoch in range(epoch):  # loop over the dataset multiple times
        start = time.time()
        running_loss = 0.0
        correct = 0
        total = 0
        train_acc = 0.0
        accuracy = 0
        total_loss = 0.0

        dataFrame.at[data_index, "lr"] = learning_rate
        dataFrame.at[data_index, "batch_size"] = batch_size
        dataFrame.at[data_index, "epoch"] = epoch + 1


        print("Epoch: ", epoch + 1)

        for i, data in enumerate(trainloader, 0):
          # get the inputs; data is a list of [inputs, labels]
          inputs, labels = data
          inputs = inputs.to(device)
          labels = labels.to(device)

          # zero the parameter gradients
          optimizer.zero_grad()

          # forward + backward + optimize
          outputs = net(inputs)
          loss = criterion(outputs, labels)

          loss.backward()
          optimizer.step()

          total_loss += loss

          ps = torch.exp(outputs)
          equal = (labels.data == ps.max(dim=1)[1])
          accuracy += equal.type(torch.FloatTensor).mean()
          pos = i + ((epoch)*len(trainloader))

    
        train_loss = total_loss / len(trainloader)
         
        print("train_acc: {:.3f}".format(accuracy/len(trainloader) * 100), "%")
        print("train_loss: {:.3f}".format(train_loss))


        dataFrame.at[data_index, "train_acc"] = (accuracy / len(trainloader)).cpu().numpy()*100
        dataFrame.at[data_index, "train_loss"] = (total_loss / len(trainloader)).detach().cpu().numpy()


        end = time.time()
        print("Time pro epoch: ", end - start, "s")

        with torch.no_grad():
            val_loss = 0.
            total = 0
            for data in testloader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss = criterion(outputs, labels)
                val_loss += loss
            valid_loss = val_loss / len(testloader)
            val_accuracy = (correct / total) * 100
            print("val_accuracy: {:.3f}".format((correct / total) * 100) , "%")
            print("val_loss: {:.3f}".format(valid_loss))


            dataFrame.at[data_index, "valid_acc"] = val_accuracy
            dataFrame.at[data_index, "valid_loss"] = valid_loss.cpu().numpy()




            
            
            data_index +=1


    print('Finished Training')
    print('------------------------------------------------------------------')
    return net, train_acc, train_loss, val_accuracy, valid_loss, dataFrame,data_index
#########################################################################################################




################# Configure calculating device and import train/test dataset #################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

#transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train, trainloader, test,testloader = import_data()

print("Data loaded")
print("------------------------------------------------------------------")
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#################################################################################################




normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

print("Data Logger started")


if search == 1:

    name = 'dataFrame_grid'
    df_grid = createDataframe(name)
    data_index = 0


    for learning_rate in learning_rate_grid:
        net = create_network(self_made_network)
        net,train_acc,train_loss, val_accuracy, val_loss, df_grid,data_index = trains(network = net,dataFrame = df_grid,data_index = data_index ,epoch = epoch,learning_rate = learning_rate,criterion = criterion)
        data_index +=1



    df_grid.to_csv("DataFrame\\dataFrame_grid.csv")
        



elif search == 2:
    name = 'dataFrame_random'
    df_random = createDataframe(name)
    data_index = 0
    for learning_rate in learning_rate_random:

        net = create_network(self_made_network)
        net,train_acc,train_loss, val_accuracy, val_loss, df_grid,data_index = trains(network = net,dataFrame = df_random,data_index = data_index ,epoch = epoch,learning_rate = learning_rate,criterion = criterion)
        data_index +=1




    df_random.to_csv("DataFrame\\dataFrame_random.csv")



cprint("#############################", 'green')
cprint("All Models have been trained!", 'green')
cprint("#############################", 'green')


print('Finished Training')

print(df_grid)
