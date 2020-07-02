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
data_index = 0

if os.path.exists("DataFrame") == True:
    print("-- DataFrame directory exists")
else:
    os.mkdir("DataFrame")
    print("-- Made directory: DataFrame")

if os.path.exists("DataFrame\\dataFrame.csv"):
    dataFrame = pd.read_csv("DataFrame\\dataframe.csv")
    data_index = dataFrame.shape[0]
    print("- Loaded DataFrame")
else:
    dataFrame = pd.DataFrame(columns = ["epoch","batch_size", "lr", "train_acc", "train_loss", "valid_acc", "valid_loss"])
    print("- Built new DataFrame")
#######################

################################
### Convolutional Neural Network
################################

''' Simple Network to compare with Grid and Random'''

class Net(nn.Module):
    def __init__(self,name):
        super().__init__()
        self.name = name
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



####################################
################ Training parameters
####################################
save = False
epoch = 2
learning_rate = [0.01, 0.001, 0.002]  #[0.003, 0.002, 0.001, 0.0005, 0.0001]
batch_size = 20
search = 0        # 0: no search, 1: grid search, 2: random search
self_made_network = False
einheit = 2
###################################
###################################

##########################
### Deterministic Training
##########################
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


####################
### GPU & CUDA check
####################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)


###############
### Data-Loader
###############
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(test, batch_size=1, shuffle=False)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
print("Data loaded")


#################
### Training loop
#################
criterion = nn.CrossEntropyLoss()
sum_epoch = 0
for epochen in range(int(epoch / einheit)):
    """
    net = models.sqeezenet1_1(pretrained=False, progress=True)
                    
                                                                    resnet18 = models.resnet18()
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
    print("Model loaded")

    losses = []          # array für losses von den verschiedenen Netzen 
    net_number = 0       # Zähler zu trainierten Netze
    sum_epoch = 0
    nets = {}
    for lr in learning_rate:
        if epochen == 0:
            net = Net('Network').to(device)
            #net = models.squeezenet1_1(pretrained=False, progress=True).to(device)
        ###########################################################################
        elif epochen > 0:
            net = pickle.loads(pickle.dumps(best_net1))
        #############################################################################    
        net = net.to(device)
        #actual_learninrate = learning_rate + variable
        print("\033[91m" + "Learning Rate: {:.5}".format(lr), "\033[0m")
        dataFrame.at[data_index, "lr"] = lr
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0)
        

        for epoch in range(einheit):  # loop over the dataset multiple times
            start = time.time()
            running_loss = 0.0
            correct = 0
            train_acc = 0.0
            accuracy = 0
            total_loss = 0.0
            print("---------------------------------")
            print("Epoch: ", epoch + 1)
            dataFrame.at[data_index, "batch_size"] = batch_size
            dataFrame.at[data_index, "epoch"] = sum_epoch

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

                #outputs = outputs.cpu().detach()
                #labels = labels.cpu().detach()

                ps = torch.exp(outputs)
                equal = (labels.data == ps.max(dim=1)[1])
                accuracy += equal.type(torch.FloatTensor).mean()
                #print(accuracy)
            
            print("train_acc: {:.3f}".format(accuracy / len(trainloader) * 100), "%")
            print("train_loss: {:.3f}".format(total_loss / len(trainloader)))
            dataFrame.at[data_index, "train_acc"] = (accuracy / len(trainloader)).cpu().numpy()
            dataFrame.at[data_index, "train_loss"] = (total_loss / len(trainloader)).detach().cpu().numpy()

            end = time.time()
            print("Time pro epoch: ", end - start, "s")

            with torch.no_grad():
                val_loss = 0.
                total = 0
                correct = 0
                for i, data in enumerate(testloader, 0):
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
                print("val_accuracy: {:.3f}".format((correct / total) * 100) , "%")
                print("val_loss: {:.3f}".format(valid_loss))
                dataFrame.at[data_index, "valid_acc"] = correct / total
                dataFrame.at[data_index, "valid_loss"] = valid_loss.cpu().numpy()

                data_index += 1

        ### Dynamic number of Networks:
        ###############################
        nets["net" + str(net_number)] = pickle.loads(pickle.dumps(net))
        #print(nets)
        net_number += 1
        ###############################

        losses.append(valid_loss)       # auf CPU + float statt Tensor
        print("val_losses by networks:", losses)

    sum_epoch += 1

    print(losses)
    #############
    best_network = np.argmin(losses)
    print("Best network:", best_network)
    net = nets[list(nets)[best_network]]
    #print(net)
    #############

  ###############################################################
    best_net1 = pickle.loads(pickle.dumps(net))
  ###############################################################
    """ 
    PATH = "./Models/cifar_net" + str(epoch) + ".pth"
    torch.save(net.state_dict(), PATH)
    print("Model saved")
    """

print('Finished Training')
print(dataFrame)
dataFrame.to_csv("C:\\Users\\borda\\Documents\\KIT ETIT\\LAMA\\Projekt\\DataFrame\\dataFrame.csv")

# Model saver
if save == True:
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)