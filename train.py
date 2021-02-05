
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import seaborn as sns
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--pretrained",
                    choices=["vgg11", "vgg13",
                             "vgg16"],
                    default="vgg16", help="Pre-trained model type")


parser.add_argument("-l", "--learning_rate", type=float, default=0.001, help="Learning rate")
parser.add_argument("-u", "--hidden_units", type=int,
                    help="Number of nodes per hidden layer")
parser.add_argument("-e", "--epochs", type=int, default=1, help="Number of training epochs")
parser.add_argument('-g','--gpu',type= str, choices=['True', 'False'],help='Use GPU if available')
args=parser.parse_args()



data_dir= 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Transform and load data

trainset = transforms.Compose([transforms.Resize(224),
                               transforms.RandomCrop(224),
                               transforms.RandomVerticalFlip(),
                               transforms.ToTensor(),
                               transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])

testset = transforms.Compose([transforms.Resize(224),
                              transforms.CenterCrop(224),
                              transforms.ToTensor(),
                              transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
                             ])

validset = transforms.Compose([transforms.Resize(224),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
                              ])

# TODO: Load the datasets with ImageFolder
train_dataset = datasets.ImageFolder(train_dir, transform=trainset)
test_dataset = datasets.ImageFolder(test_dir, transform=testset)
valid_dataset = datasets.ImageFolder(valid_dir, transform=validset)

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=True)


with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

#Allow user to choose to train on GPU or cpu
if args.gpu == 'True':
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
else:
    "cpu"




# Allow user to choose pretraining model
arch = args.pretrained
if arch == 'vgg16':
    model = models.vgg16(pretrained=True)
    in_features = 25088
    for param in model.parameters():
        param.requires_grad = False
elif arch == 'vgg11':
    model = models.vgg11(pretrained=True)
    in_features = 25088
    for param in model.parameters():
        param.requires_grad = False
elif arch == 'vgg13':
    model = models.vgg13(pretrained=True)
    in_features = 25088
    for param in model.parameters():
        param.requires_grad = False
else:
    print('Sorry base architecture not recognised')

#Define new classifier

classifier = nn.Sequential(nn.Linear(25088, 4096),
                      nn.ReLU(),
                      nn.Dropout(p =0.2),
                      nn.Linear(4096,1500),
                      nn.ReLU(),
                      nn.Dropout(p =0.2),
                      nn.Linear(1500, 512),
                      nn.LogSoftmax(dim=1)
                      )
model.classifier = classifier



#define loss function and optimizer
lr=args.learning_rate
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr)
model.to(device);




def training_model():
    epochs=args.epochs
    step=0
    running_loss=0
    print_every=60 

    for epoch in range(epochs):
        for images, labels in trainloader:
            step += 1

            images,labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            logps=model(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss +=1

            if step % print_every ==0:
                model.eval()
                test_loss=0
                accuracy=0
                with torch.no_grad():
                    for images,labels in validloader:

                        images,labels = images.to(device), labels.to(device)

                        logps= model(images)
                        loss = criterion(logps,labels)
                        test_loss += loss.item()
          
                        #calculate accuracy

                        ps=torch.exp(logps)
                        top_ps, top_class = ps.topk(1, dim=1)
                        equality = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equality.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(validloader):.3f}.. "
                      f"Test accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
training_model()               
            
# validation on the test set
def test_acc(model, testloader):
    test_accuracy = 0
    test_loss = 0
    for images,labels in testloader:
        model.eval()

        images,labels = images.to(device), labels.to(device)

        logps= model(images)
        loss = criterion(logps,labels)
        test_loss += loss.item()

        #calculate accuracy

        ps=torch.exp(logps)
        top_ps, top_class = ps.topk(1, dim=1)
        equality = top_class == labels.view(*top_class.shape)
        test_accuracy += torch.mean(equality.type(torch.FloatTensor)).item()

    print(f"Test accuracy: {test_accuracy/len(testloader):.3f}")
    
test_acc(model, testloader)

# Save the checkpoint 
def save_checkpoint(model):
    model.class_to_idx = train_dataset.class_to_idx

    checkpoint = {'arch': arch,
                'classifier': model.classifier,
                  'state_dict': model.state_dict(),
                  'index': model.class_to_idx,
                  'learning_rate': lr,
                  'optimizer': optimizer.state_dict}
    
    torch.save(checkpoint, 'checkpoint.pth')

save_checkpoint(model)   



