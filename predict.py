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


parser = argparse.ArgumentParser()
parser.add_argument( '-i','--image_path', type=str, help = 'insert image path')
parser.add_argument('-k','--top_k', type = int, default = 5, help= 'number of classes the flower could belong to')
parser.add_argument('-g','--gpu',type= str, choices=['True', 'False'],help='Use GPU if available')
parser.add_argument('-c', '--category_names', type=str, help='Use a mapping of categories to real names from a json file')
args = parser.parse_args()


def load_checkpoint(filepath):
    checkpoint=torch.load(filepath)
    
    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
        in_features = 25088
        for param in model.parameters():
            param.requires_grad = False
    elif checkpoint['arch'] == 'vgg11':
        model = models.vgg11(pretrained=True)
        in_features = 25088
        for param in model.parameters():
            param.requires_grad = False
    elif checkpoint['arch'] == 'vgg13':
        model = models.vgg13(pretrained=True)
        in_features = 25088
        for param in model.parameters():
            param.requires_grad = False
    else:
        print('Sorry base architecture not recognised')
    #Freeze parameters
    
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['index']
    model.state_dict = checkpoint['state_dict']
    model.optimizer = checkpoint['optimizer']

    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    #resize
    size = 256,256
    im=Image.open(image)
    im.thumbnail(size)
    
    #cropout the center
    
    x =224
    width, height = im.size
    left= (width - x)/2
    top= (height - x)/2
    right= (width + x)/2
    bottom= (height +x)/2
    im=im.crop((left,top,right,bottom))
    
    # Convert color channels to float
    
    np_image = np.array(im)/255
    mean_normalize = np.array([0.485, 0.456, 0.406])
    std_normalize = np.array([0.229, 0.224, 0.225])
    np_image = (np_image- mean_normalize)/std_normalize
    np_image = np_image.transpose((2,0,1))
    toFloatTensor  = torch.from_numpy(np_image ).type(torch.FloatTensor)
    return toFloatTensor

import json
if args.category_names is  None:
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
else:  
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)


def predict(args):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to('cpu')
    
    # Set model to evaluate
    model.eval()
   
    torch_image = torch.from_numpy(np.expand_dims(process_image(args.image_path), 
                                                  axis=0)).type(torch.FloatTensor).to('cpu')

  
    log_probability = model.forward(torch_image)


    linear_probability = torch.exp(log_probability)

  
    top_probability, top_labels = linear_probability.topk(args.top_k)
    
    
    top_probability = np.array(top_probability.detach())[0] 
    top_labels = np.array(top_labels.detach())[0]
    
    
    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labels]
    top_flowers = [cat_to_name[lab] for lab in top_labels]
    
    return top_probability, top_labels, top_flowers



model = load_checkpoint('checkpoint.pth')
if args.gpu == 'True':
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
else:
    "cpu"
probability, labels, flowers = predict(args)
print(probability, labels, flowers)

