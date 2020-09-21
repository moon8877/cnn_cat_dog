import cv2
import os 
import numpy as np
import torch
from torchvision import transforms
import random
def image_label(image):
    label = image.split('.')[0]
    if(label=='cat'):
        return torch.tensor(0)
    elif(label=='dog'):
        return torch.tensor(1)

def image_preprocess(dir_path):
    data = []
    for dir_path,dir_name,file_name in os.walk(dir_path):
        time = 1
        for f in file_name:
            img_path = os.path.join(dir_path,f)
            img_data = cv2.imread(img_path)
            img_data = cv2.resize(img_data,(64,64))
            img_data = cv2.cvtColor(img_data,cv2.COLOR_BGR2GRAY)
            img_data = img_data/255
            data.append([img_data,image_label(f)])
            print(time)
            time = time +1 
    return data
def image_preprocess2(dir_path):
    data = []
    for dir_path,dir_name,file_name in os.walk(dir_path):
        time = 1
        for f in file_name:
            img_path = os.path.join(dir_path,f)
            img_data = cv2.imread(img_path)
            img_data = cv2.resize(img_data,(64,64))
            img_data = cv2.cvtColor(img_data,cv2.COLOR_BGR2GRAY)
            img_data = img_data/255
            data.append(img_data)
            print(time)
            time = time +1 
    return data

class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )#output 16*32*32
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,
            ),#32*32*32
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.out = torch.nn.Linear(32*16*16,2)
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        print(x.shape)
        x = x.view(1,32*16*16)
        output = self.out(x)
        return output


'''
cnn = CNN()
optimizer = torch.optim.Adam(cnn.parameters(),lr=0.001)
loss_func = torch.nn.CrossEntropyLoss()
data = image_preprocess('train')
random.shuffle(data)

data2 = image_preprocess2('image')
print(np.shape(data2[0][0]))

a = 0
for t in range(len(data)):
    t = torch.tensor(data[t][0])
    t = torch.unsqueeze(t,dim=0)
    t = torch.unsqueeze(t,dim=0)
    t = t.float()
    u = torch.tensor(data[a][1])
    print(t)
    print(u)
    u = torch.unsqueeze(u,dim=0)
    output = cnn(t)
    print(output)
    #output = torch.squeeze(output)
    loss = loss_func(output,u)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(a)
    a = a+1

torch.save(cnn,'net.pkl')
torch.save(cnn.state_dict(),'net_parameters.pkl')


data2 = image_preprocess('image')

for t in range(len(data2)):
    t = torch.tensor(data2[t][0])
    t = torch.unsqueeze(t,dim=0)
    t = torch.unsqueeze(t,dim=0)
    t = t.float()
    
    output = cnn(t)
    pred_y = torch.max(output, 1)[1].data.numpy().squeeze()
    print(pred_y, 'prediction number')  
'''
