import torch
from cnn_cat_dog import CNN
from cnn_cat_dog import image_preprocess
import cv2
import torchvision
cnn3 = torch.load('net.pkl')
'''
img = cv2.imread('image/dog.0.jpg')
transforms2 = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
    ]
)
new_img = cv2.resize(img,(64,64))
new_img = cv2.cvtColor(new_img,cv2.COLOR_BGR2GRAY)
#new_img = 255 -new_img
#cv2.imshow('aa',new_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
'''
'''
img_cv_tensor = transforms2(new_img)
img_cv_tensor = torch.squeeze(img_cv_tensor,dim=0)
img_cv_tensor = torch.unsqueeze(img_cv_tensor,dim=0)
img_cv_tensor = torch.unsqueeze(img_cv_tensor,dim=0)
newout = cnn3(img_cv_tensor)
pred_y = torch.max(newout, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
'''
data2 = image_preprocess('image')
for t in range(len(data2)):
    print(t)
    t = data2[t][0]
    t = torch.tensor(t)
    t = torch.unsqueeze(t,dim=0)
    t = torch.unsqueeze(t,dim=0)
    t = t.float()
    print(t)
    output = cnn3(t)
    print(output)
    pred_y = torch.max(output, 1)[1].data.numpy().squeeze()
    print(pred_y, 'prediction number')  