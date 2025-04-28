import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import cv2
import numpy as np
from colormap.colors import Color, hex2rgb
import png
from torchsummary import summary

# copied Results from the terminal
trainloss = [0.942, 0.940, 1.048, 0.612,
             0.744, 0.805, 0.492]
valloss = [0.9824785716003842, 0.9705788943502638, 0.8404008236196306, 0.8498707761367162,
           0.8027641296386718, 0.8532716658380296, 0.7603517012463675]
plt.plot(range(1, len(trainloss)+1), trainloss, label='Train', color='blue')
plt.plot(range(1, len(valloss)+1), valloss, label='Validation', color='orange')
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("trainloss_valloss")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.n_class = 5

        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),  
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Conv2d(64, self.n_class, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        x = self.pool1(enc1)

        enc2 = self.enc2(x)
        x = self.pool2(enc2)

        enc3 = self.enc3(x)
        x = self.pool3(enc3)

        x = self.bottleneck(x)

        x = self.upconv3(x)
        x = torch.cat((x, enc3), dim=1)  
        x = self.dec3(x)

        x = self.upconv2(x)
        x = torch.cat((x, enc2), dim=1)  
        x = self.dec2(x)

        x = self.upconv1(x)
        x = torch.cat((x, enc1), dim=1)  
        x = self.dec1(x)

        x = self.final_conv(x)
        
        return x

def save_label(label, path):
    '''
    Function for ploting labels.
    '''
    colormap = [
        '#000000',
        '#0080FF',
        '#80FF80',
        '#FF8000',
        '#FF0000',
    ]
    assert(np.max(label)<len(colormap))
    colors = [hex2rgb(color, normalise=False) for color in colormap]
    w = png.Writer(label.shape[1], label.shape[0], palette=colors, bitdepth=4)
    with open(path, 'wb') as f:
        w.write(f, label)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

img = cv2.imread('input.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype(np.float32) / 255.0
img_tensor = torch.tensor(img).permute(2, 0, 1)
img_tensor = img_tensor.unsqueeze(0)
img_tensor = img_tensor.to(device)

net = Net().to(device)
net.load_state_dict(torch.load('./models/model_starter_net.pth'))
net.eval()
summary(net, input_size=(3, 256, 256))

output = net(img_tensor)[0].cpu().detach().numpy()

c, h, w = output.shape
assert(c == 5)
y = np.zeros((h,w)).astype('uint8')
for i in range(5):
    mask = output[i]>0.5
    y[mask] = i
save_label(y, 'output.png')