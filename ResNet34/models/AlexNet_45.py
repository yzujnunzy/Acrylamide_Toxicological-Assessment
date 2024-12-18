import torch.nn as nn
import torch
#2012year
class AlexNet(nn.Module):
    def __init__(self,num_class=1000,init_weights=False):
        super(AlexNet,self).__init__()
        self.features=nn.Sequential(
            #5 convolutional layers were used
            nn.Conv2d(3,48,kernel_size=5,stride=2,padding=1),#input[3,45,45] output[48,22,22]
            nn.ReLU(inplace=True),#It will change the value of the input data, saving the space and time of repeatedly requesting and releasing memory, and just passing the original address, which is more efficient.
            nn.MaxPool2d(kernel_size=3,stride=2),   #input[48,10,10]

            nn.Conv2d(48,128,kernel_size=5,padding=2),  #output[128,10,10]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=1),#output[128,8,8]

            nn.Conv2d(128,192,kernel_size=3,padding=1),#output[192,8,8]
            nn.ReLU(inplace=True),

            nn.Conv2d(192,192,kernel_size=3,padding=1),#output[192,8,8]
            nn.ReLU(inplace=True),

            nn.Conv2d(192,128,kernel_size=3,padding=1),#output[128,8,8]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),  #output[128,3,3]

        )

        self.classifier=nn.Sequential(
            nn.Dropout(p=0.5),#Each neuron has a 0.5 probability of not activating
            nn.Linear(128*3*3,2048),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(2048,2048),
            nn.ReLU(inplace=True),

            nn.Linear(2048,num_class)
        )

        if init_weights:
            self._initialize_weights()
    def forward(self,x):
        x=self.features(x)
        x=torch.flatten(x,start_dim=1)
        x=self.classifier(x)
        return x
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal(m.weight,mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,0,0.01)
                nn.init.constant_(m.bias,0)