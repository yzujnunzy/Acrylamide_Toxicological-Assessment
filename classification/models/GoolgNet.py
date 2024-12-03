# 引入了Inception结构（融合不同尺度的特征信息）
'''
使用1*1的卷积核进行降维以及映射处理
添加两个辅助分类器帮助训练
丢弃全连接层，使用平均池化层（大大减少模型参数）
且GoogleNet有3个输出层，其中两个是辅助分类层
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


# 13.31
class GoogleNet(nn.Module):
    def __init__(self,num_classes=1000,aux_logits=True,init_weight=False):
        super(GoogleNet,self).__init__()
        self.aux_logits=aux_logits

        self.conv1=BasicConv2d(3,64,kernel_size=7,stride=2,padding=3)
        self.maxpool1=nn.MaxPool2d(3,stride=2,ceil_mode=True)#ceil_mode=True:如果是小数，则向上取整，否则向下取整

        self.conv2=BasicConv2d(64,64,kernel_size=1)
        self.conv3=BasicConv2d(64,192,kernel_size=3,padding=1)
        self.maxpool2=nn.MaxPool2d(3,stride=2,ceil_mode=True)

        self.inception3a=Inception(192,64,96,128,16,32,32)
        self.inception3b=Inception(256,128,128,192,32,96,64)
        self.maxpool3=nn.MaxPool2d(3,stride=2,ceil_mode=True)

        self.inception4a=Inception(480,192,96,208,16,48,64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4=nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.inception5a=Inception(832,256,160,320,32,128,128)
        self.inception5b=Inception(832,384,192,384,48,128,128)
        #如果使用辅助分类器
        if aux_logits:
            self.aux1=InceptionAux(512,num_classes)
            self.aux2=InceptionAux(528,num_classes)

        #平均池化层
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))#自适应平均池化层，不管输入多大图像，得到的都是高为1，宽为1的矩阵
        self.dropout=nn.Dropout(0.4)
        self.fc=nn.Linear(1024,num_classes)
        if init_weight:
            self._initialize_weights()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    def forward(self,x):
        #Nx3x224x224
        x=self.conv1(x)
        #Nx64x112x112
        x=self.maxpool1(x)
        #Nx64x56x56
        x=self.conv2(x)
        #Nx64x56x56
        x=self.conv3(x)
        #Nx192x56x56
        x=self.maxpool2(x)

        #Nx192x28x28
        x=self.inception3a(x)
        #Nx256x28x28
        x=self.inception3b(x)
        #Nx480x28x28
        x=self.maxpool3(x)
        #Nx480x14x14
        x=self.inception4a(x)
        #Nx512x14x14

        #是否使用辅助分类器
        if self.training and self.aux_logits:
            aux1=self.aux1(x)

        x=self.inception4b(x)
        x=self.inception4c(x)
        x=self.inception4d(x)
        if self.training and self.aux_logits:
            aux2=self.aux2(x)

        x=self.inception4e(x)
        x=self.maxpool4(x)
        x=self.inception5a(x)
        x=self.inception5b(x)

        x=self.avgpool(x)
        x=torch.flatten(x,1)
        x=self.dropout(x)
        x=self.fc(x)

        if self.training and self.aux_logits:
            return x,aux1,aux2
        return x





class Inception(nn.Module):
    def __int__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()
        # 这几个分支必须保证输出大小相等
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)  # 保证输出大等于输入大小
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        # cat作用是对torch及进行链接，大小必须相同
        return torch.cat(outputs, 1)  # 通道排列顺序：batch,channels,height,width dim=1 行


# 辅助分类器
class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Inception, self).__init__()
        self.averagePool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)  # output[batch,128,4,4]

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # aux1:Nx512x14x14 aux2:Nx525x14x14
        x = self.averagePool(x)
        # aux1:Nx512x4x4 aux2:Nx525x4x4
        x = self.conv(x)
        # Nx128x4x4
        x = torch.flatten(x, 1)
        #在model.train()模式下，model.training=True,在model.eval()模式下，self.training=False
        x=F.dropout(x,0.5,training=self.training)
        #Nx2048
        x=F.relu(self.fc1(x),inplace=True)
        x=F.dropout(x,0.5,training=self.training)
        #Nx1024
        x=self.fc2(x)
        #Nxnum_classes
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x