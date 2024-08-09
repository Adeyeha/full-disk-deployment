import torch
import torch.nn as nn 
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from torchvision.transforms import ToTensor


# NET_TYPES = {'alexnet': torchvision.models.alexnet}
NET_TYPES = {
             'customalexnet': torchvision.models.alexnet,
             'customvgg16' : torchvision.models.vgg16,
             'customresnet34': torchvision.models.resnet34,
             'vgg16': None
             
}

class Custom_AlexNet(nn.Module):

    def __init__(self,
                 ipt_size=(512, 512), 
                 pretrained=True, 
                 net_type='customalexnet', 
                 num_classes=2):
        super(Custom_AlexNet, self).__init__()
        
        #add one convolution layer at the beginning
        self.first_conv_layer = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True),
            nn.ReLU(inplace=True)
            )

        # load convolutional part of AlexNet
        assert net_type in NET_TYPES, "Unknown vgg_type '{}'".format(net_type)
        net_loader = NET_TYPES[net_type]
        net = net_loader(pretrained=pretrained)
        self.features = net.features

        # init fully connected part of AlexNet
        test_ipt = Variable(torch.zeros(1,3,ipt_size[0],ipt_size[1]))
        test_out = net.features(test_ipt)
        self.n_features = test_out.size(1) * test_out.size(2) * test_out.size(3)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        self._init_classifier_weights()

    def forward(self, x):
        x = self.first_conv_layer(x)
        x = self.features(x)
        x = self.avgpool(x) 
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return F.softmax(x, dim=1)
        #return x

    def _init_classifier_weights(self):
        for m in self.first_conv_layer:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class Custom_VGG16(nn.Module):

    def __init__(self,
                 ipt_size=(512, 512), 
                 pretrained=True, 
                 net_type='customvgg16', 
                 num_classes=2, train=True):
        super(Custom_VGG16, self).__init__()


        #Mode Initialization
        self.tr = train
        #add one convolution layer at the beginning
        self.first_conv_layer = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True),
            nn.ReLU(inplace=True)
            )

        # load convolutional part of vgg
        assert net_type in NET_TYPES, "Unknown vgg_type '{}'".format(net_type)
        net_loader = NET_TYPES[net_type]
        net = net_loader(pretrained=pretrained)
        self.features = net.features

        # init fully connected part of vgg
        test_ipt = Variable(torch.zeros(1,3,ipt_size[0],ipt_size[1]))
        test_out = net.features(test_ipt)
        self.n_features = test_out.size(1) * test_out.size(2) * test_out.size(3)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

        self._init_classifier_weights()

    def forward(self, x):
        x = self.first_conv_layer(x)
        x = self.features(x)
        x = self.avgpool(x) 
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        if self.tr:
            return F.log_softmax(x) #Use this for training
        else:
            return x #Use this to get the probability scores, and apply softmax on the output scores

    def _init_classifier_weights(self):
        for m in self.first_conv_layer:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class VGG16(nn.Module):
    def __init__(self, num_classes=2):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=False),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=False),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class Custom_ResNet34(nn.Module):

    def __init__(self,
                 ipt_size=(512, 512), 
                 pretrained=True, 
                 net_type='customresnet34', 
                 num_classes=2, train=False):
        super(Custom_ResNet34, self).__init__()
        

        #Mode Initialization
        self.tr = train
        #add one convolution layer at the beginning
        self.first_conv_layer = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True),
            nn.ReLU(inplace=False)
            )

        # load convolutional part of vgg
        assert net_type in NET_TYPES, "Unknown vgg_type '{}'".format(net_type)
        net_loader = NET_TYPES[net_type]
        net = net_loader(pretrained=pretrained)
        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

        # init fully connected part of vgg
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Linear(in_features=512, out_features=num_classes))

        self._init_classifier_weights()

    def forward(self, x):
        x = self.first_conv_layer(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x) 
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if self.tr:
            return F.log_softmax(x) #Use this for training
        else:
            return x #Use this to get the probability scores, and apply softmax on the output scores
        
    def _init_classifier_weights(self):
        for m in self.first_conv_layer:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
        for m in self.fc:
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

