import torch
import torch.nn as nn

batch_size = 10
class_number = 14

class WideBottleneck(nn.Module):
    def __init__(self, input_dim, output_dim, stride = 1, downsample = False):
        super(WideBottleneck, self).__init__()

        hidden_dim = output_dim * 2

        self.conv1 = nn.Conv2d(
            input_dim, hidden_dim,
            kernel_size = 1, stride = 1, bias = False
        )
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(
            hidden_dim, hidden_dim,
            kernel_size = 3, stride = stride,
            padding = 1, bias = False
        )
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.conv3 = nn.Conv2d(
            hidden_dim, output_dim * 4,
            kernel_size = 1, stride = 1, bias = False
        )
        self.bn3 = nn.BatchNorm2d(output_dim * 4)
        self.relu = nn.ReLU(inplace = True)

        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    input_dim, output_dim * 4,
                    kernel_size = 1, stride = stride, bias = False
                ),
                nn.BatchNorm2d(output_dim * 4)
            )
        else:
            self.downsample = None

    def forward(self, x):
        shortcut = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)

        x += shortcut
        x = self.relu(x)

        return x
    
class Classifier(nn.Module):
    def __init__(self, RGB = 3, input_dim = 16, stride = 2):
        super(Classifier, self).__init__()
        self.default_CNN = nn.Sequential(
            nn.Conv2d(RGB * 4, input_dim, kernel_size = 4, stride = 1, padding = 1, bias = False),
            nn.ReLU(inplace = True),
            nn.AdaptiveMaxPool2d((32, 32)),
            nn.BatchNorm2d(input_dim)
            
        )
        self.class_layer1 = nn.Sequential(
            WideBottleneck(input_dim, input_dim, stride = stride, downsample = True),
            WideBottleneck(input_dim * 4, input_dim),
            WideBottleneck(input_dim * 4, input_dim)
        )
        self.class_layer2 = nn.Sequential(
            WideBottleneck(input_dim * 4, input_dim * 2, stride = stride, downsample = True),
            WideBottleneck(input_dim * 8, input_dim * 2),
            WideBottleneck(input_dim * 8, input_dim * 2),
            WideBottleneck(input_dim * 8, input_dim * 2)
        )
        self.class_layer3 = nn.Sequential(
            WideBottleneck(input_dim * 8, input_dim * 4, stride = 2, downsample = True),
            WideBottleneck(input_dim * 16, input_dim * 4),
            WideBottleneck(input_dim * 16, input_dim * 4),
            WideBottleneck(input_dim * 16, input_dim * 4),
            WideBottleneck(input_dim * 16, input_dim * 4),
            WideBottleneck(input_dim * 16, input_dim * 4)
        )
        self.class_layer4 = nn.Sequential(
            WideBottleneck(input_dim * 16, input_dim * 8, stride = stride, downsample = True),
            WideBottleneck(input_dim * 32, input_dim * 8),
            WideBottleneck(input_dim * 32, input_dim * 8)
        )
        self.finalPool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(input_dim * 32, class_number)
    
    def forward(self, x):
        x = self.default_CNN(x)
        x = self.class_layer1(x)
        x = self.class_layer2(x)
        x = self.class_layer3(x)
        x = self.class_layer4(x)
        x = self.finalPool(x)
        x = x.view(batch_size, -1)
        x = self.dropout(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x