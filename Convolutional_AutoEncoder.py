import torch
import torch.nn as nn

new_height = 256
new_width = 256

class Convolutional_AutoEncoder(nn.Module):
    def __init__(self, RGB = 3, ckernel_size = 4, pkernel_size = 3,  stride = 2, padding = 1, bias = True):
        super(Convolutional_AutoEncoder, self).__init__()
        self.enc_layer1 = nn.Sequential(
            nn.Conv2d(RGB, RGB * 2, kernel_size = ckernel_size, stride = stride, padding = padding, bias = True),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = pkernel_size, stride = stride - 1),
            nn.InstanceNorm2d(RGB * 2, affine = True)
        )
    
        self.enc_layer2 = nn.Sequential(
            nn.Conv2d(RGB * 2, RGB * 4, kernel_size = ckernel_size, stride = stride, padding = padding, bias = True),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = pkernel_size, stride = stride - 1),
            nn.InstanceNorm2d(RGB * 4, affine = True)
        )

        self.dec_layer1 = nn.Sequential(
            nn.ConvTranspose2d(
                RGB * 4, RGB * 4, kernel_size = ckernel_size + 1, 
                stride = stride - 1, padding = padding
            ),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(
                RGB * 4, RGB * 2, kernel_size = ckernel_size,
                stride = stride, padding = padding, bias = True
            ),
            nn.InstanceNorm2d(RGB * 2, affine = True)
        )

        self.dec_layer2 = nn.Sequential(
            nn.ConvTranspose2d(
                RGB * 2, RGB * 2, kernel_size = ckernel_size + 1, 
                stride = stride - 1, padding = padding
            ),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(
                RGB * 2, RGB, kernel_size = ckernel_size,
                stride = stride, padding = padding, bias = True
            ),
            nn.InstanceNorm2d(RGB, affine = True)
        )

        self.latent_variable = nn.Sequential(
            nn.Conv2d(RGB * 4, RGB, kernel_size = 1, padding = 0, stride = 1),
            nn.UpsamplingBilinear2d(size = (new_height, new_width)),
            nn.InstanceNorm2d(RGB, affine = True)
        )
    
    def encoder(self, x):
        x = self.enc_layer1(x)
        x = self.enc_layer2(x)
        return x
    
    def decoder(self, x):
        x = self.dec_layer1(x)
        x = self.dec_layer2(x)
        return x
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def image_change(self, x):
        x = self.encoder(x)
        x = self.latent_variable(x)
        return x
    
    def loss(self, x):
        y = self.encoder(x)
        y = self.decoder(y)
        # reconstruction lossの計算
        reconstruction = torch.mean(torch.sum(-x * self.torch_log(y) - (1 - x) * self.torch_log(1 - y), dim = 1))
        return reconstruction 
    
    # torch.log(0)によるnanを防ぐ
    def torch_log(self, x):
        return torch.log(torch.clamp(x, min = 1e-10))