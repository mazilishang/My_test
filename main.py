import torch
import torch.nn as nn
from torchsummary import summary
from thop import profile, clever_format

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MyNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(128 * 8 * 8, 1024)
        self.relu5 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = x.view(-1, 128 * 8 * 8)
        x = self.fc1(x)
        x = self.relu5(x)
        x = self.fc2(x)
        return x


net = MyNet().to(device)
input_shape = (3, 224, 224)


input_tensor = torch.randn(1, *input_shape).to(device)
flops, params = profile(net, inputs=(input_tensor,))
flops, params = clever_format([flops, params], "%.3f")
print("FLOPs: %s" % (flops))
print("params: %s" % (params))
