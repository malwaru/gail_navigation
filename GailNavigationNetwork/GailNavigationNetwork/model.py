import torch
from torch.nn import Conv2d, Linear,Parameter
from torch.nn import Module
from torchvision.models import efficientnet_b1,EfficientNet_B1_Weights



class RGBNet(Module):
    '''
    Extracts the features from the RGB image from efficientnet_b1
    architecture and returns the features of last fully connected layer
    '''
    def __init__(self,ablation_depth=2):
        super().__init__()
        resnet_model = efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V2)
        modules = list(resnet_model.children())[:-ablation_depth]
        self.backbone = torch.nn.Sequential(*modules)

    def forward(self, x):
        x = self.backbone(x)
        return x

class DepthNet(Module):
    '''
    Extract the features from the depth image using Sobel filter and
    pass it through a fully connected layer to get same dimension as RGBNet

    
    '''
    def __init__(self):
        super().__init__()
        self.filter = Conv2d(in_channels=1, out_channels=2, kernel_size=3, 
                             stride=1, padding=0, bias=False)

        Gx = torch.tensor([[2.0, 0.0, -2.0], [4.0, 0.0, -4.0], [2.0, 0.0, -2.0]])
        Gy = torch.tensor([[2.0, 4.0, 2.0], [0.0, 0.0, 0.0], [-2.0, -4.0, -2.0]])
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(1)
        self.filter.weight = Parameter(G, requires_grad=False)
        self.fc=Linear(238*318,1280)

    def forward(self, img):
        x = self.filter(img)
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        

        return x
    
class TargetNet(Module):
    '''
    Extract the features from the target vector using two fully connected layers
    so that it has the same dimension as RGBNet
    
    '''
    def __init__(self,in_channels, intermediate_channels, output_size):
        super().__init__()
        # Define the first fully connected layer
        self.fc1 = Linear(in_channels, intermediate_channels)        
        # Define the second fully connected layer
        self.fc2 = Linear(intermediate_channels, output_size)
        
    def forward(self,x):
        x = self.fc1(x)
        x = torch.tanh(x)        
        x = self.fc2(x)
        return x

class NaviNet(Module):
    '''
    A network that combines the features from RGBNet, DepthNet and TargetNet
    '''
    def __init__(self,
                 image_dims=(240,320),
                 goal_dims=7):
         super(NaviNet, self).__init__()
         self.depth_net = DepthNet()
         self.rgb_net = RGBNet(ablation_depth=1)
         self.target_net=TargetNet(goal_dims,640,1280)

    def forward(self,rgb_image, depth_image,target):      
        rgb_features = self.rgb_net(rgb_image).squeeze()
        depth_features = self.depth_net(depth_image).squeeze()
        taget_features = self.target_net(target)
        return (rgb_features, depth_features,taget_features)