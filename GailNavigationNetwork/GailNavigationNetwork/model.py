import torch
from torch.nn import Conv2d, Linear,Parameter
from torch.nn import Module
from torchvision.models import efficientnet_b1,EfficientNet_B1_Weights
from torchvision.transforms import v2



class RGBNet(Module):
    def __init__(self,ablation_depth=2):
        super().__init__()
        resnet_model = efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V2)
        modules = list(resnet_model.children())[:-ablation_depth]
        self.backbone = torch.nn.Sequential(*modules)

    def forward(self, x):
        x = self.backbone(x)
        return x

class DepthNet(Module):
    def __init__(self):
        super().__init__()
        self.filter = Conv2d(in_channels=1, out_channels=2, kernel_size=3, 
                             stride=1, padding=0, bias=False)

        Gx = torch.tensor([[2.0, 0.0, -2.0], [4.0, 0.0, -4.0], [2.0, 0.0, -2.0]])
        Gy = torch.tensor([[2.0, 4.0, 2.0], [0.0, 0.0, 0.0], [-2.0, -4.0, -2.0]])
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(1)
        self.filter.weight = Parameter(G, requires_grad=False)

    def forward(self, img):
        x = self.filter(img)
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x)
        return x

class NaviNet(Module):
    '''
    A deeplearning architecture for local navigation planning
    '''
    def __init__(self,
                 image_dims=(240,320),
                 goal_dims=7):
         super(NaviNet, self).__init__()
         self.depth_net = DepthNet()
         self.rgb_net = RGBNet(ablation_depth=2)
        #  self.fc_goal_pose = Linear(goal_dims, 128)   

    def forward(self, rgb_image, depth_image):
        rgb_features = self.rgb_net(rgb_image).squeeze()
        depth_features = self.depth_net(depth_image).squeeze()
        return (rgb_features, depth_features)