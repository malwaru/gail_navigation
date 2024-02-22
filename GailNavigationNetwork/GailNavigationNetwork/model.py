from GailNavigationNetwork import utilities
import torch
from torch.nn import Conv2d, Linear
from torch.nn import Module
from torchvision.models import resnet18, ResNet18_Weights


class RGBNet(Module):
    def __init__(self, dims=(120,160,3),ablation_depth=2):
        super().__init__()
        resnet_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        modules = list(resnet_model.children())[:-ablation_depth]
        self.backbone = torch.nn.Sequential(*modules)

    def forward(self, x):
        x = self.backbone(x)
        return x

class DepthNet(Module):
    def __init__(self,dims=(120,160,1)):
        super(DepthNet, self).__init__()
        self.conv1 = Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = Linear(32 * dims[0] * dims[1] // 4, 128)  # Adjust the input size based on your depth image size

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        return x

class NaviNet(Module):
    '''
    A deeplearning architecture for local navigation planning
    '''
    def __init__(self,
                 image_dims=(120,160),
                 goal_dims=7):
         super(NaviNet, self).__init__()
         self.depth_net = DepthNet(dims=(image_dims[0],image_dims[1],1))
         self.rgb_net = RGBNet(dims=(image_dims[0],image_dims[1],3))
         self.fc_goal_pose = Linear(goal_dims[0], 128) 

    def forward(self, rgb_image, depth_image, goal_pose):
        rgb_features = self.rgb_net(rgb_image)
        depth_features = self.depth_net(depth_image)
        goal_pose = torch.relu(self.fc_goal_pose(goal_pose))
        
        # Concatenate features
        concatenated_features = torch.cat((rgb_features.squeeze(), depth_features, goal_pose), dim=1)
        
        return concatenated_features
