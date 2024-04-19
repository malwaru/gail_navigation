import numpy as np
import torch
from torchvision.transforms import v2

if torch.cuda.is_available():
    DEVICE="cuda"
    PIN_MEMORY=True
else:
    DEVICE="cpu"
    PIN_MEMORY=False        
print(f"Available device is {DEVICE} of name {torch.cuda.get_device_name(torch.cuda.current_device())}\n")




def np_img_to_tensor(img):
    if len(img.shape) == 2:
        img = img[..., np.newaxis]
    img_tensor = torch.from_numpy(img)
    img_tensor = img_tensor.permute(2, 0, 1)
    img_tensor = torch.unsqueeze(img_tensor, 0)
    return img_tensor


def tensor_to_np_img(img_tensor):
    img = img_tensor.cpu().permute(0, 2, 3, 1).numpy()
    return img[0, ...]  # get the first element since it's batch form


def sobel_torch_version(img_np, torch_sobel):
    img_tensor = np_img_to_tensor(np.float32(img_np))
    img_edged = tensor_to_np_img(torch_sobel(img_tensor))
    img_edged = np.squeeze(img_edged)
    return img_edged

def preprocess(image):
    ''''
    Permute the image channel,width and height
    Normalize the image

    Finally add a batch dimension to the image

    
    '''
    if len(image.shape) == 2:

        image =  np.expand_dims(image, axis=0)
        image =  torch.from_numpy(image)
        depth_transform = v2.Compose([                      
                        v2.ToDtype(torch.float32),
                        v2.Normalize(mean=[0.485], std=[0.229]),
                    ])
        image = depth_transform(image)

    elif len(image.shape)==3:
        image =  torch.from_numpy(image)
        image=torch.permute(image, (2, 0, 1))
        rgb_transform =  v2.Compose([                      
                        v2.ToDtype(torch.float32),
                        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
        image = rgb_transform(image)



    return image.unsqueeze(0)





##Path to data folder
# CURRENT_PATH=os.getcwd()
# DATASET_PATH=os.path.join(CURRENT_PATH,"data")
# IMAGE_DATASET_PATH = os.path.join(DATASET_PATH,"images")
# MASK_DATASET_PATH=os.path.join(DATASET_PATH,"masks")
# print(f"the data path is at {DATASET_PATH} \n and the masks are at {MASK_DATASET_PATH}\n")
##Train | Validation | Test Split
# SPLIT=[0.5,0,25,0.25]
#Test data use 15% of all data rest for training 
# TEST_SPLIT=0.15
##Check if GPU available 

# define the number of channels in the input
# and number of levels to ablate from the ResNet
## not used right now but can be used to choose how many levels
# NUM_LEVELS = 3 
# initialize learning rate, number of epochs to train for, and the
# batch size
# INIT_LR = 0.001
# NUM_EPOCHS = 100
# BATCH_SIZE = 16 ## My  PC stop working at 128 
# define the input image dimensions
# INPUT_IMAGE_WIDTH = 120
# INPUT_IMAGE_HEIGHT = 240
# define threshold to filter weak predictions
# THRESHOLD = 0.5
# define the path to the base output directory
# BASE_OUTPUT = "output"
# define the path to the output serialized model, model training
# plot, and testing image paths unet_tgs_forest_plot_100_epochs_64channel / unet_tgs_forest_plot_100_epochs_256channel
# MODEL_PATH = os.path.join(BASE_OUTPUT, "custom.pth")
# PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
# TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])