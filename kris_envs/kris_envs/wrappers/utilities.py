import numpy as np
import cv2
import torch

def normalise_action(act_arr,
                     act_lin_rng=[-1,1],
                     act_ang_rng=[-0.785398,0.785398]):
    '''
    Normalise the action to the range of -1 to 1

    Args:
    act_arr     : np.array of shape (7,) 
                  Unnormalised action
                  with x,y,z,i,j,k,w
    act_lin_rng : the maximum and minimum values of the linear action
    act_ang_rng : the maximum and minimum values of the angular action

    '''

     # Normalize the linear x , y ,z values
    act_arr[:3] = (act_arr[:3] - act_lin_rng[0]) / (act_lin_rng[1] - act_lin_rng[0]) * 2 - 1
    # Normalize the angular i , j , k , w values
    act_arr[:3] = (act_arr[:3] - act_ang_rng[0]) / (act_ang_rng[1] - act_ang_rng[0]) * 2 - 1
    
    return act_arr

def denormalise_action(act_arr,
                       act_lin_rng=[-1,1],
                       act_ang_rng=[-0.785398,0.785398]):
    '''
    Denormalise the action to the range of -1 to 1

    Args:
    act_arr     : np.array of shape (7,)
                  Normalised action
                  with x,y,z,i,j,k,w
    act_lin_rng : the maximum and minimum values of the linear action
    act_ang_rng : the maximum and minimum values of the angular action
    '''
   # Denormalize the first three elements
    act_arr[:3] = (act_arr[:3] + 1) / 2 * (act_lin_rng[1] - act_lin_rng[0]) + act_lin_rng[0]
    # Denormalize the last four elements
    act_arr[3:] = (act_arr[3:] + 1) / 2 * (act_ang_rng[1] - act_ang_rng[0]) + act_ang_rng[0]
    
    return act_arr



def scale_arrays(array,new_min=-1, new_max=1):
    '''
    Scales the array to a new range
    Default to -1 to 1
    
    '''
    original_min = np.min(array)
    original_max = np.max(array)
    scaled_array = ((array - original_min) / (original_max - original_min)) * (new_max - new_min) + new_min
    return scaled_array

def transform_to_int8(arr,old_max=50.0):
    """
    Transform float 32 depth array to unsigned int 8 depth array.

    Args:
        arr (numpy.ndarray): The 2D array to be remapped.
        old_max (float): The maximum depth value 
                        Default max depth is 50.0 meters
        

    Returns:
        numpy.ndarray: The remapped 2D array with values as integers.
    """
    # Check if any value in the array is infinity and replace it with the maximum value
    arr = np.where(np.isinf(arr), old_max, arr)
    arr = arr/ old_max
    # normalize the data to 0 - 1
    rescaled_arr = 255 * arr # Now scale by 255
    return rescaled_arr.astype(np.int8)

def img_resize(img,scale=1.0):
        '''
        Resize the image to the given scale

        Args:
            img:   (np.array)
                    The image to be resized
            scale: (float)
                    The scale to resize the image to
        Returns:
            The resized image (np.array)
        
        '''
        return cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)), interpolation=cv2.INTER_AREA)

def preprocess_target(target):
    '''
    Preprocess the target vector to float32 tensor

    Args:
    img: np.array
         The target vector to be preprocessed
    Returns:
    The preprocessed image
    '''
    target=np.array(target,dtype=np.float32)
    target=np.reshape(target,-1)
    return torch.from_numpy(target)
