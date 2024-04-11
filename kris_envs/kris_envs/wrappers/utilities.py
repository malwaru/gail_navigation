import numpy as np


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