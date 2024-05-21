from imitation.data.types import Trajectory
from imitation.data import rollout
import numpy as np
import h5py
from GailNavigationNetwork.model import NaviNet
from GailNavigationNetwork.utilities import preprocess
from kris_envs.wrappers.utilities import normalise_action, transform_to_int8
import cv2
import glob



class TrajFromFile:
    def __init__(self,file_path,visualise_img=False) -> None:
        '''
        Args:
        file_path: str 
        Path to the hdf5 file
        visualise: bool
        If true, visualises the images

        '''
        self.file_path=file_path
        self.visualise_img=visualise_img


    def visualise(self,rgb,depth):
           '''
           
           Visualises the rgb and depth images
           '''
           for i in range(len(rgb.shape[0])):
            cv2.imshow("depth_image",depth)
            cv2.imshow("rgb_image",rgb[i])
            cv2.waitKey(0)



    def create_demos_from_file(self,DEVICE="cuda"):
        '''
        Creates a gymnasium transition from the given file path
        of hdf5 file of known structure

        Args:
        file_path: str  
        Path to the hdf5 file   

        Returns:
        batch_size: int
        rollouts: gymnasium.Transition

        '''    
        read_file= h5py.File(self.file_path, "r")
        model= NaviNet().to(DEVICE)
        model.eval()
        batch_size= read_file['kris_dynamics']['odom_data']['target_vector'].shape[0]
        rgbs=[]
        depths=[]
        targets=[]  
        acts=[]
        for i in range(batch_size):
            target=read_file['kris_dynamics']['odom_data']['target_vector'][i]
            rgb=read_file['images']['rgb_data'][i]
            depth=read_file['images']['depth_data'][i]
            act=read_file['kris_dynamics']['odom_data']['odom_data_wheel'][i]
            # Normalise action to -1 to 1
            act=normalise_action(act)
            rgb=preprocess(rgb)
            # Convert depth to int8
            depth=transform_to_int8(depth)
            depth=preprocess(depth)
            (rgb, depth) = (rgb.to(DEVICE), depth.to(DEVICE))
            rgb_features, depth_features = model(rgb,depth)
            rgb_features=rgb_features.detach().cpu().numpy()
            depth_features=depth_features.detach().cpu().numpy()
            ## If visualisation is needed
            if self.visualise_img:
                self.visualise(rgb_features,depth_features)  
            rgbs.append(rgb_features.flatten())
            depths.append(depth_features.flatten())
            targets.append(target.flatten()) 
            acts.append(act)
            

        acts=np.array(acts[:-1])
        dones=[False for i in range(batch_size)]
        dones[-1]=True
        infos= [{} for i in range(batch_size-1)]
        rgbs=np.array(rgbs)
        depths=np.array(depths)
        targets=np.array(targets)
        obs_array=np.concatenate((targets,rgbs,depths),axis=1)
        print(f"[kris_env:trajgen] observation array shape {obs_array.shape}")
        traj = Trajectory(obs=obs_array, acts=acts,infos=infos,terminal=dones)

        return batch_size,rollout.flatten_trajectories([traj])
    


    def create_demos_from_folder(self,DEVICE="cuda"):
        '''
        Creates a gymnasium transition from the given file path
        of hdf5 file of known structure

        Args:
        folder_path: str  
        Path to the hdf5 file   

        Returns:
        batch_size: int
        rollouts: gymnasium.Transition

        '''    
        rgbs=[]
        depths=[]
        targets=[]  
        acts=[]
        files=glob.glob(self.file_path+'/*.hdf5')
        for file in files:
            read_file= h5py.File(file, "r")
            model= NaviNet().to(DEVICE)
            model.eval()
            batch_size= read_file['kris_dynamics']['odom_data']['target_vector'].shape[0]           
            for i in range(batch_size):
                target=read_file['kris_dynamics']['odom_data']['target_vector'][i]
                rgb=read_file['images']['rgb_data'][i]
                depth=read_file['images']['depth_data'][i]
                act=read_file['kris_dynamics']['odom_data']['odom_data_wheel'][i]
                # Normalise action to -1 to 1
                act=normalise_action(act)
                rgb=preprocess(rgb)
                # Convert depth to int8
                depth=transform_to_int8(depth)
                depth=preprocess(depth)
                (rgb, depth) = (rgb.to(DEVICE), depth.to(DEVICE))
                rgb_features, depth_features = model(rgb,depth)
                rgb_features=rgb_features.detach().cpu().numpy()
                depth_features=depth_features.detach().cpu().numpy()
                ## If visualisation is needed
                if self.visualise_img:
                    self.visualise(rgb_features,depth_features)  
                rgbs.append(rgb_features.flatten())
                depths.append(depth_features.flatten())
                targets.append(target.flatten()) 
                acts.append(act)
            read_file.close()
                

        acts=np.array(acts[:-1])
        dones=[False for i in range(batch_size)]
        dones[-1]=True
        infos= [{} for i in range(batch_size-1)]
        rgbs=np.array(rgbs)
        depths=np.array(depths)
        targets=np.array(targets)
        obs_array=np.concatenate((targets,rgbs,depths),axis=1)
        print(f"[kris_env:trajgen] observation array shape {obs_array.shape}")
        traj = Trajectory(obs=obs_array, acts=acts,infos=infos,terminal=dones)

        return batch_size,rollout.flatten_trajectories([traj])