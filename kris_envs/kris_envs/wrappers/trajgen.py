from imitation.data.types import Trajectory
from imitation.data import rollout
import numpy as np
import h5py
from GailNavigationNetwork.model import NaviNet
from GailNavigationNetwork.utilities import preprocess
from kris_envs.wrappers.utilities import normalise_action, transform_to_int8, preprocess_target,scale_arrays
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

        The array of odom_data is shifted by one time stamp to get
        the action of the agent based on where the robot actually ended up in the next
        time step

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
            target=preprocess_target(target)
            (rgb, depth,target) = (rgb.to(DEVICE),depth.to(DEVICE),target.to(DEVICE))
            rgb_features, depth_features,target_features = model(rgb,depth,target)
            #Detach the tensors and convert to numpy arrays and scale the arrays
            rgb_features=rgb_features.detach().cpu().numpy()
            rgb_features=scale_arrays(rgb_features)
            depth_features=depth_features.detach().cpu().numpy()
            depth_features=scale_arrays(depth_features)
            target_features=target_features.detach().cpu().numpy()
            target_features=scale_arrays(target_features)
            ## If visualisation is needed
            if self.visualise_img:
                self.visualise(rgb_features,depth_features)  
            rgbs.append(rgb_features)
            depths.append(depth_features)
            targets.append(target_features) 
            acts.append(act)
            
        #Create actions by shifting the odom array to get where the agent moveed 
        # In the next time step
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
        dones=[]
        
        files=glob.glob(self.file_path+'/*.hdf5')
        total_batch_size=0
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
                target=preprocess_target(target)
                (rgb, depth,target) = (rgb.to(DEVICE),depth.to(DEVICE),target.to(DEVICE))
                rgb_features, depth_features,target_features = model(rgb,depth,target)
                #Detach the tensors and convert to numpy arrays and scale the arrays
                rgb_features=rgb_features.detach().cpu().numpy()
                rgb_features=scale_arrays(rgb_features)
                depth_features=depth_features.detach().cpu().numpy()
                depth_features=scale_arrays(depth_features)
                target_features=target_features.detach().cpu().numpy()
                target_features=scale_arrays(target_features)
                ## If visualisation is needed
                if self.visualise_img:
                    self.visualise(rgb_features,depth_features)  
                rgbs.append(rgb_features)
                depths.append(depth_features)
                targets.append(target_features) 
                acts.append(act)
            read_file.close()
            total_batch_size+=batch_size
            dones.append([False for i in range(batch_size)])
            dones[-1]=True
                

        acts=np.array(acts[:-1])
        # dones=[False for i in range(total_batch_size)]
        # dones[-1]=True
        infos= [{} for i in range(total_batch_size-1)]
        rgbs=np.array(rgbs)
        depths=np.array(depths)
        targets=np.array(targets)
        obs_array=np.concatenate((targets,rgbs,depths),axis=1)
        print(f"[kris_env:trajgen] observation array shape {obs_array.shape}")
        traj = Trajectory(obs=obs_array, acts=acts,infos=infos,terminal=dones)

        return total_batch_size,rollout.flatten_trajectories([traj])