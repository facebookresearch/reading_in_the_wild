# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc/4.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import random
import numpy as np
import pandas as pd
import torch
import cv2
from natsort import natsorted
from torch.utils.data import Dataset
from utils import create_sampled_array, get_labels, modality_dropout, gaze_rotate


class RitWDataset(Dataset):
    def __init__(self, input_sec, input_hz, gaze_input, imu_input, rgb_input, split, labels):
        data_path = "/source_1a/data/reading_in_the_wild/"
        metadata_path = os.path.join("../reading_in_the_wild_seattle/ritw_annotations.csv")
        gaze_path = os.path.join(data_path, "gaze_angles")
        odometry_path = os.path.join(data_path, "odometry")
        rgb_path = os.path.join(data_path, "rgb_crop_small")

        no_odom_ids = [888330739387898, 470176932007502, 3879462675663355]
        df = pd.read_csv(metadata_path)

        self.num_samples = int(input_sec * input_hz)
        reading_only = False
        self.split = split
        self.use_gaze = "XYZ" in gaze_input or "xy" in gaze_input or "yp" in gaze_input
        self.use_imu = "t" in imu_input or "q" in imu_input or "v" in imu_input or "w" in imu_input
        self.use_rgb = "small" in rgb_input
        self.all_snippets = []
        self.crop = None

        if type(labels) == str:
            if labels == 'age': label_list = {"18-24": 0, "25-30": 0, "31-35": -1,"36-40": -1, "41-45": 1,"46-50": 1}
            if labels == 'gender': label_list = {"Male": 0, "Female": 1}
            if labels == 'reading': label_list = {False: 0, True: 1}
            if labels == 'medium': label_list = {'Print': 0, 'Digital': 1, 'Object': 2}
            if labels == 'mode': label_list = {'Engaged reading': 0, 'Walking': 1, 'Writing/typing': 2, 'Videos': 3, 'Skimming': 4, 'Read out loud': 5, 'Scanning': 6}
            if labels == 'device': label_list = {'Laptop': 0, 'Screen': -1, 'Phone': 2, 'Tablet': 1}
            if labels in ['age', 'gender', 'medium', 'mode', 'device']:
                self.num_classes = max(label_list.values()) + 1
                reading_only = True
            else:
                self.num_classes = 2
        else:
            self.num_classes = 2

        for index, row in df.iterrows():
            label = None
            gaia_id = row['id']
            task = row['task']
            data_split = row['split']
            if type(labels) == str:
                if labels in row:
                    label = row[labels]
                    if label is None or pd.isna(label): continue
                    label = label_list[label]
            length = None
            if imu_input and gaia_id in no_odom_ids: continue
            if task == 16: continue
            if self.split != data_split: continue
            if type(labels) == list:
                if labels[task-1] == -1: continue
            if reading_only and task in [8,9,10,11]: continue

            #Gaze
            if self.use_gaze:
                gaze = pd.read_csv(os.path.join(gaze_path, str(gaia_id)+'.csv'))
                gaze_list = []
                if "XYZ" in gaze_input: #3D gaze ray
                    gaze_list.extend(['transformed_x', 'transformed_y', 'transformed_z'])
                if "xy" in gaze_input: #2D gaze projection
                    gaze_list.extend(['proj_x', 'proj_y'])            
                if "yp" in gaze_input:
                    gaze_list.extend(['left_yaw_rads_cpf', 'right_yaw_rads_cpf', 'pitch_rads_cpf'])
                if "z" in gaze_input:
                    gaze_list.extend(['depth'])
                gaze_sequence = gaze[gaze_list]
                if "d" in gaze_input: #differentiate
                    gaze_sequence = create_sampled_array(gaze_sequence, num_samples=self.num_samples+1, stride=60 // input_hz)
                    gaze_sequence = torch.Tensor(np.diff(gaze_sequence, axis=1) * input_hz)
                else:
                    gaze_sequence = create_sampled_array(gaze_sequence, num_samples=self.num_samples, stride=60 // input_hz) 
                    gaze_sequence = torch.Tensor(gaze_sequence)
                if length is None: length = len(gaze_sequence)
                
            #IMU
            if self.use_imu:
                #default: vw (linear and anglular velocity)
                odometry = pd.read_csv(os.path.join(odometry_path, str(gaia_id)+'.csv'))
                odom_list = []
                if "t" in imu_input:
                    odom_list.extend(["tx_odometry_device", "ty_odometry_device", "tz_odometry_device"])
                if "q" in imu_input:
                    odom_list.extend(["qx_odometry_device", "qy_odometry_device", "qz_odometry_device", "qw_odometry_device"])
                if "v" in imu_input:
                    odom_list.extend(["device_linear_velocity_x_odometry", "device_linear_velocity_y_odometry", "device_linear_velocity_z_odometry"])
                if "w" in imu_input:
                    odom_list.extend(["angular_velocity_x_device", "angular_velocity_y_device", "angular_velocity_z_device"])
                odometry_sequence = odometry[odom_list]
                odometry_sequence = create_sampled_array(odometry_sequence, num_samples=self.num_samples, stride=60 // input_hz) 
                odometry_sequence = torch.Tensor(odometry_sequence)
                if length is None: length = len(odometry_sequence)
                
            #RGB
            if self.use_rgb:
                #default: 64 (5 degrees FoV)
                if rgb_input[5:].isdigit():
                    self.crop = int(rgb_input[5:])
                rgb_sequence = natsorted([os.path.join(rgb_path, str(gaia_id), x) for x in os.listdir(os.path.join(rgb_path, str(gaia_id))) if '.png' in x])
                rgb_length = len(rgb_sequence)
                if length is None: length = len(rgb_sequence)
            
            if label is None:
                label = get_labels(labels, task, length, gaia_id)
            if label == -1: continue


            for i in range(length):
                snippet = {}
                if self.use_gaze:
                    snippet['gaze'] = gaze_sequence[i]
                if self.use_imu:
                    snippet['odom'] = odometry_sequence[i]
                if self.use_rgb:
                    snippet['rgb'] = rgb_sequence[np.clip(int(i * rgb_length / length), 0, rgb_length)]
                snippet['task'] = task
                snippet['label'] = label[i] if type(label) == list else label
                self.all_snippets.append(snippet)        
        self.all_snippets = pd.DataFrame(self.all_snippets) #store in df to prevent OOM (important)


    def __len__(self):
        return len(self.all_snippets)

    def __getitem__(self, idx):
        snippet = self.all_snippets.iloc[idx].to_dict()
        if self.use_rgb:
            image = cv2.imread(snippet['rgb'])
            snippet['rgb'] = (torch.Tensor(image) / 255.).permute(2, 0, 1)
            if self.crop is not None:
                start_x = (176 - self.crop) // 2
                start_y = (176 - self.crop) // 2
                snippet['rgb'] = snippet['rgb'][:, start_y:start_y + self.crop, start_x:start_x + self.crop]     

        if self.split == "train":
            if self.use_gaze:
                snippet['gaze'] *= torch.normal(1, 0.05, size=[1]) #scale
                snippet['gaze'] += torch.normal(0, 0.005, size=snippet['gaze'].size()) #noise
            if self.use_imu:
                snippet['odom'] *= torch.normal(1, 0.05, size=[1]) #scale
                snippet['odom'] += torch.normal(0, 0.005, size=snippet['odom'].size()) #noise
         
            snippet = modality_dropout(snippet, self.use_rgb, self.use_imu, self.use_gaze)
            if random.random() < 0.1:
                snippet = gaze_rotate(snippet, self.use_rgb, self.use_imu, self.use_gaze)
        return snippet

    def get_num_classes(self):
        return self.num_classes

    def get_class_weights(self):
        # Count the number of samples per class, ignoring labels like -1
        label_counts = {}
        for idx, snippet in self.all_snippets.iterrows():
            label = snippet['label']
            if label == -1:
                continue  # Skip samples with label -1
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1
        # Calculate weights
        total_samples = sum(label_counts.values())  # Only consider valid samples
        num_classes = len(label_counts)
        class_weights = {label: total_samples / (num_classes * count) for label, count in label_counts.items()}
        # Convert to a tensor
        weights_tensor = torch.tensor([class_weights[label] for label in sorted(class_weights.keys())], dtype=torch.float32)
        return weights_tensor

def load_data(task_name, input_sec, input_hz, gaze_input, imu_input, rgb_input): 
    tasks = {
        "binary" : [1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,-1,1,1,1], 
        "scenario": [1,1,2,2,2,3,1,0,0,0,0,4,5,-1,-1,-1,-1,-1,6], #out loud, normal, scan, walking,type, skim
        "media": [1,3,1,1,2,2,3,0,0,0,0,-1,-1,-1,3,-1,-1,-1,-1], #by media type : print/digital/objects
        "all": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,-1,15,16,17],
        "readonly": [1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,1,1],
        }
    if task_name in tasks:
        labels = tasks[task_name] 
    else: #    ['age', 'gender', 'reading', 'medium', 'mode', 'device']
        labels = task_name

    gaze_dim = 3 * int(("XYZ" in gaze_input)) + 2 * int(("xy" in gaze_input)) + 3 * int(("yp" in gaze_input)) + 1 * int(("z" in gaze_input))
    imu_dim = 3 * int(("t" in imu_input)) + 4 * int(("q" in imu_input)) + 3 * int(("v" in imu_input)) + 3 * int(("w" in imu_input))
    rgb_dim = 3 * int(("small" in rgb_input))
    train_set = RitWDataset(input_sec=input_sec, input_hz=input_hz, gaze_input=gaze_input, imu_input=imu_input, rgb_input=rgb_input, split="train", labels=labels)
    val_set = RitWDataset(input_sec=input_sec, input_hz=input_hz, gaze_input=gaze_input, imu_input=imu_input, rgb_input=rgb_input, split="val", labels=labels)
    test_set = RitWDataset(input_sec=input_sec, input_hz=input_hz, gaze_input=gaze_input, imu_input=imu_input, rgb_input=rgb_input, split="test", labels=labels)
    num_classes = train_set.get_num_classes()
    loss_weight = train_set.get_class_weights()
    return train_set, val_set, test_set, num_classes, [gaze_dim, imu_dim, rgb_dim], loss_weight
