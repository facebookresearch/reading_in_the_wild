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

"""
This file demos how to read .vrs, and construct the dataset for use in the model. 
In particular, we preprocess the VRS so that we don't have to load irrelevant data
and process them every time we use it, as this is time-consuming.

As can be seen from models/data.py, we need:
- gaze_angles
- odometry
- rgb_crop_small

This demo shows how to process such file for a single gaia_id. To construct the dataset,
simply add the list to _GAIA_IDS.
"""


import os
import argparse
import torch
import cv2
import pandas as pd
import numpy as np
import json
from projectaria_tools.core import data_provider
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import StreamId
from projection_utils import project_gaze, get_default_camera_config
from PIL import Image

_GAIA_IDS = ["368375496264509"] #TODO: edit this if you want to process whole dataset.
_RAW_DATA_DIR = "../../" #TODO: point at the directory data is downloaded
_PROCESSED_DATA_DIR = "../dataset/" #TODO: point at where you want the processed data to be saved
_ANNOTATIONS_DIR = "ritw_annotations.csv"
crop_size = 64

os.makedirs(_PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(_PROCESSED_DATA_DIR, 'gaze_angles'), exist_ok=True)
os.makedirs(os.path.join(_PROCESSED_DATA_DIR, 'odometry'), exist_ok=True)
os.makedirs(os.path.join(_PROCESSED_DATA_DIR, 'rgb_crop_small'), exist_ok=True)

annotations_df = pd.read_csv(_ANNOTATIONS_DIR)

for gaia_id in _GAIA_IDS:
    metadata = annotations_df[annotations_df['id'] == int(gaia_id)]

    task = metadata['task'].item()
    if task != 16:
        start_time = metadata['whisper_start'].item() * 1000
        end_time = metadata['whisper_end'].item() * 1000
    else:
        start_time = metadata['calib'].item() * 1000
        end_time = metadata['end'].item() * 1000

    """
    GAZE: We project project the rays into the 3D space, and also calculate the 2D projection onto the 2D image.
    We actually only need the former but the rest were convenient for ablations. Differentiation is done at dataloading
    """
    
    gaze_path = os.path.join(_RAW_DATA_DIR, gaia_id, "mps", "eye_gaze", "personalized_eye_gaze.csv")
    if not os.path.exists(gaze_path): #if calibration fails we use general gaze instead.
        gaze_path = os.path.join(_RAW_DATA_DIR, gaia_id, "mps", "eye_gaze", "general_eye_gaze.csv")

    gaze = pd.read_csv(gaze_path, engine='python')

    start_time_ms = gaze['tracking_timestamp_us'].iloc[0] // 1000
    end_time_ms = gaze['tracking_timestamp_us'].iloc[-1] // 1000 
    
    gaze = gaze[(gaze['tracking_timestamp_us'] >= (start_time+start_time_ms)*1000) & (gaze['tracking_timestamp_us'] <= (end_time+start_time_ms)*1000)]

    processed_gaze_df = project_gaze(gaze).ffill().bfill()

    proj_x = processed_gaze_df.loc[:,"projected_point_2d_x"].to_numpy()
    proj_y = processed_gaze_df.loc[:,"projected_point_2d_y"].to_numpy()
    transformed_x = processed_gaze_df.loc[:,"transformed_gaze_x"].to_numpy()
    transformed_y = processed_gaze_df.loc[:,"transformed_gaze_y"].to_numpy()
    transformed_z = processed_gaze_df.loc[:,"transformed_gaze_z"].to_numpy()
    depth = processed_gaze_df.loc[:,"depth_m"].to_numpy()

    #helps to retain some data for easy ablation
    processed_gaze_df = pd.DataFrame({'time': (gaze["tracking_timestamp_us"].to_numpy() - gaze['tracking_timestamp_us'].iloc[0])/1e6,
                    'proj_x': proj_x,
                    'proj_y': proj_y,
                    'transformed_x': transformed_x,
                    'transformed_y': transformed_y,
                    'transformed_z': transformed_z,
                    'depth': depth,
                    })

    gaze_save_path = os.path.join(_PROCESSED_DATA_DIR, "gaze_angles", "{}.csv".format(gaia_id))
    processed_gaze_df.to_csv(gaze_save_path, index=False)
    
    """
    IMU: The preprocessing is minimal, just extract data such that it's synced with gaze, and subsample to 60Hz
    This helps not having to load the 800/1000Hz data. We only need the velocity, translation for ablations.
    """

    odometry_path = os.path.join(_RAW_DATA_DIR, gaia_id, "mps", "slam", "open_loop_trajectory.csv")

    odometry = pd.read_csv(odometry_path, engine='python')

    odometry = odometry[(odometry['tracking_timestamp_us'] >= (start_time+start_time_ms)*1000) & (odometry['tracking_timestamp_us'] <= (end_time+start_time_ms)*1000)]

    odometry_df = odometry[["tx_odometry_device", "ty_odometry_device", "tz_odometry_device", "qx_odometry_device", "qy_odometry_device", "qz_odometry_device", "qw_odometry_device", "device_linear_velocity_x_odometry", "device_linear_velocity_y_odometry", "device_linear_velocity_z_odometry", "angular_velocity_x_device", "angular_velocity_y_device", "angular_velocity_z_device"]]
    indices = np.linspace(0, len(odometry_df) - 1, len(processed_gaze_df), dtype=int)
    odometry_df = odometry_df.iloc[indices]

    odom_save_path = os.path.join(_PROCESSED_DATA_DIR, "odometry", "{}.csv".format(gaia_id))
    odometry_df.to_csv(odom_save_path, index=False)

    """
    RGB: we want to load and save the cropped RGB, to not have to load full RGB everytime.
    The difference in RGB frequency (30Hz vs 60Hz otherwise) is handled at dataloader, so don't worry.
    If vrs is too heavy, loading .mp4 and extracting frames is a reasonable alternative.
    """
    vrs_path = os.path.join(_RAW_DATA_DIR, gaia_id, "recording.vrs")
    provider = data_provider.create_vrs_data_provider(vrs_path)
    deliver_option = provider.get_default_deliver_queued_options()
    deliver_option.deactivate_stream_all()
    deliver_option.activate_stream(StreamId("214-1"))
    
    rgb_stream_id = StreamId("214-1")
    time_domain = TimeDomain.DEVICE_TIME  # query data based on host time
    option = TimeQueryOptions.CLOSEST

    rgb_start_time = (start_time+start_time_ms) * 1e6
    rgb_end_time = (end_time+start_time_ms) * 1e6

    rgb_save_path = os.path.join(_PROCESSED_DATA_DIR, 'rgb_crop_small', gaia_id)
    os.makedirs(rgb_save_path, exist_ok=True)
    for sample_idx, sample_time in enumerate(gaze["tracking_timestamp_us"].to_numpy()[::2]):
        image_tuple = provider.get_image_data_by_time_ns(rgb_stream_id, sample_time * 1000, time_domain, option)
        image_array = image_tuple[0].to_numpy_array()
        image = Image.fromarray(image_array)
        image = image.rotate(-90)

        x_ = 1408 - np.clip(int(proj_x[sample_idx]), crop_size//2, 1408-crop_size//2)
        y_ = np.clip(int(proj_y[sample_idx]), crop_size//2, 1408-crop_size//2)
        
        image = image.crop((x_-crop_size//2, y_-crop_size//2, x_+crop_size//2, y_+crop_size//2))
        image.save(os.path.join(rgb_save_path, str(sample_idx*2) + '.png'))
