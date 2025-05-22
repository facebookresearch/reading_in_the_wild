# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import os
import argparse
import torch
import cv2
import pandas as pd
import numpy as np
from projectaria_tools.core import data_provider
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import StreamId
from torch.utils.data import DataLoader
from projection_utils import project_gaze_vrs, project_gaze
from utils import create_sampled_array
import glob
import torch.nn.functional as F
import matplotlib.pyplot as plt
from model import MultimodalTransformer

def inference(args):
    name = None
    # params
    input_hz = 60
    input_sec = 2
    crop_size = 64
    num_classes = 2
    
    name = args.model_name 
    if name == 'v1_15Hz':
        input_hz = 15
    if name == 'v1_1s':
        input_sec = 1
    if name == 'v1_mode':
        num_classes = 7 # not reading, out loud, normal, scan, walking , write/type, skim
    if name == 'v1_medium':
        num_classes = 4 # not reading, print, digital, object
    if name == 'v1_large':
        crop_size = 128
    device = 'cuda'
    input_length = input_hz * input_sec
    model = MultimodalTransformer(num_classes=2, dim_feat=32, input_dim=[3,6,3], sequence_length=input_length)
    checkpoint_path = os.path.join('models', name + '.ckpt')
    checkpoint = torch.load(checkpoint_path)
    checkpoint['state_dict'] = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items() if not k.startswith('loss_fn.')}
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model.to(device)
    
    recordings_dir = os.path.join(args.root_dir, args.vid_uid)

    ## for data recorded directly using Aria glasses (this should be the case for Columbus):
    if os.path.exists(os.path.join(recordings_dir, args.vid_uid + ".vrs")):  
        vrs_path = os.path.join(recordings_dir, args.vid_uid + ".vrs")
        gaze_path = os.path.join(recordings_dir, "mps_output", "eye_gaze", "personalized_eye_gaze.csv")
        if not os.path.exists(gaze_path):
            gaze_path = os.path.join(recordings_dir, "mps_output", "eye_gaze", "general_eye_gaze.csv")        
        gaze = project_gaze_vrs(gaze_path, vrs_path=vrs_path)
        if args.use_imu:
            odometry_path = os.path.join(recordings_dir, "mps_output", "slam", "open_loop_trajectory.csv")
            odometry = pd.read_csv(odometry_path, engine='python')
            odometry = odometry[["device_linear_velocity_x_odometry", "device_linear_velocity_y_odometry", "device_linear_velocity_z_odometry", "angular_velocity_x_device", "angular_velocity_y_device", "angular_velocity_z_device"]]            
            indices = np.linspace(0, len(odometry) - 1, len(gaze), dtype=int) #resample to 60Hz
            odometry = odometry.iloc[indices]
        if args.use_rgb:
            provider = data_provider.create_vrs_data_provider(vrs_path)
            deliver_option = provider.get_default_deliver_queued_options()
            deliver_option.deactivate_stream_all()
            deliver_option.activate_stream(StreamId("214-1"))
    ## for data recorded via Gaia data campaign (this should be the case for Seattle)
    else:
        gaze_path = os.path.join(recordings_dir, "EyeGaze", "personalized_eye_gaze.csv")
        if not os.path.exists(gaze_path):
            gaze_path = os.path.join(recordings_dir, "mps_output", "eye_gaze", "general_eye_gaze.csv")
        config_path = os.path.join(recordings_dir, "vrs_videos", "camera_config.json")
        gaze = project_gaze(gaze_path, config_path=config_path)
        if args.use_imu:
            odometry_path = os.path.join(recordings_dir, "SingleSequenceTrajectory", "open_loop_trajectory.csv")
            odometry = pd.read_csv(odometry_path, engine='python')
            odometry = odometry[["device_linear_velocity_x_odometry", "device_linear_velocity_y_odometry", "device_linear_velocity_z_odometry", "angular_velocity_x_device", "angular_velocity_y_device", "angular_velocity_z_device"]]            
            indices = np.linspace(0, len(odometry) - 1, len(gaze), dtype=int) #resample to 60Hz
            odometry = odometry.iloc[indices]
        if args.use_rgb:
            vrs_path = glob.glob(os.path.join(recordings_dir,"vrs_videos", "*Scene.vrs"))[0]
            provider = data_provider.create_vrs_data_provider(vrs_path)
            deliver_option = provider.get_default_deliver_queued_options()
            deliver_option.deactivate_stream_all()
            deliver_option.activate_stream(StreamId("214-1"))
    
    # input processing, create 2s long snippets [T-2, T] for gaze and IMU for prediction at time T
    gaze_sequence = gaze[['transformed_gaze_x', 'transformed_gaze_y', 'transformed_gaze_z']].ffill()
    gaze_sequence = create_sampled_array(gaze_sequence, num_samples=input_length+1, stride=60//input_hz)
    gaze_sequence = torch.Tensor(np.diff(gaze_sequence, axis=1) * input_hz)
    num_gaze = gaze_sequence.size(0)
    if args.use_imu:
        odometry_sequence = create_sampled_array(odometry, num_samples=input_length, stride=60//input_hz)
        odometry_sequence = torch.Tensor(odometry_sequence)[:-1]
        num_odom = odometry_sequence.size(0)
    if args.use_rgb:
        gaze_xy = np.array(gaze[['projected_point_2d_x', 'projected_point_2d_y']].ffill())
        gaze_timestamps = gaze['tracking_timestamp_us'].tolist()
        num_rgb = provider.get_num_data(StreamId("214-1")) - 60 #ignore first 2s
    test_set = []
    for i in range(round(args.start_time*60), num_gaze, round(args.snippet_gap*60)):
        snippet = {
            'gaze': torch.zeros((input_length, 3), dtype=torch.float32),
            'odom': torch.zeros((input_length, 6), dtype=torch.float32),
            'rgb': torch.zeros((3, crop_size, crop_size), dtype=torch.float32)
        }
        if args.use_gaze:
            snippet['gaze'] = gaze_sequence[i]
        if args.use_imu:
            snippet['odom'] = odometry_sequence[i]
        if args.use_rgb:
            gaze_idx = i+input_length #use RGB at the end of gaze sequence
            time = gaze_timestamps[gaze_idx] * 1000
            im = provider.get_image_data_by_time_ns(StreamId("214-1"), time, TimeDomain.DEVICE_TIME, TimeQueryOptions.CLOSEST)[0].to_numpy_array()
            im = cv2.cvtColor(cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE), cv2.COLOR_RGB2BGR)           
            x_ = 1408 - np.clip(int(gaze_xy[gaze_idx,0]), crop_size//2, 1408-crop_size//2)
            y_ = np.clip(int(gaze_xy[gaze_idx,1]), crop_size//2, 1408-crop_size//2)
            gaze_crop = im[y_-crop_size//2:y_+crop_size//2, x_-crop_size//2:x_+crop_size//2]
            snippet['rgb'] = (torch.Tensor(gaze_crop) / 255.).permute(2, 0, 1)
        test_set.append(snippet)
    data_loader = DataLoader(test_set, batch_size=50, shuffle=False)

    # dataloader and inference
    stacked_predictions = []
    for batch in data_loader:
        with torch.no_grad():
            batch = {key: value.to(device) for key, value in batch.items()}
            pred = model(batch)
        stacked_predictions.append(pred)
    stacked_predictions = torch.cat(stacked_predictions, dim=0)
    if num_classes > 2:
        pred = torch.argmax(stacked_predictions, -1).detach().cpu().numpy()
        plt.figure(figsize=(40, 4))
        plt.plot(pred, label='Predicted')
        plt.savefig('{}_{}.png'.format(name,args.vid_uid), bbox_inches='tight')
    else:
        stacked_predictions = F.softmax(stacked_predictions, -1)
        pred = stacked_predictions[:,1].detach().cpu().numpy()
        plt.figure(figsize=(40, 4))
        plt.plot(pred, label='Predicted')
        plt.savefig('{}_{}.png'.format(name,args.vid_uid), bbox_inches='tight')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_time", type=float, default=60., help='start time in the video (s), >0 to skip eye calibration')
    parser.add_argument("--snippet_gap", type=float, default=1/60, help='time gap between each snippet (s) (default: 1/60 to match 60Hz gaze)')
    parser.add_argument('--model_name', choices=['v0', 'v1_default', 'v1_15Hz', 'v1_1s', 'v1_mode', 'v1_medium', 'v1_large'], default='v1_default', help='model')
    parser.add_argument('--use_gaze', action='store_true', help='Use gaze data')
    parser.add_argument('--use_imu', action='store_true', help='Use IMU data')
    parser.add_argument('--use_rgb', action='store_true', help='Use RGB data')
    parser.add_argument("--output_save_path", type=str, default="output/")
    args = parser.parse_args()

    #first one is recorded directly in Aria, with mps output, second one is via data campaign
    roots = ["demo/"]
    vid_uids = ["907129427418886"]
    
    for i in range(len(vid_uids)):
        args.root_dir = roots[i]
        args.vid_uid = vid_uids[i]
        inference(args)