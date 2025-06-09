import os
import json
import argparse
import lightning as L
import glob
import torch
import pandas as pd
import cv2
from tqdm import tqdm
import numpy as np
from projectaria_tools.core import data_provider
from projectaria_tools.core.sensor_data import TimeDomain
from projectaria_tools.core.stream_id import StreamId


df = pd.read_csv('ritw_filtered_eye_whisper.csv')
recordings_dir = "/source_1a/data/reading_itw_vendor/"
gaze_dir = "/source_1a/data/reading_itw_vendor/gaze"
save_path = "/source_1a/data/reading_itw_vendor/rgb_crop"
t16_df = pd.read_csv('manual_whisper_T16.csv')

task_dict = {
    1: "1660121041505660",
    2: "471713642369193",
    3: "1903003383499528",
    4: "1029167338369509",
    5: "8169720896473885",
    6: "878245494182416",
    7: "1182052209534700",
    8: "490386170604779",
    9: "840090821570517",
    10: "1103551751360064",
    11: "1693498954555370",
    12: "1362952591757706",
    13: "875225777602175",
    14: "995379855951436",
    15: "1100390031652022",
    16: "556691300286531",
    17: "548735124338854",
    18: "569543562160818",
    19: "508704305267507"
}

video_list = df['id'].tolist()[:650]#[:200] #200 425 650
print(len(video_list))
video_list = [1234125171184395] #todo remove
for video in video_list:
    video = str(video)
    idx = int(video)
    print(video,idx)
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, video), exist_ok=True)
    os.makedirs(save_path+'_lowres', exist_ok=True)
    os.makedirs(os.path.join(save_path+'_lowres', video), exist_ok=True)
    os.makedirs(save_path+'_small', exist_ok=True)
    os.makedirs(os.path.join(save_path+'_small', video), exist_ok=True)

    task =  int(df.loc[df['id'] == int(idx), 'task'].values[0])
    if task == 16: #todo change
        continue
    if os.path.exists('{}/{}.csv'.format(save_path,video)):
        continue
    task_id = task_dict[task]
    target_label = task
    recordings_dir_task = os.path.join(recordings_dir, "task_{}".format(task_id))
    vrs_path = glob.glob(os.path.join(recordings_dir_task,video,"vrs_videos", "*Scene.vrs"))[0]


    #get start/end time from gaze
    gaze_base_path = os.path.join(recordings_dir_task,video,"EyeGaze")
    if not os.path.exists(gaze_base_path):
        gaze_base_path = os.path.join(recordings_dir_task,video,"EyeGaze_0")
    #file paths
    gaze_path = os.path.join(gaze_base_path, "personalized_eye_gaze.csv") #some have no calib
    gaze_ok = os.path.exists(gaze_path)
    config_path = os.path.join(recordings_dir_task, video, "vrs_videos", "camera_config.json") #some have no calib
    config_ok = os.path.exists(config_path)
    if not gaze_ok:
        gaze_path = os.path.join(gaze_base_path, "general_eye_gaze.csv")
    gaze = pd.read_csv(gaze_path, engine='python')


    if task == 16:
        start_time = int(t16_df.loc[t16_df['id'] == int(idx), 'calib'].values[0] * 1000)
        end_time = int(t16_df.loc[t16_df['id'] == int(idx), 'end'].values[0] * 1000)
    else:
        start_time = int(df.loc[df['id'] == int(idx), 'whisper_start'].values[0] * 1000)
        end_time = int(df.loc[df['id'] == int(idx), 'whisper_end'].values[0] * 1000)


    start_time_ms = gaze['tracking_timestamp_us'].iloc[0] // 1000
    end_time_ms = gaze['tracking_timestamp_us'].iloc[-1] // 1000 

    gaze_df = gaze[(gaze['tracking_timestamp_us'] >= (start_time+start_time_ms)*1000) & (gaze['tracking_timestamp_us'] <= (end_time+start_time_ms)*1000)]
    gaze_df = gaze_df['tracking_timestamp_us'].tolist()

    #get xy position from processed gaze
    processed_gaze_path = os.path.join(gaze_dir, video+'.csv')
    processed_df = pd.read_csv(processed_gaze_path)
    x = processed_df['proj_x'].tolist()
    y = processed_df['proj_y'].tolist()

    # Starts by default options which activates all sensors
    #if len(os.listdir(os.path.join(save_path, video))) > 1000:
    #    continue

    provider = data_provider.create_vrs_data_provider(vrs_path)
    if provider is None:
        print(vrs_path, provider)
        continue
    
    deliver_option = provider.get_default_deliver_queued_options()

    deliver_option.set_truncate_first_device_time_ns(int(start_time*1e6))
    deliver_option.set_truncate_last_device_time_ns(int(1e6*(end_time_ms-start_time_ms-end_time)))

    # Only play data from RGB cameras
    deliver_option.deactivate_stream_all()
    deliver_option.activate_stream(StreamId("214-1"))
    print(video)
    #crop time first no?
    for idx, data in tqdm(enumerate(provider.deliver_queued_sensor_data(deliver_option))):
        img_timestamp = data.get_time_ns(TimeDomain.DEVICE_TIME)
        gaze_idx = np.searchsorted(gaze_df, round(data.get_time_ns(TimeDomain.DEVICE_TIME)/1e3))
        if gaze_idx < len(gaze_df):
            gaze_timestamp = gaze_df[gaze_idx]

            image_array = data.image_data_and_record()[0].to_numpy_array()
            if image_array.ndim < 3:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_BAYER_BG2BGR)
            
            im = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
            crop_size = 176*2
            
            x_ = 1408 - np.clip(int(x[gaze_idx]), crop_size//2, 1408-crop_size//2)
            y_ = np.clip(int(y[gaze_idx]), crop_size//2, 1408-crop_size//2)
            
            gaze_crop = im[y_-crop_size//2:y_+crop_size//2, x_-crop_size//2:x_+crop_size//2].astype(np.uint8) #(1/4)^2 crop

            cv2.imwrite(os.path.join(save_path, video,'{}.png'.format(gaze_idx)), gaze_crop)
            #todo remove comment
            #cv2.imwrite(os.path.join(save_path+'_small', video,'{}.png'.format(gaze_idx)), gaze_crop[int(crop_size*1/4):int(crop_size*3/4), int(crop_size*1/4):int(crop_size*3/4), :])
            #cv2.imwrite(os.path.join(save_path+'_lowres', video,'{}.png'.format(gaze_idx)), cv2.resize(gaze_crop, (crop_size//2, crop_size//2)))