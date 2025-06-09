import os
from projection_utils import project_gaze, load_camera_config_from_json
import pandas as pd
import ujson as json
import numpy as np
import re

df = pd.read_csv('ritw_filtered_eye_whisper.csv')
recordings_dir = "/source_1a/data/reading_itw_vendor/"
gaze_dir = "/source_1a/data/reading_itw_vendor/gaze"
save_path = "/source_1a/data/reading_itw_vendor/odometry"
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

video_list = df['id'].tolist()
for video in video_list:
    video = str(video)
    idx = int(video)
    task =  int(df.loc[df['id'] == int(idx), 'task'].values[0])
    if task == 16:
        print(task)
    if os.path.exists('{}/{}.csv'.format(save_path,video)):
        continue
    task_id = task_dict[task]
    target_label = task
    recordings_dir_task = os.path.join(recordings_dir, "task_{}".format(task_id))
    odometry_path = os.path.join(recordings_dir_task,video,"SingleSequenceTrajectory", "open_loop_trajectory.csv")
    if not os.path.exists(odometry_path):
        print(odometry_path)
        continue
    odometry = pd.read_csv(odometry_path, engine='python')


    #gaze
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
    odometry_df = odometry[(odometry['tracking_timestamp_us'] >= (start_time+start_time_ms)*1000) & (odometry['tracking_timestamp_us'] <= (end_time+start_time_ms)*1000)]
    odometry_df = odometry[["tx_odometry_device", "ty_odometry_device", "tz_odometry_device", "qx_odometry_device", "qy_odometry_device", "qz_odometry_device", "qw_odometry_device", "device_linear_velocity_x_odometry", "device_linear_velocity_y_odometry", "device_linear_velocity_z_odometry", "angular_velocity_x_device", "angular_velocity_y_device", "angular_velocity_z_device"]]
    indices = np.linspace(0, len(odometry_df) - 1, len(gaze_df), dtype=int)
    odometry_df = odometry_df.iloc[indices]
    odometry_df.to_csv('{}/{}.csv'.format(save_path,video), index=False)


