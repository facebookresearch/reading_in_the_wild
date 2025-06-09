import os
from projection_utils import project_gaze, load_camera_config_from_json
import pandas as pd
import ujson as json
import numpy as np
import re

df = pd.read_csv('ritw_filtered_eye_whisper.csv')
t16_df = pd.read_csv('manual_whisper_T16.csv')

recordings_dir = "/source_1a/data/reading_itw_vendor/"
save_path = "/source_1a/data/reading_itw_vendor/gaze"

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
    #if task == 16:
    #    continue
    task_id = task_dict[task]
    target_label = task
    recordings_dir_task = os.path.join(recordings_dir, "task_{}".format(task_id))

    if os.path.exists('{}/{}.csv'.format(save_path,video)):
        continue

    print(video)
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

    if config_ok:
        try:
            camera_config = load_camera_config_from_json(config_path)
        except Exception:
            print('cannot read', config_path)
            config_ok = False

    try:
        gaze = pd.read_csv(gaze_path, engine='python')
    except Exception:
        print('cannot read', gaze_path)
        continue
    start_time_ms = gaze['tracking_timestamp_us'].iloc[0] // 1000
    end_time_ms = gaze['tracking_timestamp_us'].iloc[-1] // 1000 
    
    if task == 16:
        start_time = int(t16_df.loc[t16_df['id'] == int(idx), 'calib'].values[0] * 1000)
        end_time = int(t16_df.loc[t16_df['id'] == int(idx), 'end'].values[0] * 1000)
    else:
        start_time = int(df.loc[df['id'] == int(idx), 'whisper_start'].values[0] * 1000)
        end_time = int(df.loc[df['id'] == int(idx), 'whisper_end'].values[0] * 1000)
    

    #GAZE 2
    personalized_df = gaze[(gaze['tracking_timestamp_us'] >= (start_time+start_time_ms)*1000) & (gaze['tracking_timestamp_us'] <= (end_time+start_time_ms)*1000)]

    #GAZE - preprocessing
    processed_gaze_df = project_gaze(personalized_df,config_path=config_path).ffill().bfill()
    proj_x = processed_gaze_df.loc[:,"projected_point_2d_x"].to_numpy()
    proj_y = processed_gaze_df.loc[:,"projected_point_2d_y"].to_numpy()
    transformed_x = processed_gaze_df.loc[:,"transformed_gaze_x"].to_numpy()
    transformed_y = processed_gaze_df.loc[:,"transformed_gaze_y"].to_numpy()
    transformed_z = processed_gaze_df.loc[:,"transformed_gaze_z"].to_numpy()
    depth = processed_gaze_df.loc[:,"depth_m"].to_numpy()

    if np.isnan(proj_x).any():
        print(proj_x)
        print(np.isnan(proj_x).sum())
        continue
    
    #helps to retain some data for easy ablation
    all_data = pd.DataFrame({'time': (personalized_df["tracking_timestamp_us"].to_numpy() - personalized_df['tracking_timestamp_us'].iloc[0])/1e6,
                    'proj_x': proj_x,
                    'proj_y': proj_y,
                    'transformed_x': transformed_x,
                    'transformed_y': transformed_y,
                    'transformed_z': transformed_z,
                    'depth': depth,
                    })

    all_data.to_csv('{}/{}.csv'.format(save_path,video), index=False)
    #general only another one

    #print(all_data)

raise Exception
pd.set_option('future.no_silent_downcasting', True)
pd.options.mode.copy_on_write = True
recordings_dir = "/source_1a/data/reading_itw/recordings_v2_1/"
whisper_dir = "/source_1a/data/reading_itw/asr_out/asr_out_only/" #CHANGE THIS BEFORE RUNNING TOMORROW
task_ids = ["864060922076548", "1121648862210316", "918966193292843", "384817364318975", "3733689280196576", "475425891730904", "1005147787493464"]
neg_task_ids = ["384817364318975", "3733689280196576", "475425891730904"]
start_words =  ['start', 'i\’m', 'begin', 'started', 'star', 'stuart', 'starting']
end_words =  ['finish', 'finished', 'stop', 'end', 'done', 'stopped', 'complete']
save_path = "/work_1a/charig/data/reading_itw_campaign/"

T_end_buffer = 20000
T_duration = 60000
    
os.makedirs(save_path, exist_ok=True)
for task_id in task_ids:
    target_label = 0 if task_id in neg_task_ids else 1
    out = {}
    out2 = []
    recordings_dir_task = os.path.join(recordings_dir, "task_{}".format(task_id))
    task_video_list = os.listdir(recordings_dir_task)
    for video in task_video_list:
        #file paths
        gaze_path = os.path.join(recordings_dir_task, video, "EyeGaze", "personalized_eye_gaze.csv") #some have no calib
        config_path = os.path.join(recordings_dir_task, video, "vrs_videos", "camera_config.json") #some have no calib
        whisper_path = os.path.join(whisper_dir, "gaia:{}".format(video), "asr", "speech.csv")

        #check for existence
        whisper_ok = os.path.exists(whisper_path)
        gaze_ok = os.path.exists(gaze_path)
        config_ok = os.path.exists(config_path) #set to 0 for debug -- json load takes a long time

        if config_ok:
            try:
                camera_config = load_camera_config_from_json(config_path)
            except Exception:
                print('cannot read', config_path)
                config_ok = False
        
        if whisper_ok:
            try:
                whisper = pd.read_csv(whisper_path)
            except Exception:
                print('cannot read', whisper_path)
                whisper_ok = False

        ###GAZE
        if not gaze_ok: #if no calibrated gaze, use normal one
            gaze_path = os.path.join(recordings_dir_task, video, "EyeGaze", "general_eye_gaze.csv") #some have no calib
            gaze_ok = os.path.exists(gaze_path)
        if not gaze_ok: #if no normal gaze, then skip the video
            continue
        try:
            gaze = pd.read_csv(gaze_path, engine='python')
        except Exception:
            print('cannot read', gaze_path)
            continue
        start_time_ms = gaze['tracking_timestamp_us'].iloc[0] // 1000
        end_time_ms = gaze['tracking_timestamp_us'].iloc[-1] // 1000 
        duration = (end_time_ms - start_time_ms) 

        if not config_ok:
            config_path = None

        ###ASR
        if whisper_ok:
            whisper = pd.read_csv(whisper_path)
            #this filters out reading and the row before -- to prevent random start/stop from background noise
            condition = whisper['word'].str.contains('eading')
            shifted_condition = condition.shift(-1)
            filtered_whisper = whisper[shifted_condition.fillna(False)]
            start_whisper = filtered_whisper[filtered_whisper['word'].str.contains('|'.join(start_words),flags=re.I)]
            end_whisper = filtered_whisper[filtered_whisper['word'].str.contains('|'.join(end_words),flags=re.I)]
            
            start_detected = start_whisper.shape[0] > 0
            end_detected = end_whisper.shape[0] > 0
        else:
            start_detected = False
            end_detected = False
        
        if target_label == 0: T_duration = 120000
        if start_detected and end_detected:
            end_time = end_whisper['startTime_ms'].iloc[0] - 1000
            start_time = start_whisper['endTime_ms'].iloc[-1] + 1000
        elif start_detected and not end_detected:
            end_time = duration - T_end_buffer
            start_time = start_whisper['endTime_ms'].iloc[-1] - 1000
        elif end_detected and not start_detected:
            end_time = end_whisper['startTime_ms'].iloc[0] + 1000
            start_time = max(120000, end_time-T_duration)
        else:
            end_time = duration - T_end_buffer
            start_time = max(120000, end_time-T_duration)

        #for daily acitivites, just use entire video, including eye calibration, as not reading
        if task_id == "475425891730904":
            start_time = 0
            end_time = duration

        vid_length = end_time - start_time
        if vid_length < 30000:
            continue
        if vid_length > 120000:
            start_time = end_time - 120000
        

        #GAZE 2
        personalized_df = gaze[(gaze['tracking_timestamp_us'] >= (start_time+start_time_ms)*1000) & (gaze['tracking_timestamp_us'] <= (end_time+start_time_ms)*1000)]

        #GAZE - preprocessing
        processed_gaze_df = project_gaze(personalized_df,config_path=config_path).ffill()
        proj_x = processed_gaze_df.loc[:,"projected_point_2d_x"].to_numpy()
        proj_y = processed_gaze_df.loc[:,"projected_point_2d_y"].to_numpy()
        transformed_x = processed_gaze_df.loc[:,"transformed_gaze_x"].to_numpy()
        transformed_y = processed_gaze_df.loc[:,"transformed_gaze_y"].to_numpy()
        transformed_z = processed_gaze_df.loc[:,"transformed_gaze_z"].to_numpy()
        depth = processed_gaze_df.loc[:,"depth_m"].to_numpy()

        if np.isnan(proj_x).any():
            print(np.isnan(proj_x).sum())
            continue
        
        #helps to retain some data for easy ablation
        all_data = pd.DataFrame({'time': (personalized_df["tracking_timestamp_us"].to_numpy() - personalized_df['tracking_timestamp_us'].iloc[0])/1e6,
                        'proj_x': proj_x,
                        'proj_y': proj_y,
                        'transformed_x': transformed_x,
                        'transformed_y': transformed_y,
                        'transformed_z': transformed_z,
                        'depth': depth,
                        })
        
        #filter out time (again)
        all_data = all_data[all_data['time'] >= 0]

        snippet_dur = 0.5
        et_freq = 60
        samples_per_snippet = int(snippet_dur * et_freq)
        num_snippets = int(len(all_data.index) // samples_per_snippet)

        snippets = []
        for snippet_idx in range(num_snippets):
            snippet = all_data[snippet_idx * samples_per_snippet : (snippet_idx + 1) * samples_per_snippet]
            if target_label >= 0:
                #can also add category, maybe later
                gt = target_label                    
            type = None
            if isinstance(gt, tuple):
                gt, type = gt
            if gt >= 0:
                snippet = snippet.to_dict('list')
                snippet["gt"] = gt
                snippet["idx"] = snippet_idx
                del snippet["time"]
                if type:
                    snippet["type"] = type
                snippets.append(snippet)
        #for every video
        out[video] = snippets
        out2.append(vid_length)
    #for every task
    print(len(out2), sum(out2))
    json.dump(out, open(os.path.join(save_path, task_id + '_v3.json'), 'w'))
