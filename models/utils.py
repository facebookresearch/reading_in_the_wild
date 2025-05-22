import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import io
import cv2
import torch
def get_img_from_fig(fig, dpi=75):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def draw_projections_on_image(
    image_array: np.ndarray,
    x,y,
    gaze_rad: int = 10,
    skip = False
):
    CMAP = plt.get_cmap('Spectral_r')
    image = Image.fromarray(image_array)
    x = 1408 - x
    if not skip:
        draw = ImageDraw.Draw(image)
        #projections to draw is just a list of xy
        projections_to_draw = np.stack([x,y],-1) #N, 2    
        num_points = len(projections_to_draw)
        
        for idx, proj in enumerate(projections_to_draw):
            if idx>0:
                draw.line([prev_proj[0], prev_proj[1], proj[0], proj[1]])
                
            rgba = CMAP(idx / num_points)
            color = tuple((np.array(rgba[:3]) * 255).astype(int))

            draw.ellipse(
                (proj[0] - gaze_rad, proj[1] - gaze_rad, proj[0] + gaze_rad, proj[1] + gaze_rad),
                fill=color,
                outline=color,
            )        
            prev_proj = proj

    #image = image.rotate(-90)

    return np.array(image).astype(np.uint8)

def create_sampled_array(df, num_samples, stride):
    data = df.values
    sampled_arrays = []
    max_start_index = data.shape[0] - (num_samples - 1) * stride

    for start_index in range(max_start_index):
        indices = range(start_index, start_index + stride * num_samples, stride)
        if indices[-1] < data.shape[0]:
            sampled_arrays.append(data[indices])
    
    result_array = np.array(sampled_arrays)
    return result_array

def get_labels(labels, task, length, gaia_id):
    if type(labels) == list:
        #in normal cases
        label = labels[task-1]
        if label == -1:
            return -1
        elif task != 16:
            return [label] * length
        else: #task 16 AND not marked with -1
            return get_t16_labels(length, gaia_id)
    elif label == 'fatigue':        
        #first 1/4 as 0, last 1/4 as 1
        if task in [1,3,4,5,6,12,14,18,19]:
            return [0] * length//4 + [-1] * (length//2) + [1] * (length - length//4 - length//2)
        else:
            return -1
    

def get_t16_labels(length, gaia_id):
    T16_path = "/work_1a/charig/reading_itw/dataset/manual_whisper_T16.csv"
    df = pd.read_csv(T16_path)    
    t16_row = df.loc[df['id'] == gaia_id].to_dict()
    row_label = [0] * length
    calib = next(iter(t16_row['calib'].values()))       
    # Calculate end-calib
    end_calib = next(iter(t16_row['end'].values())) - calib
    
    # Iterate over the start and finish columns
    for i in range(1, 11):  # Assuming there are up to 10 start/finish pairs
        start_col = f'start {i}'
        finish_col = f'finish {i}'
        if start_col in t16_row and finish_col in t16_row:
            start = next(iter(t16_row[start_col].values()))
            finish = next(iter(t16_row[finish_col].values()))
            # Check if the start and finish values are not NaN
            if pd.notna(start) and pd.notna(finish):
                start_index = int((start - calib) * length / end_calib)
                finish_index = int((finish - calib) * length / end_calib)
                row_label[start_index:finish_index] = [1] * (finish_index - start_index)
    return row_label


def modality_dropout(snippet, use_rgb, use_imu, use_gaze):
    active_modalities = []
    if use_rgb:
        active_modalities.append('rgb')
    if use_imu:
        active_modalities.append('odom')
    if use_gaze:
        active_modalities.append('gaze')
    num_modalities = len(active_modalities)
    num_to_drop = random.sample(list(range(num_modalities)), 1)[0]
    modalities_to_drop = random.sample(active_modalities, num_to_drop)
    # Drop the selected modalities
    for modality in modalities_to_drop:
        snippet[modality] *= 0
    return snippet

def gaze_rotate(snippet, use_rgb, use_imu, use_gaze):
    if use_gaze:
        gaze = snippet['gaze']
        snippet['gaze'] = torch.stack([-gaze[:,1], gaze[:,0], gaze[:,2]],1)
    if use_imu:
        odom = snippet['odom']
        snippet['odom'] = torch.stack([-odom[:,1], odom[:,0], odom[:,2],-odom[:,4], odom[:,3], odom[:,5]],1)
    if use_rgb:
        rgb = snippet['rgb']
        snippet['rgb'] = rgb.transpose(1, 2).flip(2)
    return snippet
