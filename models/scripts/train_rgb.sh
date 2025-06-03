#!/bin/bash

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

#SBATCH --job-name=train_reading
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition lowpri
#SBATCH --output=slurm/train_reading-%j.log
#SBATCH --mem=100G

cat /etc/hosts

TASK_NAME=binary

BATCH_SIZE=512
LR=1e-3
EPOCH=5

GAZE_INPUT=dXYZ
IMU_INPUT=vw
RGB_INPUT=small64

INPUT_SEC=2
INPUT_HZ=60

VERSION=v1
VERBOSE=v1

python train.py \
    --task_name $TASK_NAME \
    --gaze_input $GAZE_INPUT \
    --imu_input $IMU_INPUT \
    --rgb_input $RGB_INPUT \
    --input_hz $INPUT_HZ \
    --input_sec $INPUT_SEC \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --epoch $EPOCH \
    --slurm_id $SLURM_JOB_ID \
    --verbose $VERBOSE \
    --version $VERSION \
#   # 