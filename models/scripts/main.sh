#!/bin/bash

#SBATCH --job-name=train_reading
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition lowpri
#SBATCH --output=slurm/train_reading-%j.log

cat /etc/hosts

TASK_NAME=binary

BATCH_SIZE=7680
LR=3e-3
EPOCH=10

GAZE_INPUT=dXYZ
IMU_INPUT=None
RGB_INPUT=None

INPUT_SEC=2
INPUT_HZ=60

VERSION=v0
VERBOSE=v0

python main.py \
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