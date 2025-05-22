#!/bin/bash

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