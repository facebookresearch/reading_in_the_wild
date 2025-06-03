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
import json
import torch
import wandb
import argparse
import lightning as L
from data import load_data
from model import MultimodalTransformer
from trainer import ReadingClassifier
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint


def train(args):
    # Args
    task_name = args.task_name
    max_epochs = args.epoch
    lr = args.lr
    batch_size = args.batch_size
    input_sec = args.input_sec
    input_hz = args.input_hz
    gaze_input = args.gaze_input
    imu_input = args.imu_input
    rgb_input = args.rgb_input
    slurm_id =  args.slurm_id
    input_length = input_sec * input_hz
    print(json.dumps(vars(args), indent=4))
    default_root_dir="/work_1a/charig/reading/models/"

    # Logging
    verbose = args.verbose
    version = args.version
    run_name = f"{slurm_id}_g{gaze_input}_i{imu_input}_r{rgb_input}_f{input_hz}_s{input_sec}_{verbose}"
    project_name = f"{task_name}_{version}"
    wandb_logger = WandbLogger(log_model="all", project=project_name, name=run_name, config=args)

    # Dataset
    train_set, val_set, test_set, num_classes, input_dim, loss_weight = load_data(task_name=task_name,input_sec=input_sec,input_hz=input_hz,gaze_input=gaze_input,imu_input=imu_input,rgb_input=rgb_input)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=16)
    val_loader = DataLoader(val_set, batch_size=max(1, batch_size//2), shuffle=False, num_workers=16)
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", every_n_epochs=1, save_top_k=1, dirpath=os.path.join(default_root_dir,run_name))
    print(len(train_set), len(val_set), len(test_set))

    # Train
    model = MultimodalTransformer(num_classes=num_classes, dim_feat=32, input_dim=input_dim, sequence_length=input_length, dropout = 0.3)
    reading_classifier = ReadingClassifier(model, num_classes=num_classes, lr=lr, use_loss_weight=True, loss_weight=loss_weight)
    trainer = L.Trainer(max_epochs=max_epochs, logger=wandb_logger, default_root_dir=default_root_dir,
                        check_val_every_n_epoch=1, accelerator="gpu", devices=1, log_every_n_steps=50, callbacks=[checkpoint_callback])
    trainer.fit(model=reading_classifier, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Test
    best_model_path = checkpoint_callback.best_model_path
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
    trainer.test(model=reading_classifier, ckpt_path=best_model_path, dataloaders=test_loader)
    wandb.finish()
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="binary")
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128*60)
    parser.add_argument("--lr", type=float, default="3e-3")
    parser.add_argument("--input_sec", type=int, default=2)
    parser.add_argument("--input_hz", type=int, default=60)
    parser.add_argument("--gaze_input", type=str, default="dXYZ") #XYZ is 3D projection, xy is 2D
    parser.add_argument("--imu_input", type=str, default="vw") #t=translation, q=quarternion (rotation), v=velocity, w=angular velocity
    parser.add_argument("--rgb_input", type=str, default="None") #crop, small, lowres
    parser.add_argument("--slurm_id", type=str, default="")
    parser.add_argument("--version", type=str, default="multi")
    parser.add_argument("--verbose", type=str, default="")
    args = parser.parse_args()
    train(args)
