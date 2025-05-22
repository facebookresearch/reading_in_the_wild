# Reading Recognition in the Wild

This project focuses on reading recognition in diverse environments.

## Prerequisites

To get started, ensure you have the following dependencies installed:

- `pytorch`
- `pytorch-lightning`
- `einops`
- `pandas`

## Demo

[Note: the `demo` folder only include gaze (no IMU and RGB), to save storage. Please download the full dataset to use all of them, or use `main.py`.]

```bash
python predict.py --use_gaze
```

## Inference

To perform inference, refer to the predict folder for the necessary scripts and instructions.

## Training

To train the model, you can use the following commands:

For non-RGB data:

```sbatch scripts/main.py```

For RGB data:

```sbatch scripts/main_rgb.py```

We allow for different input representations:
- **Gaze**: `XYZ` (3D gaze point) `yp` (yaw and pitch angles), `xy` (2D projected gaze), `z` depth, add `d` to differentiate and use differences. Default: `dXYZ`
- **RGB**: default to 64px (5 deg FoV), adjustable
- **Head pose**: `t` (translation), `q` (angular translation), `v` (linear velocity), `w` (angular velocity). Default: `vw`

## Predict

[Note: the `demo` folder only include gaze (no IMU and RGB), to save storage. Please download the full dataset to use all of them, or use `main.py`.]

### Arguments

- **Start Time**: Specify the start time in the video (in seconds), for example, to skip eye calibration.
- **Snippet Gap**: Define the time gap between each snippet to do inference (in seconds). The default is set to `1/60` to match a 60Hz gaze frequency.
- **Modality Options**: Choose to use gaze, IMU, and/or RGB data. The model is flexible and can work with any combinations of these modalities.

### Usage

To run the model with all modalities, performing inference at every second, use the following command:

```bash
python predict.py --use_rgb --use_imu --use_gaze --start_time 0 --snippet_gap 1
```

To run the model with gaze only, performing inference at 60Hz, starting from T=60s, use the following command:
```bash
python predict.py --use_gaze --start_time 60 --snippet_gap 0.0166666667
```

### Model variations

The argument `--model_name` has the following components:

- **Choices**: 
  - The `choices` parameter restricts the possible values that `--model_name` can take. The available options are:
    - `v0` -- the same model as last time
    - `v1_default` -- slightly updated model, should handle vertical texts better
    - `v1_15Hz` -- with lower frequency (for ablation, might see more attention towards RGB)
    - `v1_1s` -- with lower temporal context, should have better latency
    - `v1_mode` -- reading mode [not reading, out loud, normal, scan, walking , write/type, skim]
    - `v1_medium` -- reading medium [not reading, print, digital, object] (note, we do not use RGB for this option)
    - `v1_large` -- with larger RGB crop
      
- **Default Value**: 
  - The `default` parameter sets the default value for `--model_name` if the user does not provide one. In this case, the default is `v1_default` for main evaluation.
