# Scaling Offline Model-Based RL via Jointly-Optimized World-Action Model Pretraining

Welcome to the official repository for our paper: "Scaling Offline Model-Based RL via Jointly-Optimized World-Action Model Pretraining"

## Coming Soon!

We are in the process of preparing the code and models for release. Stay tuned for updates and additional resources!

## Requirements
### Installation

- python 3.8: `conda create -n jowa python=3.8`
- `pip install setuptools==65.5.0 wheel==0.38.4 packaging`
- Install [other dependencies](requirements.txt): `pip install -r requirements.txt`
- Install Flash-Attention: `pip install flash-attn --no-build-isolation`
- Wandb account: `wandb login`, `<API keys>`

### Dataset

- Download original DQN-Replay dataset: `python src/datasets/download.py`
- Save filterd trajectory-level dataset in csv format: `python src/datasets/filter_in_csv.py`
- Save filtered trajectory-level dataset in structured dir (obs in .png format, others in .npy for each trajectory): `python src/datasets/save.py`
- Save filtered segment-level dataset in csv format: `python src/datasets/save_segment_in_csv.py`

## Training

`python src/pretrain.py hydra/job_logging=disabled hydra/hydra_logging=disabled`
