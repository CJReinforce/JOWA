# Scaling Offline Model-Based RL via Jointly-Optimized World-Action Model Pretraining

Welcome to the official repository for our paper: "Scaling Offline Model-Based RL via Jointly-Optimized World-Action Model Pretraining".

**TL;DR:** A single JOWA-150M agent masters 15 Atari games at 84.7% human-level and 119.5% DQN-level, and can adapt to novel games with ~4 expert demos.

We are still updating and optimizing the code. Stay tuned for updates and additional resources!

## Requirements
### Installation

- python 3.8: `conda create -n jowa python=3.8 && conda activate jowa`
- Install basic dependencies for compatibility: `pip install setuptools==65.5.0 wheel==0.38.4 packaging`
- Install [other dependencies](requirements.txt): `pip install -r requirements.txt`
- (Optional but recommended) Install Flash-Attention: `pip install flash-attn --no-build-isolation`
- If evaluating, install jq to parse JSON files: `apt install jq`
- If training, set the wandb account: `wandb login`, `<API keys>`

The model weights of 3 JOWA variants are already in [this repo](checkpoints/JOWA).

## Eval

Set hyperparameters of evaluation in `eval/eval.sh`, such as `model_name`, `game`, `num_rollouts`, and `num_gpus`. Then run this script:

```bash
bash eval/eval.sh
```

The above command will run the setting number of rollouts (episodes) in parallel. After completing all evaluation processes, get the aggregate scores using the following command:

```python
python eval/aggregate_scores.py --name JOWA_150M --env BeamRider
```

Reproduce results of all JOWA variants and baselines through the following command:

```python
python results/results.py
```

> [!NOTE]
> We found that some hyperparameters in a few groups of previous evaluation experiments (shown in the early version of paper) were set incorrectly. After the correction (already done in the current version of code), all JOWA variants achieved higher IQM HNS, and the scaling trend still holds. We will update the paper to show the corrected results.

### Dataset

- Download original DQN-Replay dataset: `python src/datasets/download.py`
- Save filterd trajectory-level dataset in csv format: `python src/datasets/filter_in_csv.py`
- Save filtered trajectory-level dataset in structured dir (obs in .png format, others in .npy for each trajectory): `python src/datasets/save.py`
- Save filtered segment-level dataset in csv format: `python src/datasets/save_segment_in_csv.py`

> [!NOTE]
> We will optimize the data preprocessing code to make it more simple.

## Pretraining

```python
python src/pretrain.py hydra/job_logging=disabled hydra/hydra_logging=disabled
```

## Fine-tuning

```python
python src/fine_tune.py hydra/job_logging=disabled hydra/hydra_logging=disabled
```

> [!NOTE]
> The pre-training and fine-tuning codes are still being organized, due to lots of variable renaming.

## TODO

- \[x\] Release model weights.
- \[x\] Optimize evaluation codes.
- \[ \] Merge and optimize pretraining & fine-tuning codes.