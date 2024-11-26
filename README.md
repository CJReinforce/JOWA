<div align="center">
   <img src="static/img/logo_2.png" alt="Logo" width="40%" max-width=512px>
</div>

---

Welcome to the official repository for our paper: "Scaling Offline Model-Based RL via Jointly-Optimized World-Action Model Pretraining".

**TL;DR:** A single JOWA-150M agent masters 15 Atari games at 84.7% human-level and 119.5% DQN-level, and can adapt to novel games with ~4 expert demos.

<div class="gallery">
    <div class="row">
      <img src="https://raw.githubusercontent.com/CJReinforce/JOWA_agents/refs/heads/master/static/images/Assault.gif" alt="Assault" width=16%>
      <img src="https://raw.githubusercontent.com/CJReinforce/JOWA_agents/refs/heads/master/static/images/Atlantis.gif" alt="Atlantis" width=19%>
      <img src="https://raw.githubusercontent.com/CJReinforce/JOWA_agents/refs/heads/master/static/images/NameThisGame.gif" alt="NameThisGame" width=19%>
      <img src="https://raw.githubusercontent.com/CJReinforce/JOWA_agents/refs/heads/master/static/images/ChopperCommand.gif" alt="ChopperCommand" width=19%>
      <img src="https://raw.githubusercontent.com/CJReinforce/JOWA_agents/refs/heads/master/static/images/DemonAttack.gif" alt="DemonAttack" width=19%>
    </div>
    <div class="row">
      <img src="https://raw.githubusercontent.com/CJReinforce/JOWA_agents/refs/heads/master/static/images/Carnival.gif" alt="Carnival" width=16%>
      <img src="https://raw.githubusercontent.com/CJReinforce/JOWA_agents/refs/heads/master/static/images/Seaquest.gif" alt="Seaquest" width=19%>
      <img src="https://raw.githubusercontent.com/CJReinforce/JOWA_agents/refs/heads/master/static/images/SpaceInvaders.gif" alt="SpaceInvaders" width=19%>
      <img src="https://raw.githubusercontent.com/CJReinforce/JOWA_agents/refs/heads/master/static/images/TimePilot.gif" alt="TimePilot" width=19%>
      <img src="https://raw.githubusercontent.com/CJReinforce/JOWA_agents/refs/heads/master/static/images/Zaxxon.gif" alt="Zaxxon" width=19%>
    </div>
    <p class="note">
      *Training used 84x84 grayscale images. RGB demos shown here for better visualization. For Atlantis (second in the first row),
        only the first 7 minutes of the 2-hour gameplay are displayed. More demos are <a href=static/mp4>here</a>.
    </p>
</div>
  
🚧 ***We are still updating and optimizing the code. Stay tuned for updates and additional resources!***

## 🚀 Installation

- python 3.8: `conda create -n jowa python=3.8 && conda activate jowa`
- Install basic dependencies for compatibility: `pip install setuptools==65.5.0 wheel==0.38.4 packaging pip==24.0`
- Install [other dependencies](requirements.txt): `pip install -r requirements.txt`
- (Optional but recommended) Install Flash-Attention: `pip install flash-attn --no-build-isolation`
- If evaluating, install jq to parse JSON files: `apt install jq`
- If training, set the wandb account: `wandb login`, `<API keys>`

## 📊 Eval

Download the model weights from [here](checkpoints/JOWA) or [google drive](https://drive.google.com/drive/folders/1YHaCemhobchJWE5zt28TUrKp2tp8yysF?usp=sharing).

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

## 🔧 Training
### Dataset for pretraining

1. Download raw dataset

```bash
python src/datasets/download.py
```

This will enable multi-process to download the original DQN-Replay dataset (~1TB for the default 20 games and 3.1TB for all 60 games).

2. Process and downsample data

```bash
python src/datasets/downsample.py
```

This command processes the raw data into two formats: **(i) Trajectory-level dataset:** A hdf5 file containing transitions for each trajectory. Total size: ~1TB for 15 pretraining games. **(ii) Segment-level dataset:** CSV files containing segment indices in the correspoding trajectory.

### Pre-training

Start pre-training (including stage 1 and 2) with the following `torchrun` command:

```python
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=39500 src/train.py hydra/job_logging=disabled hydra/hydra_logging=disabled
```

However, after stage 1, we highly recommend the following steps to speed up training: 

1. Use the pre-trained vqvae to encode all images in advance: `python src/save_obs_in_token.py` file.
2. Use the `AtariTrajWithObsTokenInMemory` class as the dataset (uncomment lines 161~170 in the `src/train.py` file) to speed up loading, since vqvae is frozen in the second stage.
3. Modify the config `config/train_xxM.yaml`. Set the `initialization.load_tokenizer` to the path of stage-1's checkpoint. Then set `training.action.train_critic_after_n_steps: -1` to train the critic immediately.
4. Re-run the above `torchrun` command.

### Fine-tuning

The fine-tuning dataset is [here](fine_tuning_dataset.zip). Please unzip this compressed file. Then set the id of a single GPU like: `export DEVICE_ID=0` and run the command:

```python
torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=39500 src/fine_tune.py hydra/job_logging=disabled hydra/hydra_logging=disabled
```

After fine-tuning, modify and run `eval/eval_wo_planning.sh` to evaluate the checkpoint.

## 📝 TODO

- \[x\] Release model weights.
- \[x\] Optimize evaluation code.
- \[x\] Optimize data preprocessing.
- \[x\] Merge and optimize pretraining & fine-tuning codes.
- \[ \] Multi-env evaluation.