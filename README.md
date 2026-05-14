# SimLingo + Fail2Drive: Long-Tail Scenario Evaluation & Data Collection

> This repository extends [SimLingo (CVPR'25 Highlight)](https://github.com/RenzKa/simlingo) with [Fail2Drive](https://github.com/autonomousvision/fail2drive) support, enabling evaluation and PDM-lite data collection on long-tail unseen scenarios.

---

# [CVPR'25, Highlight] SimLingo: Vision-Only Closed-Loop Autonomous Driving with Language-Action Alignment

<p align="center">
  <h3 align="center">
    <a href="https://arxiv.org/abs/2503.09594"> Paper</a> | <a href="https://www.youtube.com/watch?v=Mpbnz2AKaNA&t=15s">Video</a> | <a href="https://www.katrinrenz.de/simlingo/">Website</a> | <a href="https://huggingface.co/datasets/RenzKa/simlingo">Dataset</a> | <a href="https://huggingface.co/RenzKa/simlingo">Model</a> 
  </h3>
</p>

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/carllava-vision-language-models-for-camera/carla-leaderboard-2-0-on-carla)](https://paperswithcode.com/sota/carla-leaderboard-2-0-on-carla?p=carllava-vision-language-models-for-camera)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/carllava-vision-language-models-for-camera/bench2drive-on-bench2drive)](https://paperswithcode.com/sota/bench2drive-on-bench2drive?p=carllava-vision-language-models-for-camera)

<p align="center" style="font-size:17px;">
SimLingo is a Vision-Language-Action (VLA) model that achieves state-of-the-art driving performance on the CARLA Leaderboard and Bench2Drive, while simultaniously including language capabilities like VQA, commentary, and instruction following.
</p>

<p align="center">
  <img src="assets/simlingo_teaser.png">
</p>



This repository is based on [Carla Garage](https://github.com/autonomousvision/carla_garage) and includes the PDM-lite expert, data collection code, language label generation, dreaming data generation, training of the base and final model, and evaluation of closed-loop driving and the language capabilities.

## 6-minute summary <a name="summary"></a> 

[<img src="assets/thumbnail.png" width="100%">](https://youtu.be/Mpbnz2AKaNA?si=qdQfhGIwnCbtD2DQ)


## News <a name="news"></a>
- **`[2025/06/25]`** We released the simlingo model checkpoints and inference code.
- **`[2025/05/26]`** We released the full dataset on huggingface.
- **`[2025/05/08]`** Initial code release.
- **`[2025/04/28]`** SimLingo is accepted to CVPR as a highlight paper.


## Contents
1. [Setup](#setup)
   - [simlingo_f2d environment (Fail2Drive + SimLingo)](#simlingo_f2d-environment)
2. [Repository structure](#repository-structure)
3. [Dataset download](#dataset-download)
4. [Data Generation](#data-generation)
    - [Driving Data](#driving-data)
    - [Language Data](#language-data)
    - [Dreamer Data](#dreamer-data)
5. [Training](#training)
6. [Evaluation](#evaluation)
    - [Closed-loop driving/ Bench2Drive](#bench2drive)
    - [Language eval](#language-eval)
7. [Fail2Drive Integration](#fail2drive-integration)
    - [Creating New Scenes](#creating-new-scenes)
    - [Evaluating SimLingo on Fail2Drive Scenes](#evaluating-simlingo-on-fail2drive-scenes)
    - [Collecting Training Data on Fail2Drive Scenes](#collecting-training-data-on-fail2drive-scenes)
8. [Citations](#citations)
   
   
## Setup

Clone the repository, setup CARLA 0.9.15, and build the conda environment:
```Shell
git clone git@github.com:RenzKa/simlingo.git
cd simlingo
chmod +x setup_carla.sh
./setup_carla.sh

# Create base environment
conda env create -f environment.yaml
conda activate simlingo

# Install PyTorch separately to ensure correct CUDA version
pip install torch==2.2.0

# Install flash-attn separately
pip install flash-attn==2.7.0.post2
```

Before running the code, you will need to add the following paths to PYTHONPATH on your system:
```Shell
export CARLA_ROOT=/path/to/CARLA/root
export WORK_DIR=/path/to/simlingo
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}
```

### simlingo_f2d environment

This fork adds a second conda environment `simlingo_f2d` that supports both SimLingo and Fail2Drive.
It uses **Python 3.10** and the **CARLA 0.9.15 Python 3.10 wheel** from Fail2Drive.

#### Prerequisites
- CARLA 0.9.15 installation with the Python 3.10 wheel
  (the Fail2Drive distribution ships `CarlaUE4/PythonAPI/carla/dist/carla-0.9.15-cp310-cp310-linux_x86_64.whl`)
- conda / mamba

#### Step 1 — Clone this repo
```bash
git clone git@github.com:zhumorui/simlingo_f2d.git
cd simlingo_f2d
```

#### Step 2 — Symlink CARLA
Point `f2d_carla` inside the repo root at your CARLA 0.9.15 installation:
```bash
# Replace the path below with where your CARLA 0.9.15 is installed
ln -s /path/to/CarlaUE4_0.9.15  f2d_carla
```

#### Step 3 — Create the conda environment
```bash
conda env create -f environment_f2d.yaml
conda activate simlingo_f2d
```
This installs Python 3.10 and all packages from `team_code/requirements.txt`.

#### Step 4 — Install CARLA Python API (Python 3.10 wheel)
```bash
pip install f2d_carla/PythonAPI/carla/dist/carla-0.9.15-cp310-cp310-linux_x86_64.whl
```

#### Step 5 — Install flash-attn
```bash
pip install flash-attn
```

#### Step 6 — Configure environment paths
Run the setup script once to bake the paths into the conda env:
```bash
source env_vars.sh
```
After this step, activating `simlingo_f2d` automatically sets all required `PYTHONPATH` / `CARLA_ROOT` variables.

#### Verification
```bash
conda activate simlingo_f2d
python -c "import carla; print('CARLA OK:', carla.__version__)"
python -c "import torch; print('PyTorch OK:', torch.__version__)"
```



## Repository structure
The main structure of this repository is taken from [Carla Garage](https://github.com/autonomousvision/carla_garage). Please check it out for more detailed information.

**CARLA**: We have the `leaderboard_autopilot` and `scenario_runner_autopilot` folders for running data collection. The `leaderboard` and `scenario_runner` folder are currently mostly unused (they just contain the route files for evaluation) but can be used to run evaluation on the CARLA eval routes or longest6_v2 or other benchmarks (see [Carla Garage](https://github.com/autonomousvision/carla_garage)). The folder `Bench2Drive` (with its own leaderboard and scenario_runner folders) is used to run closed-loop eval on the Bench2Drive Benchmark. The `team_code` folder is used for all files to run closed-loop agents in carla (for the expert, simlingo and simlingo_base).

**Training**: `simlingo_base_training` and `simlingo_training` contain all files to run training. `simlingo_training` also contains the files to start the language evaluation.

**Dataset**: Our dataset is stored in a folder called `database`.



## Dataset download
You can find our dataset here: https://huggingface.co/datasets/RenzKa/simlingo
The uploaded data contains the driving dataset, VQA, Commentary, and Dreamer labels.
### Download the whole dataset using git with Git LFS

```bash
# Clone the repository
git clone https://huggingface.co/datasets/RenzKa/simlingo

# Navigate to the directory
cd simlingo

# Pull the LFS files
git lfs pull
```

### Download a single file with wget

```bash
# Download individual files (replace with actual file URLs from Hugging Face)
wget https://huggingface.co/datasets/RenzKa/simlingo/resolve/main/[filename].tar.gz
```

### Extract to a single directory - please specify the location where you want to store the dataset
```bash
# Create output directory
mkdir -p database/simlingo

# Extract all archives to the same directory
for file in *.tar.gz; do
    echo "Extracting $file to database/simlingo/..."
    tar -xzf "$file" -C database/simlingo/
done
```


## Dataset generation
If you download our dataset from Huggingface, you don't need to follow any of the steps from this section.
If you only want to perfrom closed-loop driving evaluation, there is no need to download our dataset.

### Driving Data

This repository uses the open-source expert PDM-Lite from the paper [DriveLM](https://arxiv.org/abs/2312.14150) to generate the driving dataset. Most of the code for the data collection is taken from [Carla Garage](https://github.com/autonomousvision/carla_garage). However, we changed some hyperparameter and used the data_agent from DriveLM which saves the required auxiliary information during data collection which is needed to generate the VQA and commentary data.

**Generate driving data:** To re-generate the data, we provide a script for a SLURM cluster, which parallelizes data collection across many GPUs (2080ti in our case). First, adjust the paths etc. in lines 213-230 of [collect_dataset_slurm.py](collect_dataset_slurm.py). You can specify the SLURM partition in [partition.txt](partition.txt) and change it during runtime. [max_num_jobs.txt](max_num_jobs.txt) specifies how many parallel SLURM jobs are submitted. This can also be changed during runtime. The data collection is started via `sbatch 0_run_collect_dataset_slurm.sh`, which calls `collect_dataset_slurm.py`. 
Increase the number in [max_num_jobs.txt](max_num_jobs.txt) once your setup works. 

**Dataset cleaning:** After the dataset is collected you can use `dataset_generation/delete_failed_runs.py` and `dataset_generation/delete_infraction_routes.py` to delete routes where the expert failed or carla crashed and the routes had to be restarted.

**Route files:** The routes for data collection are stored in [data/simlingo](data/simlingo/). **Note:** These are different route files as used in the Carla Garage. To generate our route files, you can use the following script that generates our modified route files from the original Carla route files: `bash dataset_generation/split_route_files.sh`
This splits the long training and validation route files provided by Carla into short routes with max 1 or 3 scenarios and balances and upsamples the scenarios.

PDM-Lite uses a modified version of the CARLA leaderboard that exposes additional information about the scenarios and makes data collection easier. They can be found in the [leaderboard_autopilot](leaderboard_autopilot) and [scenario_runner_autopilot](scenario_runner_autopilot) folders.

The dataset provided in this repository is not perfect. At some point while improving the model, you will likely need to collect an improved version.

### Data buckets
Our bucket file is included in the released dataset. Check out our Huggingface repo.
If you want to generate your own buckets you can use the script `dataset_generation/data_buckets/carla_get_buckets.py`.

### Language Data
**VQA (DriveLM):** We use the script (with minor modifications) from [DriveLM](https://github.com/OpenDriveLab/DriveLM/tree/DriveLM-CARLA) to generate VQA labels. You can run `dataset_generation/language_labels/drivelm/carla_vqa_generator_main.py` to generate the VQA labels for your dataset. We used ChatGPT to augment the questions and answers. We provide the augmented templates, which we load during training in the folder [data/augmented_templates/drivelm_train_augmented_v2](data/augmented_templates/drivelm_train_augmented_v2). An example script to generate those augmented sentences can be found here: [dataset_generation/get_augmentations/gpt_augment_vqa.py](dataset_generation/get_augmentations/gpt_augment_vqa.py). **Note:** To be able to generate the VQA labels we save many auxiliary information of the simulator state during data collection. If you use a different dataset, it is likely that this labelling script does not work.

**Commentary:** In this work we provide a new script to generate commentary labels. To generate commentary labels for your dataset, run `dataset_generation/language_labels/commentary/carla_commentary_generator_main.py`. We used ChatGPT to augment the questions and answers. We provide the augmented templates, which we load during training in the folder [data/augmented_templates/commentary_augmented.json](data/augmented_templates/commentary_augmented.json). Unfortunately, based on how the project evolved the augmentations were first done manually for subsentences and later merged. If helpful, we provide the subsentence level augmentationes [here](data/augmented_templates/commentary_subsentence.json) and the script to merge those to the final ones [here](dataset_generation/get_augmentations/commentary_merge_augmented.py). **Note:** To be able to generate the commentary labels, we save auxiliary information of the simulator state during data collection. If you use a different dataset, it is likely that this labelling script does not work.

_File structure:_
``` bash
"image": # Path to RGB image
"commentary": # Commentary string (not augmented)
"commentary_template": # Commentary with placeholders for changing parts (e.g., object description, location). This is used to retrieve the augmentations.
"cause_object_visible_in_image": # Whether the object that causes the expert actions is visible in the front view image. Could be used to filter samples where the commentary describes an action based on an object not visible.
"cause_object": # Dictionary with attributes of the object causing the expert action.
"cause_object_string": # Language description of the cause object (e.g., dark green car that is to the front)
"scenario_name": # Name of the active CARLA scenario
"placeholder": # Dictionary to be able to replace the placeholders in commentary_template.
```

### Dreamer Data
To improve the alignment of language and actions, we propose _Action Dreaming_ for which we provide a dataset with multiple different future trajectories given a language instruction. The language instructions cover a wide range of modes (e.g., speed changes, lane changes, object-centric navigation, crashes) with a label indicating whether the execution is allowed and safe or not. 
To generate the labels, run `dataset_generation/dreamer_data/dreamer_generator.py`.

_File structure:_
``` bash
category: # e.g. "target_speed", "stop", "faster", "crash", ...
      "waypoints": # Dreaming waypoints
      "route": # Dreaming path
      "rgb_path": # Path to RGB image in dataset
      "allowed": # Flag if execution is allowed.
      "mode": # category
      "info": # more information, e.g., about current, target, and final speed
      "route_reasoning": # Language description about the route.
      "dreamer_instruction": # Language instruction.
      "instructions_templates": # Instruction with placeholders for changing parts (e.g., object description, location). This is used to retrieve the augmentations.
      "templates_placeholders": # Dictionary to be able to replace the placeholders in commentary_template.
      "dreamer_answer_safety": # Answer when safety mode is activated.
      "safe_to_execute": # Flag if the instruction is safe to execute.
```

## Training
We provide code for the smaller model SimLingo-Base (previously CarLLaVA - without language capabilities) in the folder `simlingo_base_training` and for the full model SimLingo in `simlingo_training`. For the config managment we use hydra. The config parameters are defined in the `config.py` file and can be adjusted in the `.yaml` files inside the `config` folder. **Note:** You should double check if the paths to the dataset is correct.

We provide a SLURM script to start training: [train_simlingo_seed1.sh](train_simlingo_seed1.sh). This can be easily converted to a bash script to locally start the training. The entry file for training is [simlingo_training/train.py](simlingo_training/train.py).

With the default config, the training logs to Wandb. Login is required. We also include a visualization callback that plots ground truth and predicted waypoints during training.


## Evaluation

The model file can be downloaded from huggingface: https://huggingface.co/RenzKa/simlingo.
If you only want to perfrom closed-loop driving evaluation, there is no need to download our dataset.


### Bench2Drive
Bench2Drive is a CARLA benchmark proposed by the paper [Bench2Drive: Towards Multi-Ability Benchmarking of Closed-Loop End-To-End Autonomous Driving](https://arxiv.org/abs/2406.03877). It consists of 220 very short (~150m) routes split across all towns with 1 safety critical scenario in each route.
Since it uses all towns for training, the methods have seen the test towns during training, so it can be considered a 'training' benchmark (reminiscent of level 4 driving).
The benchmark also comes with a training dataset generated by the [Think2Drive](https://arxiv.org/abs/2402.16720) expert, but we use the open-source expert [PDM-Lite](https://arxiv.org/abs/2312.14150) that achieves better resuslts and can be adapted to collect the necessary labels to produce VQA, Commentary and Dreamer data.
The benchmark and additional instructions can be found in the [Bench2Drive](Bench2Drive) folder.

**Start eval:** Evaluation on a SLURM cluster can be run with [start_eval_simlingo.py](start_eval_simlingo.py). The config dictionary needs to be adjusted with the correct names and paths. Most things that need to be changed are marked with TODO tags in [start_eval_simlingo.py](start_eval_simlingo.py). 

**Get results:** The script [Bench2Drive/tools/merge_route_json.py](Bench2Drive/tools/merge_route_json.py) can be used to obtain the final metrics after the evaluation is done. Make sure that all 220 routes are evaluated.

The Bench2Drive folder is based on version 0.0.3 of the [Bench2Drive repository](https://github.com/Thinklab-SJTU/Bench2Drive). Please cite the [Bench2Drive paper](https://arxiv.org/abs/2406.03877) when using the benchmark.

### Language eval
NOTE: Files might get cleaned at some point in the future (maybe not, depending on my time). Since the dataset and model are a reproduction and not the original ones from the paper, numbers deviate slightly. However, conclusions drawn in the paper still hold. We will update the numbers shortly.

Entry point for the language evaluation is [simlingo_training/eval.py](simlingo_training/eval.py). Please change the variable `eval_mode` to `QA`, `commentary` or `Dreaming`.
Afterwards, to obtain the metrics you can run [simlingo_training/eval_metrics.py](simlingo_training/eval_metrics.py). For this you first need to specify an OpenAI key here: [simlingo_training/utils/gpt_eval.py](simlingo_training/utils/gpt_eval.py)


---

## Fail2Drive Integration

[Fail2Drive](https://github.com/autonomousvision/fail2drive) is a CARLA 0.9.15 benchmark that tests closed-loop generalization on 17 categories of truly unseen long-tail scenarios (animals, bad parking, road blocks, pedestrian crowds, obscured stop signs, etc.).

This repo integrates Fail2Drive so that you can:
1. **Evaluate SimLingo** on official and custom Fail2Drive routes.
2. **Collect training data** with PDM-lite on those routes.

### What was added (vs. upstream SimLingo)

| Component | Change |
|---|---|
| `scenario_runner/srunner/scenarios/` | +5 Fail2Drive scenario types: `roadblocked`, `image_on_object`, `obscured_stop_sign`, `pedestrian_crowd`, `pedestrian_on_road` |
| `scenario_runner_autopilot/srunner/scenarios/` | same 5 files (for PDM-lite data collection) |
| `leaderboard/leaderboard/utils/route_parser.py` | Parses optional `<metrics><skip>` blocks in route XMLs |
| `leaderboard/leaderboard/scenarios/route_scenario.py` | Respects `skip_criteria` (e.g. skip `MinSpeedTest` for blocked-road scenarios) |
| `leaderboard_autopilot/` | Same route_parser and route_scenario changes |
| `leaderboard/data/fail2drive_split/` | 100 official Fail2Drive route XMLs (17 categories) |
| `leaderboard/data/fail2drive_customized/` | Custom routes you create with the toolbox |
| `start_eval_simlingo.py` | New `bash_file_fail2drive()` function + `"benchmark": "fail2drive"` config support |
| `collect_dataset_f2d.py` | New data-collection script for Fail2Drive routes using PDM-lite |
| `env_vars.sh` | Environment setup for `simlingo_f2d` conda env |
| `environment_f2d.yaml` | Python 3.10 conda env spec |

---

### Creating New Scenes

Fail2Drive provides a GUI toolbox to create custom route XMLs without writing code.

#### Step 1 — Launch the Fail2Drive toolbox
The toolbox is included in the original [fail2drive](https://github.com/autonomousvision/fail2drive) repo under `toolbox/`.
You need a running CARLA instance and the `fail2drive` conda env:
```bash
conda activate fail2drive
cd /path/to/fail2drive
# Start CARLA first (in another terminal)
bash start_carla.sh

# Then launch the toolbox
bash toolbox/start_window.sh
```

#### Step 2 — Design your route and scenario
In the toolbox GUI you can:
- Pick a start waypoint and end waypoint on the CARLA map
- Select a scenario type from the supported list
- Configure scenario parameters (obstacle model, trigger distance, direction, etc.)
- Export the route as an XML file

#### Step 3 — Save your custom route XML
Place the exported XML file in:
```
simlingo_f2d/leaderboard/data/fail2drive_customized/
```

**Naming convention** (recommended, matches the official fail2drive format):
```
<Category>_<four-digit-id>.xml
```
Examples:
```
leaderboard/data/fail2drive_customized/MyObstacle_0000.xml
leaderboard/data/fail2drive_customized/LeftTurnBlock_0001.xml
```

Each XML encodes the town, waypoints, weather, and scenario parameters:
```xml
<routes>
  <route id="0" town="Town12">
    <weathers>
      <weather route_percentage="0" cloudiness="5.0" ... sun_altitude_angle="45.0" .../>
    </weathers>
    <waypoints>
      <position x="-1703.7" y="4889.6" z="376.8"/>
      <position x="-1665.6" y="4856.0" z="376.8"/>
    </waypoints>
    <scenarios>
      <scenario name="DynamicObjectCrossing_0" type="DynamicObjectCrossing">
        <distance value="12"/>
        <direction value="right"/>
        <blocker_model value="static.prop.vendingmachine"/>
        <walker value="walker.pedestrian.*"/>
        <trigger_point x="-1665.5" y="4865.5" z="376.8" yaw="269.9"/>
      </scenario>
    </scenarios>
  </route>
</routes>
```

For scenarios where `MinSpeedTest` would unfairly penalise the agent (e.g. road fully blocked), add a metrics block:
```xml
<metrics>
  <skip name="MinSpeedTest"/>
</metrics>
```

---

### Evaluating SimLingo on Fail2Drive Scenes

Edit `start_eval_simlingo.py` and uncomment + fill in the `fail2drive` config block:

```python
configs = [
    {
        "agent": "simlingo",
        "checkpoint": "/path/to/simlingo_f2d/outputs/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt",
        "benchmark": "fail2drive",
        "route_path": "/path/to/simlingo_f2d/leaderboard/data/fail2drive_split",  # or fail2drive_customized
        "seeds": [1],
        "tries": 2,
        "out_root": "/path/to/simlingo_f2d/eval_results/Fail2Drive",
        "carla_root": "/path/to/CarlaUE4_0.9.15",
        "repo_root": "/path/to/simlingo_f2d",
        "agent_file": "/path/to/simlingo_f2d/team_code/agent_simlingo.py",
        "team_code": "team_code",
        "agent_config": "not_used",
        "username": "YOUR_USERNAME",
    }
]
```

Then run:
```bash
conda activate simlingo_f2d
cd /path/to/simlingo_f2d
python start_eval_simlingo.py
```

Results are written to `eval_results/Fail2Drive/<agent>/<seed>/res/`.

---

### Collecting Training Data on Fail2Drive Scenes

PDM-lite (the same expert used for SimLingo's original dataset) can collect training data on Fail2Drive routes.

#### Edit `collect_dataset_f2d.py`
Fill in your paths and cluster settings at the bottom of the file:
```python
default_partition = "YOUR_PARTITION"   # SLURM partition name
username         = "YOUR_USERNAME"     # your cluster username
code_root        = "/path/to/simlingo_f2d"
carla_root       = "/path/to/CarlaUE4_0.9.15"
```

By default the script collects on **all** official Fail2Drive routes **plus** any XMLs you put in `fail2drive_customized/`.
To restrict to only your custom scenes, comment out the `route_folder_official` line.

#### Run data collection
```bash
conda activate simlingo_f2d
cd /path/to/simlingo_f2d
# Optionally set max parallel SLURM jobs:
echo "4" > max_num_jobs.txt
python collect_dataset_f2d.py
```

Data is saved to `database/simlingo_f2d_<date>/data/<route_id>/`.
Checkpoint JSON results go to `database/simlingo_f2d_<date>/results/`.

#### Using the collected data for training
The collected data has the same format as the original SimLingo dataset.
Add the new data directory to your training config (in `simlingo_training/config/`) alongside the original dataset.

---

## Citations
If you find this repository useful, please consider giving us a star &#127775;.
Please cite the following papers for the respective components of the repo:

SimLingo:
```BibTeX
@InProceedings{Renz2025cvpr,
  title={SimLingo: Vision-Only Closed-Loop Autonomous Driving with Language-Action Alignment},
  author={Renz, Katrin and Chen, Long and Arani, Elahe and Sinavski, Oleg},
  booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```

PDM-Lite expert:
```BibTeX
@inproceedings{Sima2024ECCV,
  title={DriveLM: Driving with Graph Visual Question Answering},
  author={Chonghao Sima and Katrin Renz and Kashyap Chitta and Li Chen and Hanxue Zhang and Chengen Xie and Jens Beißwenger and Ping Luo and Andreas Geiger and Hongyang Li},
  booktitle={Proc. of the European Conf. on Computer Vision (ECCV)},
  year={2024}
}
```

Bench2Drive benchmark:

```BibTeX
@inproceedings{Jia2024NeurIPS,
  title={Bench2Drive: Towards Multi-Ability Benchmarking of Closed-Loop End-To-End Autonomous Driving},
  author={Xiaosong Jia and Zhenjie Yang and Qifeng Li and Zhiyuan Zhang and Junchi Yan},
  booktitle={NeurIPS 2024 Datasets and Benchmarks Track},
  year={2024}
}
```

Fail2Drive benchmark:
```BibTeX
@inproceedings{Hoss2025fail2drive,
  title={Fail2Drive: A Benchmark for Closed-Loop Generalization on Unseen Long-Tail Scenarios},
  author={Hoss, Michael and others},
  booktitle={arXiv},
  year={2025}
}
```

## Other Resources
- [tuPlan garage](https://github.com/autonomousvision/tuplan_garage) | [CARLA garage](https://github.com/autonomousvision/carla_garage) | [Survey on E2EAD](https://github.com/OpenDriveLab/End-to-end-Autonomous-Driving)
- [DriveLM](https://github.com/OpenDriveLab/DriveLM/tree/main) | [PlanT](https://github.com/autonomousvision/plant) | [KING](https://github.com/autonomousvision/king) | [TransFuser](https://github.com/autonomousvision/transfuser) | [NEAT](https://github.com/autonomousvision/neat)
- [Fail2Drive](https://github.com/autonomousvision/fail2drive)

