# SimLingo + Fail2Drive

This repository extends [SimLingo (CVPR'25 Highlight)](https://github.com/RenzKa/simlingo) with [Fail2Drive](https://github.com/autonomousvision/fail2drive) support, enabling closed-loop evaluation and PDM-lite data collection on long-tail unseen scenarios using CARLA 0.9.15.

> **Upstream:** [RenzKa/simlingo](https://github.com/RenzKa/simlingo) — see that repo for original training, dataset, and Bench2Drive documentation.

---

## Contents

1. [Environment Setup](#environment-setup)
2. [Compatibility Fixes Applied](#compatibility-fixes-applied)
3. [Repository Structure](#repository-structure)
4. [Evaluating on Fail2Drive Scenes (Local, No SLURM)](#evaluating-on-fail2drive-scenes-local-no-slurm)
   - [Config Reference](#config-reference)
   - [GPU Assignment Notes](#gpu-assignment-notes)
   - [Checking Results](#checking-results)
5. [Creating New Custom Scenes](#creating-new-custom-scenes)
6. [Collecting Training Data on Fail2Drive Scenes](#collecting-training-data-on-fail2drive-scenes)
7. [Citations](#citations)

---

## Environment Setup

### Prerequisites

- CARLA 0.9.15 with the **Python 3.10 wheel** (the Fail2Drive distribution ships this at `CarlaUE4/PythonAPI/carla/dist/carla-0.9.15-cp310-cp310-linux_x86_64.whl`)
- conda / mamba
- NVIDIA GPU with CUDA 12.x (flash-attn is compiled for CUDA 12.4)

### Step 1 — Clone

```bash
git clone git@github.com:zhumorui/simlingo_f2d.git
cd simlingo_f2d
```

### Step 2 — Symlink CARLA

```bash
ln -s /path/to/CarlaUE4_0.9.15  f2d_carla
```

### Step 3 — Create the conda environment

```bash
conda env create -f environment_f2d.yaml
conda activate simlingo_f2d
```

This installs Python 3.10 and the packages in `team_code/requirements.txt`.

### Step 4 — Install CARLA Python wheel

```bash
pip install f2d_carla/PythonAPI/carla/dist/carla-0.9.15-cp310-cp310-linux_x86_64.whl
```

### Step 5 — Install flash-attn

flash-attn must be compiled against the installed CUDA toolkit. On a CUDA 12.x machine:

```bash
pip install flash-attn
```

If this fails, build from source:
```bash
pip install flash-attn --no-build-isolation
```

### Step 6 — Download / place the SimLingo model checkpoint

Download from [HuggingFace (RenzKa/simlingo)](https://huggingface.co/RenzKa/simlingo) and place at:
```
/path/to/models/simlingo_official/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt
```

### Step 7 — Place the InternVL2-1B base model locally

`agent_simlingo.py` loads `OpenGVLab/InternVL2-1B` via HuggingFace AutoModel. When running offline (`TRANSFORMERS_OFFLINE=1`), it must be resolved from a local directory. Create a symlink in `pretrained/`:

```bash
mkdir -p pretrained
# Download InternVL2-1B from HuggingFace once, then symlink:
ln -s /path/to/InternVL2-1B  pretrained/InternVL2-1B
```

The agent automatically detects and uses the local path at startup (see [Compatibility Fixes](#compatibility-fixes-applied)).

### Verification

```bash
conda activate simlingo_f2d
python -c "import carla; print('CARLA OK:', carla.__version__)"
python -c "import torch; print('PyTorch OK:', torch.__version__)"
python -c "import pytorch_lightning; print('PL OK:', pytorch_lightning.__version__)"
python -c "import peft; print('peft OK:', peft.__version__)"
```

---

## Compatibility Fixes Applied

The upstream SimLingo code was written for Python 3.8 / numpy 1.23 / transformers 4.21. This fork patches the following incompatibilities to run with CARLA 0.9.15 (Python 3.10), numpy 1.26, and transformers 4.46.

### `team_code/agent_simlingo.py`

| Issue | Root Cause | Fix |
|---|---|---|
| `OSError: OpenGVLab/InternVL2-1B does not appear to have a file named config.json` | `AutoProcessor.from_pretrained` tried the HuggingFace hub while `TRANSFORMERS_OFFLINE=1` | After loading hydra config, resolve the local `pretrained/<model>` directory and override `cfg.model.vision_model.variant` to the absolute path |
| `ImportError: cannot import name 'LlamaConfig' from 'transformers'` | `transformers==4.21.2` is too old; `LlamaConfig` was added in 4.28 | Upgraded to `transformers==4.46.3` |
| `ValueError: setting an array element with a sequence. inhomogeneous shape` | numpy 1.26 is stricter — `velocity[0].data.cpu().numpy()` returns shape `[1]`, which caused the PID controller's speed window to become inhomogeneous | Changed to `float(velocity[0].data.cpu().numpy())` |
| `TypeError: unsupported format string passed to Tensor.__format__` | Debug f-string `f"gt_vel={gt_velocity:.3f}"` with `gt_velocity` still being a tensor | Changed to `gt_velocity = float(tick_data['speed'])` |
| Inference too slow for real-time (0.006x ratio) | Model inference (~2.5 s/step on a single A6000) ran every CARLA frame (20 FPS) | Added frame-skip: `SIMLINGO_INFERENCE_SKIP=N` runs the VLM only every N steps; stale control is reused on skipped frames |

**Local model path fix** (in `agent_simlingo.py`, after hydra config load):
```python
_model_name = cfg.model.vision_model.variant.split('/')[-1]
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_local_pretrained = os.path.join(_repo_root, 'pretrained', _model_name)
if os.path.isdir(_local_pretrained):
    self.cfg.model.vision_model.variant = _local_pretrained
    cfg = self.cfg
```

**Frame-skip** (in `run_step`, after `tick_data = self.tick(input_data)`):
```python
_inference_skip = int(os.environ.get("SIMLINGO_INFERENCE_SKIP", "1"))
if _inference_skip > 1 and self.step % _inference_skip != 0:
    return self.control
```

### `team_code/requirements.txt`

Updated package versions required on Python 3.10 / CARLA 0.9.15:

```
transformers==4.46.3   # was 4.21.2 — needed for LlamaConfig in InternVL2
pytorch-lightning==2.4.0
peft==0.13.2
flash-attn             # compiled against CUDA 12.4
hydra-core==1.3.2
```

---

## Repository Structure

```
simlingo_f2d/
├── team_code/
│   ├── agent_simlingo.py        # SimLingo closed-loop agent (patched for F2D compat)
│   └── requirements.txt         # pip dependencies for simlingo_f2d env
├── leaderboard/
│   └── data/
│       ├── fail2drive_split/    # 100 official Fail2Drive route XMLs (17 categories)
│       └── fail2drive_customized/  # custom routes you create
├── scenario_runner/srunner/scenarios/
│   └── (+ 5 Fail2Drive scenario types: roadblocked, image_on_object,
│          obscured_stop_sign, pedestrian_crowd, pedestrian_on_road)
├── start_eval_local.py          # local (no-SLURM) eval script — run one route at a time
├── start_eval_simlingo.py       # SLURM eval script (Bench2Drive + Fail2Drive)
├── collect_dataset_f2d.py       # PDM-lite data collection on Fail2Drive routes (SLURM)
├── environment_f2d.yaml         # conda env spec (Python 3.10)
└── pretrained/
    └── InternVL2-1B -> /path/to/InternVL2-1B  # symlink to local model
```

---

## Evaluating on Fail2Drive Scenes (Local, No SLURM)

`start_eval_local.py` runs one route at a time on your local machine without SLURM. It starts and stops CARLA automatically for each route, writes results to JSON, and retries failed routes.

```bash
conda activate simlingo_f2d
cd /path/to/simlingo_f2d
python start_eval_local.py
```

### Config Reference

Edit the `configs` list at the top of `start_eval_local.py`:

```python
configs = [
    {
        # Path to the SimLingo pytorch_model.pt checkpoint
        "checkpoint": "/path/to/models/simlingo_official/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt",

        # "fail2drive" uses 4-digit zero-padding for route IDs
        "benchmark": "fail2drive",

        # Folder containing .xml route files to evaluate
        "route_path": "/path/to/simlingo_f2d/leaderboard/data/fail2drive_customized",
        # or for the 100 official routes:
        # "route_path": "/path/to/simlingo_f2d/leaderboard/data/fail2drive_split",

        # Traffic-manager seeds (one eval pass per seed)
        "seeds": [1],

        # How many times to retry a failed route before giving up
        "tries": 2,

        # Root output directory; results go under <out_root>/<agent>/<benchmark>/<seed>/
        "out_root": "/path/to/simlingo_f2d/eval_results/Fail2Drive",

        # CARLA 0.9.15 installation root (must have CarlaUE4.sh)
        "carla_root": "/path/to/CarlaUE4_0.9.15",

        # This repo root
        "repo_root": "/path/to/simlingo_f2d",

        # Agent entry point
        "agent_file": "/path/to/simlingo_f2d/team_code/agent_simlingo.py",

        # CUDA_VISIBLE_DEVICES for the agent process
        # CARLA renders via Vulkan on the display GPU (graphicsadapter=0);
        # pick a different GPU for the agent to avoid contention.
        "cuda_device": "0",

        # Run VLM inference every N frames (1 = every frame, accurate but slow).
        # At inference_skip=5 on an A6000: ~0.08x game/real ratio.
        # Use inference_skip=1 when you have a fast GPU or need full accuracy.
        "inference_skip": 1,
    },
]
```

**Key constants** (also at the top of the file):

| Constant | Default | Meaning |
|---|---|---|
| `CARLA_STARTUP_WAIT` | 60 s | Seconds to wait for CARLA to initialize before starting the agent |
| `ROUTE_TIMEOUT` | 300 s | Leaderboard game-time timeout per route |
| `SUBPROCESS_TIMEOUT` | `ROUTE_TIMEOUT * 100` | Wall-clock timeout for the subprocess; increase if inference is very slow |

### GPU Assignment Notes

CARLA always uses the GPU selected by Vulkan (`-graphicsadapter=0`), which is the first display-capable GPU regardless of `CUDA_VISIBLE_DEVICES`. Set `cuda_device` to any other GPU index to avoid the agent and renderer competing.

| Scenario | Recommended setting |
|---|---|
| Dedicated GPU for agent (fast) | `cuda_device: "0"`, `inference_skip: 1` |
| CARLA on GPU 0, agent on GPU 1 | `cuda_device: "1"`, `inference_skip: 1` |
| Only one free GPU (slow machine) | `cuda_device: "<free>", inference_skip: 5` (~0.08x ratio) |

**Performance reference** (NVIDIA RTX A6000, `inference_skip=5`):

| Metric | Value |
|---|---|
| Game / real-time ratio | ~0.083x |
| Wall time per 300-game-second route | ~60 minutes |
| VLM inference frequency | every 5 frames (4 FPS effective) |

### Checking Results

Results are written as JSON files:
```
eval_results/Fail2Drive/<agent>/fail2drive/<seed>/
    res/<route_id>_res.json    # leaderboard result JSON
    out/<route_id>_out.log     # stdout from leaderboard_evaluator
    err/<route_id>_err.log     # stderr
    viz/<route_id>/            # saved sensor images (if SAVE_PATH used)
```

Quick summary across all results:
```bash
python3 - <<'EOF'
import ujson, glob, os
for f in sorted(glob.glob("eval_results/Fail2Drive/simlingo/fail2drive/1/res/*.json")):
    d = ujson.load(open(f))
    p = d["_checkpoint"]["progress"]
    recs = d["_checkpoint"]["records"]
    name = os.path.basename(f)
    if recs:
        r = recs[0]
        print(f"{name}: {r['status']} | route={r['scores'].get('score_route','?')} penalty={r['scores'].get('score_penalty','?')}")
    else:
        print(f"{name}: progress={p} (in progress)")
EOF
```

To merge all results into a single score (same tool as Bench2Drive):
```bash
python Bench2Drive/tools/merge_route_json.py \
    --results eval_results/Fail2Drive/simlingo/fail2drive/1/res/
```

---

## Creating New Custom Scenes

Fail2Drive provides a GUI toolbox for designing route XMLs without writing code.

### Step 1 — Launch the Fail2Drive toolbox

```bash
conda activate fail2drive
cd /path/to/fail2drive

# Start CARLA (separate terminal)
bash start_carla.sh

# Launch the toolbox
bash toolbox/start_window.sh
```

### Step 2 — Design your route

In the GUI:
- Pick start and end waypoints on the CARLA map
- Select a scenario type (obstacle, pedestrian crowd, blocked intersection, etc.)
- Configure parameters (trigger distance, direction, actor model, etc.)
- Export the route as XML

### Step 3 — Save to the customized folder

```
simlingo_f2d/leaderboard/data/fail2drive_customized/<name>.xml
```

**Naming convention:**
```
testing_<description>_<id>.xml
```

### Route XML format

```xml
<routes>
  <route id="0" town="Town12">
    <weathers>
      <weather route_percentage="0" cloudiness="5.0" sun_altitude_angle="45.0" .../>
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

For scenarios where a blocked road would unfairly fail `MinSpeedTest`, add:
```xml
<metrics>
  <skip name="MinSpeedTest"/>
</metrics>
```

---

## Collecting Training Data on Fail2Drive Scenes

PDM-lite can collect training data on Fail2Drive routes (for fine-tuning SimLingo on new scenarios).

Edit `collect_dataset_f2d.py` and fill in paths at the bottom:
```python
default_partition = "YOUR_SLURM_PARTITION"
username          = "YOUR_USERNAME"
code_root         = "/path/to/simlingo_f2d"
carla_root        = "/path/to/CarlaUE4_0.9.15"
```

Run:
```bash
conda activate simlingo_f2d
echo "4" > max_num_jobs.txt    # max parallel SLURM jobs
python collect_dataset_f2d.py
```

Data is saved to `database/simlingo_f2d_<date>/data/<route_id>/` in the same format as the original SimLingo dataset. Add the new path to `simlingo_training/config/` alongside the original dataset.

---

## Citations

SimLingo:
```bibtex
@InProceedings{Renz2025cvpr,
  title={SimLingo: Vision-Only Closed-Loop Autonomous Driving with Language-Action Alignment},
  author={Renz, Katrin and Chen, Long and Arani, Elahe and Sinavski, Oleg},
  booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```

Fail2Drive:
```bibtex
@inproceedings{Hoss2025fail2drive,
  title={Fail2Drive: A Benchmark for Closed-Loop Generalization on Unseen Long-Tail Scenarios},
  author={Hoss, Michael and others},
  booktitle={arXiv},
  year={2025}
}
```

PDM-Lite expert:
```bibtex
@inproceedings{Sima2024ECCV,
  title={DriveLM: Driving with Graph Visual Question Answering},
  author={Chonghao Sima and Katrin Renz and Kashyap Chitta and Li Chen and Hanxue Zhang and Chengen Xie and Jens Beißwenger and Ping Luo and Andreas Geiger and Hongyang Li},
  booktitle={Proc. of the European Conf. on Computer Vision (ECCV)},
  year={2024}
}
```
