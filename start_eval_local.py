"""
Local (non-SLURM) evaluation script for SimLingo on Fail2Drive or Bench2Drive routes.
Runs one route at a time sequentially on this machine.

Usage:
    conda activate simlingo_f2d
    cd /mnt/SSD/Coop_closed_loop/simlingo_f2d
    python start_eval_local.py
"""

import os
import sys
import subprocess
import time
import signal
import socket
import ujson
import atexit
from tqdm import tqdm

_carla_procs = []

def _cleanup():
    for proc in _carla_procs:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass

atexit.register(_cleanup)
signal.signal(signal.SIGINT,  lambda s, f: (print("\nCleaning up CARLA..."), _cleanup(), exit(1)))
signal.signal(signal.SIGTERM, lambda s, f: (print("\nCleaning up CARLA..."), _cleanup(), exit(1)))

# ── CONFIG ────────────────────────────────────────────────────────────────────
configs = [
    {
        "agent": "simlingo",
        "checkpoint": "/mnt/SSD/Coop_closed_loop/models/simlingo_official/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt",
        "benchmark": "fail2drive",
        "route_path": "/mnt/SSD/Coop_closed_loop/simlingo_f2d/leaderboard/data/fail2drive_customized",
        "seeds": [1],
        "tries": 2,
        "out_root": "/mnt/SSD/Coop_closed_loop/simlingo_f2d/eval_results/Fail2Drive",
        "carla_root": "/mnt/SSD/Coop_closed_loop/fail2drive/f2d_carla",
        "repo_root": "/mnt/SSD/Coop_closed_loop/simlingo_f2d",
        "agent_file": "/mnt/SSD/Coop_closed_loop/simlingo_f2d/team_code/agent_simlingo.py",
        "cuda_device": "3",      # GPU index for CUDA_VISIBLE_DEVICES (GPU 3 is idle; CARLA uses GPU 2)
        "inference_skip": 5,     # run model inference every N frames; 1=every frame (accurate), 5=5x faster
    },
]

CARLA_STARTUP_WAIT = 60    # seconds to wait for CARLA to start
ROUTE_TIMEOUT     = 300    # leaderboard --timeout value (seconds of game time)
# Wall-clock timeout for subprocess: multiply by 100 to handle slow GPU inference
# At 0.08x game/real ratio, 300 game seconds ≈ 3750 real seconds
SUBPROCESS_TIMEOUT = ROUTE_TIMEOUT * 100  # 30000 seconds ≈ 8 hours max
# ─────────────────────────────────────────────────────────────────────────────


def find_free_port(start=20000, end=21000):
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found in [{start}, {end})")


def is_done(result_file, tries_left):
    if not os.path.exists(result_file):
        return False
    try:
        with open(result_file, "r") as f:
            data = ujson.load(f)
        progress = data["_checkpoint"]["progress"]
        if len(progress) < 2 or progress[0] < progress[1]:
            return False
        for record in data["_checkpoint"]["records"]:
            if record["status"] in (
                "Failed - Agent couldn't be set up",
                "Failed",
                "Failed - Simulation crashed",
                "Failed - Agent crashed",
            ):
                return False
        return True
    except Exception:
        return False


def run_route(cfg, route, route_id, seed, viz_path, result_file, log_file, err_file):
    carla_root = cfg["carla_root"]
    repo_root  = cfg["repo_root"]

    world_port     = find_free_port(20000, 20500)
    streaming_port = find_free_port(20500, 21000)
    tm_port        = find_free_port(21000, 21500)

    import site as _site
    _site_pkgs = ":".join(_site.getsitepackages())

    env = os.environ.copy()
    env["CARLA_ROOT"]          = carla_root
    env["WORK_DIR"]            = repo_root
    env["LEADERBOARD_ROOT"]    = f"{repo_root}/leaderboard"
    env["SCENARIO_RUNNER_ROOT"]= f"{repo_root}/scenario_runner"
    env["PYTHONPATH"]          = (
        f"{carla_root}/PythonAPI/carla:"
        f"{repo_root}/leaderboard:"
        f"{repo_root}/scenario_runner:"
        f"{repo_root}:"
        f"{_site_pkgs}:"
        + env.get("PYTHONPATH", "")
    )
    env["VK_ICD_FILENAMES"]    = "/usr/share/vulkan/icd.d/nvidia_icd.json"
    env["SAVE_PATH"]           = viz_path
    env["VIZ_PATH"]            = viz_path
    env["TRANSFORMERS_OFFLINE"]= "1"
    env["HF_HUB_OFFLINE"]      = "1"
    env["CUDA_VISIBLE_DEVICES"]     = cfg.get("cuda_device", "2")
    env["SIMLINGO_INFERENCE_SKIP"]  = str(cfg.get("inference_skip", "1"))

    os.makedirs(viz_path, exist_ok=True)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    os.makedirs(os.path.dirname(err_file), exist_ok=True)

    # Start CARLA
    carla_cmd = [
        f"{carla_root}/CarlaUE4.sh",
        f"-carla-rpc-port={world_port}",
        "-nosound", "-RenderOffScreen",
        "-carla-primary-port=0",
        "-graphicsadapter=0",
        f"-carla-streaming-port={streaming_port}",
    ]
    print(f"  Starting CARLA on port {world_port}...")
    carla_proc = subprocess.Popen(
        carla_cmd, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )
    _carla_procs.append(carla_proc)
    time.sleep(CARLA_STARTUP_WAIT)

    # Run leaderboard evaluator
    eval_cmd = [
        sys.executable, "-u",
        f"{repo_root}/leaderboard/leaderboard/leaderboard_evaluator.py",
        f"--routes={route}",
        "--repetitions=1",
        "--track=SENSORS",
        f"--checkpoint={result_file}",
        f"--timeout={ROUTE_TIMEOUT}",
        f"--agent={cfg['agent_file']}",
        f"--agent-config={cfg['checkpoint']}",
        f"--traffic-manager-seed={seed}",
        f"--port={world_port}",
        f"--traffic-manager-port={tm_port}",
    ]

    success = False
    try:
        with open(log_file, "w") as lf, open(err_file, "w") as ef:
            result = subprocess.run(
                eval_cmd, env=env,
                stdout=lf, stderr=ef,
                timeout=SUBPROCESS_TIMEOUT,
            )
        success = (result.returncode == 0)
        if not success:
            with open(log_file) as lf:
                out_lines = lf.readlines()
            print(f"  Route {route_id} failed (rc={result.returncode}). Last stdout lines:")
            for line in out_lines[-30:]:
                print("   ", line, end="")
            with open(err_file) as ef:
                err_lines = ef.readlines()
            print(f"  Last stderr lines:")
            for line in err_lines[-10:]:
                print("   ", line, end="")
    except subprocess.TimeoutExpired:
        print(f"  Route {route_id} timed out.")
    except Exception as e:
        print(f"  Route {route_id} error: {e}")
    finally:
        # Kill CARLA and all its children
        try:
            os.killpg(os.getpgid(carla_proc.pid), signal.SIGKILL)
        except Exception:
            pass
        carla_proc.wait()
        if carla_proc in _carla_procs:
            _carla_procs.remove(carla_proc)

    return success


def main():
    for cfg in configs:
        route_path = cfg["route_path"]
        routes = sorted(x for x in os.listdir(route_path) if x.endswith(".xml"))

        fill_zeros = 3 if cfg["benchmark"] == "bench2drive" else 4

        job_queue = []
        for seed in cfg["seeds"]:
            seed = str(seed)
            base_dir = os.path.join(cfg["out_root"], cfg["agent"], cfg["benchmark"], seed)
            for fname in routes:
                route_id    = fname.split("_")[-1][:-4].zfill(fill_zeros)
                route       = os.path.join(route_path, fname)
                viz_path    = os.path.join(base_dir, "viz", route_id)
                result_file = os.path.join(base_dir, "res", f"{route_id}_res.json")
                log_file    = os.path.join(base_dir, "out", f"{route_id}_out.log")
                err_file    = os.path.join(base_dir, "err", f"{route_id}_err.log")
                for d in [os.path.join(base_dir, "res"),
                          os.path.join(base_dir, "out"),
                          os.path.join(base_dir, "err")]:
                    os.makedirs(d, exist_ok=True)
                job_queue.append({
                    "route": route, "route_id": route_id, "seed": seed,
                    "viz_path": viz_path, "result_file": result_file,
                    "log_file": log_file, "err_file": err_file,
                    "tries": cfg["tries"],
                })

        total = len(job_queue)
        print(f"\nRunning {total} routes for benchmark={cfg['benchmark']}, seeds={cfg['seeds']}")
        progress = tqdm(total=total)

        while job_queue:
            job = job_queue.pop(0)

            if is_done(job["result_file"], job["tries"]):
                progress.update(1)
                continue

            print(f"\n[{total - len(job_queue)}/{total}] Route {job['route_id']} seed={job['seed']} (tries left: {job['tries']})")
            run_route(cfg, job["route"], job["route_id"], job["seed"],
                      job["viz_path"], job["result_file"], job["log_file"], job["err_file"])

            if is_done(job["result_file"], job["tries"]):
                progress.update(1)
            elif job["tries"] > 1:
                job["tries"] -= 1
                job_queue.append(job)   # retry at end of queue
                print(f"  Will retry route {job['route_id']} ({job['tries']} tries left)")
            else:
                print(f"  Route {job['route_id']} failed after all retries.")
                progress.update(1)

        progress.close()
        print(f"\nDone. Results in: {cfg['out_root']}")


if __name__ == "__main__":
    main()
