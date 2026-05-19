"""
Local (no-SLURM) PDM-lite data collection on Fail2Drive or custom routes.
Runs one route at a time, saves sensor data (RGB, LiDAR, BEV, boxes) to disk.

Usage:
    conda activate simlingo_f2d
    cd /mnt/SSD/Coop_closed_loop/simlingo_f2d
    python collect_local.py

Set TMP_VISU=1 in config to also save merged BEV+RGB visualization frames.
After collection, find saved data under <out_root>/<route_id>/rgb/, lidar/, etc.
"""

import os
import sys
import site
import subprocess
import time
import signal
import socket
import ujson
import atexit
from datetime import datetime
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
_date = datetime.now().strftime("%Y%m%d_%H%M%S")

config = {
    # Set route_file to a single XML path to run just one scene.
    # Leave as None to run all XMLs in route_path.
    "route_file": None,
    # "route_file": "/mnt/SSD/Coop_closed_loop/simlingo_f2d/leaderboard/data/fail2drive_customized/testing_leftTurn_object.xml",

    "route_path": "/mnt/SSD/Coop_closed_loop/simlingo_f2d/leaderboard/data/fail2drive_customized",
    # Or for the 100 official routes:
    # "route_path": "/mnt/SSD/Coop_closed_loop/simlingo_f2d/leaderboard/data/fail2drive_split",

    "out_root": f"/mnt/SSD/Coop_closed_loop/simlingo_f2d/database/collect_local_{_date}",

    "carla_root": "/mnt/SSD/Coop_closed_loop/fail2drive/f2d_carla",
    "repo_root":  "/mnt/SSD/Coop_closed_loop/simlingo_f2d",

    "seed":  1,
    "tries": 2,

    # visu_only: save only front RGB frames — no LiDAR, boxes, or labels.
    # Use this to verify a scene looks correct before committing to full collection.
    # After the run: ffmpeg -framerate 20 -i <out_root>/data/<route_id>/rgb/%04d.jpg out.mp4
    "visu_only": 1,

    # Full data collection options (ignored when visu_only=1):
    # Set tmp_visu=1 to also save merged BEV+RGB composite frames.
    "tmp_visu": 0,
    # Set save_tf_labels=1 to also save semantics/depth (needed for full label generation).
    "save_tf_labels": 0,
}

CARLA_STARTUP_WAIT = 60
ROUTE_TIMEOUT     = 600   # game-time seconds; longer than eval because PDM-lite is thorough
SUBPROCESS_TIMEOUT = ROUTE_TIMEOUT * 50  # wall-clock upper bound
# ─────────────────────────────────────────────────────────────────────────────


def find_free_port(start, end):
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port in [{start}, {end})")


def is_done(result_file):
    if not os.path.exists(result_file):
        return False
    try:
        with open(result_file) as f:
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


def run_route(cfg, route_xml, route_id, save_path, result_file, log_file, err_file):
    carla_root = cfg["carla_root"]
    repo_root  = cfg["repo_root"]

    world_port     = find_free_port(20000, 20500)
    streaming_port = find_free_port(20500, 21000)
    tm_port        = find_free_port(21000, 21500)

    site_pkgs = ":".join(site.getsitepackages())

    env = os.environ.copy()
    env["CARLA_ROOT"]           = carla_root
    env["WORK_DIR"]             = repo_root
    env["LEADERBOARD_ROOT"]     = f"{repo_root}/leaderboard_autopilot"
    env["SCENARIO_RUNNER_ROOT"] = f"{repo_root}/scenario_runner_autopilot"
    env["PYTHONPATH"] = (
        f"{carla_root}/PythonAPI/carla:"
        f"{repo_root}/leaderboard_autopilot:"
        f"{repo_root}/scenario_runner_autopilot:"
        f"{repo_root}:"
        f"{site_pkgs}:"
        + env.get("PYTHONPATH", "")
    )
    env["VK_ICD_FILENAMES"]  = "/usr/share/vulkan/icd.d/nvidia_icd.json"
    env["SAVE_PATH"]         = save_path
    env["VISU_ONLY"]         = str(cfg.get("visu_only", 0))
    env["DATAGEN"]           = "0" if cfg.get("visu_only") else "1"
    env["TMP_VISU"]          = str(cfg.get("tmp_visu", 0))
    env["SAVE_TF_LABELS"]    = str(cfg.get("save_tf_labels", 0))
    env["RESUME"]            = "1"

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    os.makedirs(os.path.dirname(err_file), exist_ok=True)
    os.makedirs(os.path.dirname(result_file), exist_ok=True)

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

    # Run PDM-lite data collection via leaderboard_evaluator_local
    collect_cmd = [
        sys.executable, "-u",
        f"{repo_root}/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py",
        f"--routes={route_xml}",
        "--repetitions=1",
        "--track=MAP",
        f"--checkpoint={result_file}",
        f"--timeout={ROUTE_TIMEOUT}",
        f"--agent={repo_root}/team_code/data_agent.py",
        f"--agent-config={route_xml}",
        f"--traffic-manager-seed={cfg['seed']}",
        f"--port={world_port}",
        f"--traffic-manager-port={tm_port}",
        "--debug=0",
        "--resume=1",
    ]

    success = False
    try:
        with open(log_file, "w") as lf, open(err_file, "w") as ef:
            result = subprocess.run(
                collect_cmd, env=env,
                stdout=lf, stderr=ef,
                timeout=SUBPROCESS_TIMEOUT,
            )
        success = (result.returncode == 0)
        if not success:
            with open(log_file) as lf:
                lines = lf.readlines()
            print(f"  Route {route_id} failed (rc={result.returncode}). Last stdout:")
            for line in lines[-30:]:
                print("   ", line, end="")
            with open(err_file) as ef:
                lines = ef.readlines()
            print("  Last stderr:")
            for line in lines[-10:]:
                print("   ", line, end="")
    except subprocess.TimeoutExpired:
        print(f"  Route {route_id} timed out.")
    except Exception as e:
        print(f"  Route {route_id} error: {e}")
    finally:
        try:
            os.killpg(os.getpgid(carla_proc.pid), signal.SIGKILL)
        except Exception:
            pass
        carla_proc.wait()
        if carla_proc in _carla_procs:
            _carla_procs.remove(carla_proc)

    return success


def main():
    cfg = config
    out_root = cfg["out_root"]

    if cfg.get("route_file"):
        route_xmls = [cfg["route_file"]]
    else:
        route_path = cfg["route_path"]
        route_xmls = sorted(os.path.join(route_path, x)
                            for x in os.listdir(route_path) if x.endswith(".xml"))

    job_queue = []
    for route_xml in route_xmls:
        fname = os.path.basename(route_xml)
        route_id    = os.path.splitext(fname)[0]
        save_path   = os.path.join(out_root, "data", route_id)
        result_file = os.path.join(out_root, "results", f"{route_id}_res.json")
        log_file    = os.path.join(out_root, "logs", f"{route_id}_out.log")
        err_file    = os.path.join(out_root, "logs", f"{route_id}_err.log")
        job_queue.append({
            "route_xml": route_xml, "route_id": route_id,
            "save_path": save_path, "result_file": result_file,
            "log_file": log_file, "err_file": err_file,
            "tries": cfg["tries"],
        })

    total = len(job_queue)
    print(f"\nCollecting {total} routes → {out_root}")
    if cfg.get("tmp_visu"):
        print("  TMP_VISU=1: BEV+RGB visualization frames will be saved alongside sensor data")
    progress = tqdm(total=total)

    while job_queue:
        job = job_queue.pop(0)

        if is_done(job["result_file"]):
            progress.update(1)
            continue

        print(f"\n[{total - len(job_queue)}/{total}] Route {job['route_id']} (tries left: {job['tries']})")
        run_route(cfg, job["route_xml"], job["route_id"],
                  job["save_path"], job["result_file"],
                  job["log_file"], job["err_file"])

        if is_done(job["result_file"]):
            progress.update(1)
            print(f"  Saved to: {job['save_path']}")
        elif job["tries"] > 1:
            job["tries"] -= 1
            job_queue.append(job)
            print(f"  Will retry ({job['tries']} tries left)")
        else:
            print(f"  Route {job['route_id']} failed after all retries.")
            progress.update(1)

    progress.close()
    print(f"\nDone. Data in: {out_root}/data/")
    print(f"       Results: {out_root}/results/")


if __name__ == "__main__":
    main()
