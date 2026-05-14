"""
Collect training data using PDM-lite expert on Fail2Drive long-tail routes.

Each route XML is run as a separate SLURM job.
Monitors progress and resubmits crashed/failed jobs.
Best run inside a tmux terminal.

Usage:
    conda activate simlingo_f2d
    cd /path/to/simlingo_f2d
    python collect_dataset_f2d.py
"""

from datetime import datetime
import os
import subprocess
import time
import glob
import json
from pathlib import Path
import random
import re
import xml.etree.ElementTree as ET


def get_town_from_xml(route_xml_path):
    """Extract town name from a fail2drive route XML file."""
    try:
        tree = ET.parse(route_xml_path)
        for route in tree.iter("route"):
            return route.attrib.get("town", "Town12")
    except Exception:
        return "Town12"


def make_bash(code_dir, route_file_number, agent_name, route_file, checkpoint_endpoint, save_pth, seed, carla_root, town, repetition):
    save_slurm = save_pth.replace("data/", "slurm/")

    jobfile = f"{save_slurm}/run_files/start_files/{route_file_number}_Rep{repetition}.sh"
    Path(jobfile).parent.mkdir(parents=True, exist_ok=True)

    run_command = (
        "python leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py "
        "--port=${FREE_WORLD_PORT} "
        "--traffic-manager-port=${TM_PORT} --traffic-manager-seed=${TM_SEED} "
        "--routes=${ROUTES} --repetitions=${REPETITIONS} "
        "--track=${CHALLENGE_TRACK_CODENAME} --checkpoint=${CHECKPOINT_ENDPOINT} "
        "--agent=${TEAM_AGENT} --agent-config=${TEAM_CONFIG} "
        "--debug=0 --resume=${RESUME} --timeout=600"
    )

    qsub_template = f"""#!/bin/bash
export SCENARIO_RUNNER_ROOT={code_dir}/scenario_runner_autopilot
export LEADERBOARD_ROOT={code_dir}/leaderboard_autopilot

# CARLA 0.9.15 (Python 3.10)
export CARLA_ROOT={carla_root}
export CARLA_SERVER={carla_root}/CarlaUE4.sh
export PYTHONPATH={carla_root}/PythonAPI/carla:{code_dir}/leaderboard_autopilot:{code_dir}/scenario_runner_autopilot
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

export REPETITIONS=1
export DEBUG_CHALLENGE=0
export TEAM_AGENT={agent_name}
export CHALLENGE_TRACK_CODENAME=MAP
export ROUTES={route_file}
export TOWN={town}
export REPETITION={repetition}
export TM_SEED={seed}

export CHECKPOINT_ENDPOINT={checkpoint_endpoint}
export TEAM_CONFIG={route_file}
export RESUME=1
export DATAGEN=1
export SAVE_PATH={save_pth}

echo "Start python"

export FREE_STREAMING_PORT=$1
export FREE_WORLD_PORT=$2
export TM_PORT=$3

echo "FREE_STREAMING_PORT: $FREE_STREAMING_PORT"
echo "FREE_WORLD_PORT: $FREE_WORLD_PORT"
echo "TM_PORT: $TM_PORT"

bash {carla_root}/CarlaUE4.sh --world-port=$FREE_WORLD_PORT -RenderOffScreen -nosound -graphicsadapter=0 -carla-streaming-port=$FREE_STREAMING_PORT &

sleep 180

cd {code_dir}
{run_command}
"""

    with open(jobfile, "w", encoding="utf-8") as f:
        f.write(qsub_template)
    return jobfile


def get_running_jobs(jobname, user_name):
    job_list = subprocess.check_output(
        (
            f"SQUEUE_FORMAT2='jobid:10,username:{len(user_name)},name:130' squeue --sort V | grep {user_name} | "
            f"grep {jobname} || true"
        ),
        shell=True,
    ).decode("utf-8").splitlines()
    currently_num_running_jobs = len(job_list)
    routefile_number_list = [line.split("_")[-2] + "_" + line.split("_")[-1].strip() for line in job_list]
    pid_list = [line.split(" ")[0] for line in job_list]
    return currently_num_running_jobs, routefile_number_list, pid_list


def get_last_line_from_file(filepath):
    try:
        with open(filepath, "rb") as f:
            try:
                f.seek(-2, os.SEEK_END)
                while f.read(1) != b"\n":
                    f.seek(-2, os.SEEK_CUR)
            except OSError:
                f.seek(0)
            last_line = f.readline().decode()
    except Exception:
        last_line = ""
    return last_line


def cancel_jobs_with_err_in_log(logroot, jobname, user_name):
    print("Checking logs for errors...")
    _, routefile_number_list, pid_list = get_running_jobs(jobname, user_name)
    for i, rf_num in enumerate(routefile_number_list):
        logfile_path = os.path.join(logroot, f"run_files/logs/qsub_out{rf_num}.log")
        last_line = get_last_line_from_file(logfile_path)
        terminate = False
        if "Actor" in last_line and "not found!" in last_line:
            terminate = True
        if "Watchdog exception - Timeout" in last_line:
            terminate = True
        if "Engine crash handling finished; re-raising signal 11" in last_line:
            terminate = True
        if terminate:
            print(f"Terminating route {rf_num} with pid {pid_list[i]} due to error in logfile.")
            subprocess.check_output(f"scancel {pid_list[i]}", shell=True)


def wait_for_jobs_to_finish(logroot, jobname, user_name, max_n_parallel_jobs):
    currently_running_jobs, _, _ = get_running_jobs(jobname, user_name)
    print(f"{currently_running_jobs}/{max_n_parallel_jobs} jobs are running...")
    counter = 0
    while currently_running_jobs >= max_n_parallel_jobs:
        if counter == 0:
            cancel_jobs_with_err_in_log(logroot, jobname, user_name)
        time.sleep(5)
        currently_running_jobs, _, _ = get_running_jobs(jobname, user_name)
        counter = (counter + 1) % 4


def get_num_jobs(job_name, username):
    len_usrn = len(username)
    num_running_jobs = int(
        subprocess.check_output(
            f"SQUEUE_FORMAT2='username:{len_usrn},name:130' squeue --sort V | grep {username} | grep {job_name} | wc -l",
            shell=True,
        ).decode('utf-8').replace('\n', ''))
    try:
        with open('max_num_jobs.txt', 'r', encoding='utf-8') as f:
            max_num_parallel_jobs = int(f.read())
    except Exception:
        max_num_parallel_jobs = 1
    return num_running_jobs, max_num_parallel_jobs


def get_which_partition(default):
    try:
        with open('partition.txt', 'r', encoding='utf-8') as f:
            partition_name = f.read().strip()
    except Exception:
        print("partition.txt not found. Using default partition.")
        partition_name = default
    return partition_name


def make_jobsub_file(save_path_data, jobname, route_file_number, partition_name, repetition, timeout="0-04:00"):
    save_slurm = save_path_data.replace("data/", "slurm/")
    os.makedirs(f"{save_slurm}/run_files/logs", exist_ok=True)
    os.makedirs(f"{save_slurm}/run_files/job_files", exist_ok=True)
    os.makedirs(f"{save_slurm}/run_files/start_files", exist_ok=True)
    jobfile = f"{save_slurm}/run_files/job_files/{route_file_number}_Rep{repetition}.sh"
    qsub_template = f"""#!/bin/bash
#SBATCH --job-name={jobname}_{route_file_number}
#SBATCH --partition={partition_name}
#SBATCH -o {save_slurm}/run_files/logs/qsub_out{route_file_number}.log
#SBATCH -e {save_slurm}/run_files/logs/qsub_err{route_file_number}.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50gb
#SBATCH --time={timeout}
#SBATCH --gres=gpu:1

echo "SLURMD_NODENAME: $SLURMD_NODENAME"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
scontrol show job $SLURM_JOB_ID

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "Job started: $dt"

source ~/.bashrc
. ~/anaconda3/etc/profile.d/conda.sh
conda activate simlingo_f2d

export FREE_STREAMING_PORT=`comm -23 <(seq 10000 10400 | sort) <(ss -Htan | awk \'{{print $4}}\' | cut -d\':\' -f2 | sort -u) | shuf | head -n 1`
export FREE_WORLD_PORT=`comm -23 <(seq 20000 20400 | sort) <(ss -Htan | awk \'{{print $4}}\' | cut -d\':\' -f2 | sort -u) | shuf | head -n 1`
export TM_PORT=`comm -23 <(seq 30000 30400 | sort) <(ss -Htan | awk '{{print $4}}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`

sleep 2

echo "start python"
pwd
bash {save_slurm}/run_files/start_files/{route_file_number}_Rep{repetition}.sh $FREE_STREAMING_PORT $FREE_WORLD_PORT $TM_PORT
"""
    with open(jobfile, "w", encoding="utf-8") as f:
        f.write(qsub_template)
    return jobfile


if __name__ == "__main__":
    repetitions = 1
    repetition_start = 0
    default_partition = "YOUR_PARTITION"   # TODO: change to your SLURM partition
    job_name = "collect_f2d"
    username = "YOUR_USERNAME"              # TODO: change to your username
    code_root = "/path/to/simlingo_f2d"    # TODO: absolute path to this repo
    carla_root = "/path/to/CARLA_0915"     # TODO: path to CARLA 0.9.15 installation

    date = datetime.today().strftime("%Y_%m_%d")
    dataset_name = "simlingo_f2d_" + date
    root_folder = "database/"
    data_save_directory = root_folder + dataset_name
    log_root = f"{data_save_directory}/slurm"

    # --- Route sources ---
    # Official fail2drive routes (100 route XMLs for 17 scenario categories)
    route_folder_official = f"{code_root}/leaderboard/data/fail2drive_split"
    # Custom routes you created with the fail2drive toolbox
    route_folder_custom = f"{code_root}/leaderboard/data/fail2drive_customized"

    routes = glob.glob(f"{route_folder_official}/*.xml")
    routes += glob.glob(f"{route_folder_custom}/*.xml")

    port_offset = 0
    job_number = 1
    meta_jobs = {}

    random.seed(42)
    random.shuffle(routes)
    seed_counter = 1000000 * repetition_start - 1

    num_routes = len(routes)
    print(f"Found {num_routes} routes to collect data for.")

    for repetition in range(repetition_start, repetitions):
        for route in routes:
            seed_counter += 1

            # Town is embedded in the XML, not the filename for fail2drive routes
            town = get_town_from_xml(route)

            # Route identifier: Base_Category_XXXX.xml → XXXX
            routefile_number = Path(route).stem  # e.g. "Base_Animals_0075"

            ckpt_endpoint = f"{code_root}/{data_save_directory}/results/{routefile_number}_result.json"
            save_path = f"{code_root}/{data_save_directory}/data/{routefile_number}"
            Path(save_path).mkdir(parents=True, exist_ok=True)

            agent = f"{code_root}/team_code/data_agent.py"
            partition_name = get_which_partition(default_partition)

            bash_file = make_bash(
                code_root, routefile_number, agent, route,
                ckpt_endpoint, save_path, seed_counter, carla_root, town, repetition
            )
            job_file = make_jobsub_file(save_path, job_name, routefile_number, partition_name, repetition, "0-04:00")

            num_running_jobs, max_num_parallel_jobs = get_num_jobs(job_name=job_name, username=username)
            print(f'{num_running_jobs}/{max_num_parallel_jobs} jobs are running...')
            while num_running_jobs >= max_num_parallel_jobs:
                num_running_jobs, max_num_parallel_jobs = get_num_jobs(job_name=job_name, username=username)
                time.sleep(0.05)

            print(f"Submitting job {job_number}/{num_routes}: {job_name}_{routefile_number}. ", end="")
            time.sleep(1)
            jobid = subprocess.check_output(f"sbatch {job_file}", shell=True).decode("utf-8") \
                                                            .strip().rsplit(" ", maxsplit=1)[-1]
            print(f"Jobid: {jobid}")
            meta_jobs[jobid] = (False, job_file, ckpt_endpoint, 0)
            job_number += 1

    time.sleep(1)
    training_finished = False
    while not training_finished:
        num_running_jobs, _, _ = get_running_jobs(job_name, username)
        print(f"{num_running_jobs} jobs are running... Job: {job_name}")
        cancel_jobs_with_err_in_log(log_root, job_name, username)
        time.sleep(20)

        for k in list(meta_jobs.keys()):
            job_finished, job_file, result_file, resubmitted = meta_jobs[k]
            need_to_resubmit = False
            if not job_finished and resubmitted < 3:
                if int(subprocess.check_output(f"squeue | grep {k} | wc -l", shell=True).decode("utf-8").strip()) == 0:
                    if os.path.exists(result_file):
                        with open(result_file, "r", encoding="utf-8") as f_result:
                            evaluation_data = json.load(f_result)
                        progress = evaluation_data["_checkpoint"]["progress"]
                        if len(progress) < 2 or progress[0] < progress[1]:
                            need_to_resubmit = True
                        else:
                            for record in evaluation_data["_checkpoint"]["records"]:
                                if record["scores"]["score_route"] <= 1e-10:
                                    need_to_resubmit = True
                                if record["status"] in (
                                    "Failed - Agent couldn't be set up",
                                    "Failed",
                                    "Failed - Simulation crashed",
                                    "Failed - Agent crashed",
                                ):
                                    need_to_resubmit = True
                        if not need_to_resubmit:
                            print(f"Finished job {job_file}")
                            meta_jobs[k] = (True, None, None, 0)
                    else:
                        need_to_resubmit = True

            if need_to_resubmit:
                routefile_number = Path(job_file).stem
                print(f"Resubmit job {routefile_number} (previous id: {k}). Waiting for jobs to finish...")
                with open('max_num_jobs.txt', 'r', encoding='utf-8') as f:
                    max_num_parallel_jobs = int(f.read())
                wait_for_jobs_to_finish(log_root, job_name, username, max_num_parallel_jobs)

                time_now_log = time.time()
                os.system(f'mkdir -p "{log_root}/run_files/logs_{routefile_number}_{time_now_log}"')
                os.system(
                    f"cp {log_root}/run_files/logs/qsub_err{routefile_number}.log "
                    f"{log_root}/run_files/logs_{routefile_number}_{time_now_log}/"
                )
                os.system(
                    f"cp {log_root}/run_files/logs/qsub_out{routefile_number}.log "
                    f"{log_root}/run_files/logs_{routefile_number}_{time_now_log}/"
                )

                jobid = subprocess.check_output(f"sbatch {job_file}", shell=True).decode("utf-8").strip() \
                                                                                .rsplit(" ", maxsplit=1)[-1]
                meta_jobs[jobid] = (False, job_file, result_file, resubmitted + 1)
                meta_jobs[k] = (True, None, None, 0)
                print(f"Resubmitted job {routefile_number}. (new id: {jobid})")

        time.sleep(10)

        if num_running_jobs == 0:
            training_finished = True
