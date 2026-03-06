"""
Test NCCL weight sync with separate CUDA_VISIBLE_DEVICES and DP,
mimicking the real training setup.

Usage:
    python tmp-scripts/test_nccl_dp.py [--vllm-gpus 2,3] [--trainer-gpu 0]
"""
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import argparse

import requests
import torch
from transformers import AutoModelForCausalLM

MODEL = "Qwen/Qwen2-1.5B-Instruct"


def wait_for_health(url, timeout=180):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{url}/health", timeout=5)
            if r.status_code == 200:
                r2 = requests.get(f"{url}/v1/models", timeout=5)
                if r2.status_code == 200 and r2.json().get("data"):
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-gpus", default="2,3", help="GPUs for vLLM")
    parser.add_argument("--trainer-gpu", default="0", help="GPU for trainer")
    args = parser.parse_args()

    vllm_gpus = args.vllm_gpus
    trainer_gpu = args.trainer_gpu
    n_dp = len(vllm_gpus.split(","))

    print(f"Trainer GPU: {trainer_gpu}")
    print(f"vLLM GPUs:   {vllm_gpus} ({n_dp} DP workers)")

    # 1. Find free port
    with socket.socket() as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    url = f"http://localhost:{port}"

    # 2. Start vLLM
    wt_config = json.dumps({"backend": "nccl"})
    vllm_cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
        "--served-model-name", "policy",
        "--port", str(port),
        "--gpu-memory-utilization", "0.5",
        "--max-model-len", "512",
        "--dtype", "float16",
        "--load-format", "dummy",
        "--weight-transfer-config", wt_config,
        "--disable-log-requests",
        "--data-parallel-size", str(n_dp),
    ]
    vllm_env = os.environ.copy()
    vllm_env["CUDA_VISIBLE_DEVICES"] = vllm_gpus
    vllm_env["VLLM_SERVER_DEV_MODE"] = "1"
    vllm_env["NCCL_P2P_DISABLE"] = "1"
    vllm_env["NCCL_DEBUG"] = "INFO"

    print(f"Starting vLLM on GPUs {vllm_gpus} (port {port})...")
    vllm_proc = subprocess.Popen(
        vllm_cmd, env=vllm_env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )

    def stream_output():
        for line in vllm_proc.stdout:
            sys.stdout.write(f"[VLLM] {line.decode(errors='replace')}")
            sys.stdout.flush()

    t = threading.Thread(target=stream_output, daemon=True)
    t.start()

    try:
        if not wait_for_health(url, timeout=180):
            print("FAILED: vLLM didn't start")
            return 1

        print(f"vLLM ready at {url}")

        # 3. Query world size
        ws = requests.get(f"{url}/get_world_size", timeout=10).json()["world_size"]
        print(f"vLLM world_size = {ws}")
        nccl_world_size = 1 + ws

        # 4. Load model on trainer GPU
        # Set CUDA_VISIBLE_DEVICES to just the trainer GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = trainer_gpu
        torch.cuda.set_device(0)

        print(f"Loading model on physical GPU {trainer_gpu} (cuda:0)...")
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
        model.to("cuda:0")
        print("Model loaded")

        # 5. Init NCCL weight transfer
        from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine

        with socket.socket() as s:
            s.bind(("", 0))
            nccl_port = s.getsockname()[1]

        master_addr = "127.0.0.1"
        rank_offset = 1

        # Disable P2P on trainer side too
        os.environ["NCCL_P2P_DISABLE"] = "1"
        os.environ["NCCL_DEBUG"] = "INFO"

        print(f"NCCL init: master={master_addr}:{nccl_port}, world_size={nccl_world_size}")
        print(f"  trainer: rank=0, device=cuda:0 (physical GPU {trainer_gpu})")
        print(f"  vLLM:    ranks 1-{ws}, DP workers on GPUs {vllm_gpus}")

        init_error = [None]

        def init_vllm():
            try:
                r = requests.post(
                    f"{url}/init_weight_transfer_engine",
                    json={"init_info": {
                        "master_address": master_addr,
                        "master_port": nccl_port,
                        "rank_offset": rank_offset,
                        "world_size": nccl_world_size,
                    }},
                    timeout=300,
                )
                r.raise_for_status()
                print(">>> vLLM init_weight_transfer_engine: OK")
            except Exception as e:
                init_error[0] = e
                print(f">>> vLLM init_weight_transfer_engine FAILED: {e}")

        init_thread = threading.Thread(target=init_vllm)
        init_thread.start()

        print("Calling trainer_init()...")
        t0 = time.time()
        try:
            group = NCCLWeightTransferEngine.trainer_init(
                dict(
                    master_address=master_addr,
                    master_port=nccl_port,
                    world_size=nccl_world_size,
                ),
            )
            print(f">>> trainer_init: OK ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f">>> trainer_init FAILED ({time.time()-t0:.1f}s): {e}")
            init_thread.join(timeout=10)
            return 1

        init_thread.join()
        if init_error[0]:
            print(f"Init failed on vLLM side: {init_error[0]}")
            return 1

        # 6. Sync weights
        print("Pausing vLLM...")
        requests.post(f"{url}/pause", timeout=30).raise_for_status()

        names, dtype_names, shapes = [], [], []
        for name, p in model.named_parameters():
            names.append(name)
            dtype_names.append(str(p.dtype).split(".")[-1])
            shapes.append(list(p.shape))

        def do_update():
            try:
                r = requests.post(
                    f"{url}/update_weights",
                    json={"update_info": {
                        "names": names, "dtype_names": dtype_names,
                        "shapes": shapes, "packed": True,
                    }},
                    timeout=300,
                )
                r.raise_for_status()
                print(">>> update_weights: OK")
            except Exception as e:
                print(f">>> update_weights FAILED: {e}")

        update_thread = threading.Thread(target=do_update)
        update_thread.start()

        print("Broadcasting weights via NCCL...")
        t0 = time.time()
        NCCLWeightTransferEngine.trainer_send_weights(
            iterator=model.named_parameters(),
            group=group,
            packed=True,
        )
        print(f">>> trainer_send_weights: OK ({time.time()-t0:.1f}s)")
        update_thread.join()

        print("Resuming vLLM...")
        requests.post(f"{url}/resume", timeout=30).raise_for_status()

        # 7. Test
        print("\nTesting generation...")
        from openai import OpenAI
        client = OpenAI(base_url=f"{url}/v1", api_key="EMPTY")
        resp = client.completions.create(
            model="policy", prompt="Hello, my name is", max_tokens=20, temperature=0,
        )
        print(f"Generated: {resp.choices[0].text!r}")
        print("\nSUCCESS!")
        return 0

    finally:
        vllm_proc.send_signal(signal.SIGTERM)
        try:
            vllm_proc.wait(timeout=15)
        except Exception:
            vllm_proc.kill()


if __name__ == "__main__":
    sys.exit(main())
