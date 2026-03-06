"""
Test NCCL weight sync with SHARED CUDA_VISIBLE_DEVICES.
All processes see all GPUs.  vLLM is told to start from a specific GPU rank.

Usage:
    python tmp-scripts/test_nccl_shared_vis.py \
        --trainer-gpu 0 --vllm-start-gpu 2 --n-dp 2
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
    parser.add_argument("--trainer-gpu", type=int, default=0)
    parser.add_argument("--vllm-start-gpu", type=int, default=2)
    parser.add_argument("--n-dp", type=int, default=2)
    args = parser.parse_args()

    print(f"Trainer GPU:      {args.trainer_gpu}")
    print(f"vLLM start GPU:   {args.vllm_start_gpu}")
    print(f"vLLM DP workers:  {args.n_dp}")
    print(f"All GPUs visible to all processes (no CUDA_VISIBLE_DEVICES split)")

    # 1. Port
    with socket.socket() as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    url = f"http://localhost:{port}"

    # 2. Start vLLM — ALL GPUs visible, use --data-parallel-start-rank
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
        "--data-parallel-size", str(args.n_dp),
        "--data-parallel-start-rank", str(args.vllm_start_gpu),
    ]
    vllm_env = os.environ.copy()
    # NO CUDA_VISIBLE_DEVICES override — all GPUs visible
    vllm_env["VLLM_SERVER_DEV_MODE"] = "1"
    vllm_env["NCCL_DEBUG"] = "WARN"

    print(f"Starting vLLM (port {port})...")
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

        ws = requests.get(f"{url}/get_world_size", timeout=10).json()["world_size"]
        print(f"vLLM world_size = {ws}")
        nccl_world_size = 1 + ws

        # 3. Load model on trainer GPU
        torch.cuda.set_device(args.trainer_gpu)
        device = f"cuda:{args.trainer_gpu}"

        print(f"Loading model on {device}...")
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
        model.to(device)
        print("Model loaded")

        # 4. NCCL weight transfer init
        from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine

        with socket.socket() as s:
            s.bind(("", 0))
            nccl_port = s.getsockname()[1]

        master_addr = "127.0.0.1"

        print(f"NCCL init: master={master_addr}:{nccl_port}, world_size={nccl_world_size}")

        init_error = [None]

        def init_vllm():
            try:
                r = requests.post(
                    f"{url}/init_weight_transfer_engine",
                    json={"init_info": {
                        "master_address": master_addr,
                        "master_port": nccl_port,
                        "rank_offset": 1,
                        "world_size": nccl_world_size,
                    }},
                    timeout=300,
                )
                r.raise_for_status()
                print(">>> vLLM init: OK")
            except Exception as e:
                init_error[0] = e
                print(f">>> vLLM init FAILED: {e}")

        init_thread = threading.Thread(target=init_vllm)
        init_thread.start()

        print("Calling trainer_init()...")
        t0 = time.time()
        group = NCCLWeightTransferEngine.trainer_init(
            dict(master_address=master_addr, master_port=nccl_port,
                 world_size=nccl_world_size),
        )
        print(f">>> trainer_init: OK ({time.time()-t0:.1f}s)")

        init_thread.join()
        if init_error[0]:
            print(f"Init failed: {init_error[0]}")
            return 1

        # 5. Sync weights
        requests.post(f"{url}/pause", timeout=30).raise_for_status()

        names, dtype_names, shapes = [], [], []
        for n, p in model.named_parameters():
            names.append(n)
            dtype_names.append(str(p.dtype).split(".")[-1])
            shapes.append(list(p.shape))

        def do_update():
            try:
                requests.post(
                    f"{url}/update_weights",
                    json={"update_info": {
                        "names": names, "dtype_names": dtype_names,
                        "shapes": shapes, "packed": True,
                    }}, timeout=300).raise_for_status()
                print(">>> update_weights: OK")
            except Exception as e:
                print(f">>> update_weights FAILED: {e}")

        ut = threading.Thread(target=do_update)
        ut.start()

        t0 = time.time()
        NCCLWeightTransferEngine.trainer_send_weights(
            iterator=model.named_parameters(), group=group, packed=True)
        print(f">>> send_weights: OK ({time.time()-t0:.1f}s)")
        ut.join()

        requests.post(f"{url}/resume", timeout=30).raise_for_status()

        # 6. Test
        from openai import OpenAI
        client = OpenAI(base_url=f"{url}/v1", api_key="EMPTY")
        resp = client.completions.create(
            model="policy", prompt="Hello, my name is", max_tokens=20, temperature=0)
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
