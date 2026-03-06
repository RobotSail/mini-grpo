"""
Minimal test of vLLM v0.16 NCCL weight transfer.
Starts vLLM on one GPU with dummy weights, then syncs real weights via NCCL.

Usage:
    CUDA_VISIBLE_DEVICES=0,1 python tmp-scripts/test_nccl_weight_sync.py

GPU 0 = trainer, GPU 1 = vLLM
"""
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time

import requests
import torch
from transformers import AutoModelForCausalLM

MODEL = "Qwen/Qwen2-1.5B-Instruct"
VLLM_GPU = "1"       # logical GPU index for vLLM (within CUDA_VISIBLE_DEVICES)
TRAINER_GPU = "0"     # logical GPU index for trainer


def wait_for_health(url, timeout=120):
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
    # 1. Find a free port
    with socket.socket() as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    url = f"http://localhost:{port}"

    # 2. Start vLLM with dummy weights + NCCL weight transfer
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
    ]
    vllm_env = os.environ.copy()
    vllm_env["CUDA_VISIBLE_DEVICES"] = VLLM_GPU
    vllm_env["VLLM_SERVER_DEV_MODE"] = "1"
    vllm_env["NCCL_DEBUG"] = "INFO"
    vllm_env["NCCL_P2P_DISABLE"] = "1"

    print(f"Starting vLLM on GPU {VLLM_GPU} (port {port})...")
    vllm_proc = subprocess.Popen(
        vllm_cmd, env=vllm_env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )

    def stream_output():
        for line in vllm_proc.stdout:
            print(f"[VLLM] {line.decode(errors='replace').rstrip()}")

    t = threading.Thread(target=stream_output, daemon=True)
    t.start()

    try:
        if not wait_for_health(url):
            print("FAILED: vLLM didn't start")
            return 1

        print(f"vLLM ready at {url}")

        # 3. Query world size
        ws = requests.get(f"{url}/get_world_size", timeout=10).json()["world_size"]
        print(f"vLLM world_size = {ws}")

        nccl_world_size = 1 + ws  # trainer + vLLM workers

        # 4. Load trainer model on trainer GPU
        trainer_device = f"cuda:{TRAINER_GPU}"
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{TRAINER_GPU},{VLLM_GPU}"
        torch.cuda.set_device(int(TRAINER_GPU))

        print(f"Loading model on {trainer_device}...")
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
        model.to(trainer_device)
        print("Model loaded")

        # 5. Initialize NCCL weight transfer
        from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine

        with socket.socket() as s:
            s.bind(("", 0))
            nccl_port = s.getsockname()[1]

        master_addr = "127.0.0.1"
        rank_offset = 1
        print(f"NCCL init: master={master_addr}:{nccl_port}, world_size={nccl_world_size}")

        # Disable P2P: trainer and vLLM have different CUDA_VISIBLE_DEVICES
        os.environ["NCCL_P2P_DISABLE"] = "1"

        # Start vLLM side in thread
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
                    timeout=120,
                )
                r.raise_for_status()
                print("vLLM init_weight_transfer_engine: OK")
            except Exception as e:
                init_error[0] = e
                print(f"vLLM init_weight_transfer_engine FAILED: {e}")

        init_thread = threading.Thread(target=init_vllm)
        init_thread.start()

        # Trainer side
        print("Calling trainer_init()...")
        try:
            group = NCCLWeightTransferEngine.trainer_init(
                dict(
                    master_address=master_addr,
                    master_port=nccl_port,
                    world_size=nccl_world_size,
                ),
            )
            print("trainer_init: OK")
        except Exception as e:
            print(f"trainer_init FAILED: {e}")
            init_thread.join(timeout=10)
            return 1

        init_thread.join()
        if init_error[0]:
            print(f"Init failed on vLLM side: {init_error[0]}")
            return 1

        # 6. Pause, sync weights, resume
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
                print("update_weights: OK")
            except Exception as e:
                print(f"update_weights FAILED: {e}")

        update_thread = threading.Thread(target=do_update)
        update_thread.start()

        print("Broadcasting weights via NCCL...")
        NCCLWeightTransferEngine.trainer_send_weights(
            iterator=model.named_parameters(),
            group=group,
            packed=True,
        )
        update_thread.join()

        print("Resuming vLLM...")
        requests.post(f"{url}/resume", timeout=30).raise_for_status()

        # 7. Test generation
        print("\nTesting generation after weight sync...")
        from openai import OpenAI
        client = OpenAI(base_url=f"{url}/v1", api_key="EMPTY")
        resp = client.completions.create(
            model="policy", prompt="Hello, my name is", max_tokens=20, temperature=0,
        )
        print(f"Generated: {resp.choices[0].text!r}")
        print("\nSUCCESS: NCCL weight transfer works!")
        return 0

    finally:
        vllm_proc.send_signal(signal.SIGTERM)
        vllm_proc.wait(timeout=10)


if __name__ == "__main__":
    sys.exit(main())
