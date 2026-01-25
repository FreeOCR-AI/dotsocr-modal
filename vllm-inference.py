import modal

# configuration for the vLLM server
GPU_CONFIG = "A10:1"
MAX_CONCURRENT_INPUTS = 16  # how many requests can one replica handle? tune carefully!

# constants
MINUTES = 60  # seconds
VLLM_PORT = 8000
MODEL_NAME = "rednote-hilab/dots.ocr"
MODEL_REVISION = "c0111ce6bc07803dbc267932ffef0ae3a51dc951"  # avoid nasty surprises when repos update!

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)


vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.13.0",
        "huggingface-hub==0.36.0",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})  # faster model transfers
)


app = modal.App("dots-ocr")


@app.function(
    image=vllm_image,
    gpu=GPU_CONFIG,  # 24G
    scaledown_window=15 * MINUTES,  # how long should we stay up with no requests?
    timeout=10 * MINUTES,  # how long should we wait for container start?
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(max_inputs=MAX_CONCURRENT_INPUTS)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve():
    import subprocess

    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        MODEL_NAME,
        "--revision",
        MODEL_REVISION,
        "--served-model-name",
        MODEL_NAME,
        "--trust-remote-code",
        "--async-scheduling",
        "--gpu-memory-utilization",
        "0.95",
        "--tensor-parallel-size",
        "1",
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
    ]

    print(*cmd)

    subprocess.Popen(" ".join(cmd), shell=True)
