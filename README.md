# Dots OCR Modal Inference

This project provides a setup to run the [rednote-hilab/dots.ocr](https://huggingface.co/rednote-hilab/dots.ocr) model on [Modal](https://modal.com/) using vLLM, along with a client script to test the inference.

## Prerequisites

Ensure you have Python installed and the necessary packages:

```bash
pip install modal openai typer
```

You also need to set up Modal locally:

```bash
modal setup
```

## Deployment

The inference server is defined in `vllm-inference.py`. It uses an NVIDIA A10 GPU to serve the model.

To deploy the server to Modal:

```bash
modal deploy vllm-inference.py
```

After deploying, Modal will output a URL for the web server (e.g., `https://your-app-url.modal.run`).

## Usage

Use the `test.py` script to send an image to your deployed server and extract layout information (bboxes, categories, text).

### Command Syntax

```bash
python test.py <path_to_image> --url <MODAL_URL>/v1
```

> **Note**: You must append `/v1` to the URL provided by Modal because the vLLM server expects standard OpenAI-compatible endpoints.

## Configuration

The following constants in `vllm-inference.py` can be adjusted to tune performance and resource usage:

| Constant                | Default Value | Description                                                                     |
| :---------------------- | :------------ | :------------------------------------------------------------------------------ |
| `GPU_CONFIG`            | `"A10:1"`     | The GPU type and count used for inference.                                      |
| `MAX_CONCURRENT_INPUTS` | `16`          | Maximum concurrent requests per replica. Tune based on VRAM usage.              |
| `FAST_BOOT`             | `True`        | If `True`, enables `--enforce-eager` for faster startup (disables CUDA graphs). |
