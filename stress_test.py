import asyncio
import base64
import os
import random
import time
from dataclasses import dataclass
from typing import Annotated, List

import typer
from openai import AsyncOpenAI

app = typer.Typer()

DEFAULT_PROMPT = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
"""


@dataclass
class Stats:
    completed: int = 0
    total_time: float = 0.0
    total_tokens: int = 0
    lock: asyncio.Lock = None


def encode_image(image_path: str):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


async def worker(
    worker_id: int,
    client: AsyncOpenAI,
    model: str,
    image_paths: List[str],
    stats: Stats,
):
    while True:
        image_path = random.choice(image_paths)
        start_time = time.time()

        try:
            # Offload file reading to avoid blocking the event loop
            base64_image = await asyncio.to_thread(encode_image, image_path)

            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": DEFAULT_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                },
                            },
                        ],
                    }
                ],
                temperature=0.1,
                max_tokens=16000,
            )

            duration = time.time() - start_time
            usage = response.usage
            tokens = usage.completion_tokens if usage else 0

            async with stats.lock:
                stats.completed += 1
                stats.total_time += duration
                stats.total_tokens += tokens

                avg_time = stats.total_time / stats.completed
                # Clear line or just print
                print(
                    f"Tasks: {stats.completed} | Avg Time: {avg_time:.2f}s | Total Tokens: {stats.total_tokens} | Last Request: {duration:.2f}s (Worker {worker_id})"
                )

        except Exception as e:
            print(f"Error in worker {worker_id}: {e}")
            await asyncio.sleep(1)  # Simple backoff


async def run_stress_test(
    image_folder: str, concurrency: int, url: str, key: str, model: str
):
    # Collect images
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
    image_paths = []
    for root, dirs, files in os.walk(image_folder):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_paths.append(os.path.join(root, file))

    if not image_paths:
        print(f"No images found in {image_folder}")
        return

    print(
        f"Found {len(image_paths)} images. Starting stress test with concurrency {concurrency}..."
    )
    print(f"Target URL: {url}")
    print(f"Model: {model}")

    client = AsyncOpenAI(base_url=url, api_key=key)
    stats = Stats(lock=asyncio.Lock())

    tasks = [
        asyncio.create_task(worker(i, client, model, image_paths, stats))
        for i in range(concurrency)
    ]

    try:
        # Wait indefinitely
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        print("\nStopping stress test...")


@app.command()
def main(
    folder: str = typer.Argument(..., help="Folder containing images"),
    concurrency: int = typer.Option(
        1, "--concurrency", "-c", help="Number of concurrent requests"
    ),
    url: Annotated[str, typer.Option(help="Base URL for the API")] = os.getenv(
        "OPENAI_BASE_URL", "https://api.openai.com/v1"
    ),
    key: Annotated[str, typer.Option(help="API Key")] = os.getenv(
        "OPENAI_API_KEY", "no-key"
    ),
    model: Annotated[
        str, typer.Option(help="Model name to use")
    ] = "rednote-hilab/dots.ocr",
):
    """
    Stress test the OCR API with concurrent requests.
    """
    try:
        asyncio.run(run_stress_test(folder, concurrency, url, key, model))
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")


if __name__ == "__main__":
    app()
