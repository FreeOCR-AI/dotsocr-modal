import base64
import os
from typing import Annotated

import typer
from openai import OpenAI

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


def encode_image(image_path: str):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


@app.command()
def main(
    image_path: str = typer.Argument(..., help="Path to the image file"),
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
    client = OpenAI(
        base_url=url,
        api_key=key,
    )

    base64_image = encode_image(image_path)

    response = client.chat.completions.create(
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

    typer.echo(response.choices[0].message.content)


if __name__ == "__main__":
    app()
