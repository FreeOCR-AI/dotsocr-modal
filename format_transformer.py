import re


def has_latex_markdown(text: str) -> bool:
    """
    Checks if a string contains LaTeX markdown patterns.

    Args:
        text (str): The string to check.

    Returns:
        bool: True if LaTeX markdown is found, otherwise False.
    """
    if not isinstance(text, str):
        return False

    # Define regular expression patterns for LaTeX markdown
    latex_patterns = [
        r"\$\$.*?\$\$",  # Block-level math formula $$...$$
        r"\$[^$\n]+?\$",  # Inline math formula $...$
        r"\\begin\{.*?\}.*?\\end\{.*?\}",  # LaTeX environment \begin{...}...\end{...}
        r"\\[a-zA-Z]+\{.*?\}",  # LaTeX command \command{...}
        r"\\[a-zA-Z]+",  # Simple LaTeX command \command
        r"\\\[.*?\\\]",  # Display math formula \[...\]
        r"\\\(.*?\\\)",  # Inline math formula \(...\)
    ]

    # Check if any of the patterns match
    for pattern in latex_patterns:
        if re.search(pattern, text, re.DOTALL):
            return True

    return False


def clean_latex_preamble(latex_text: str) -> str:
    """
    Removes LaTeX preamble commands like document class and package imports.

    Args:
        latex_text (str): The original LaTeX text.

    Returns:
        str: The cleaned LaTeX text without preamble commands.
    """
    # Define patterns to be removed
    patterns = [
        r"\\documentclass\{[^}]+\}",  # \documentclass{...}
        r"\\usepackage\{[^}]+\}",  # \usepackage{...}
        r"\\usepackage\[[^\]]*\]\{[^}]+\}",  # \usepackage[options]{...}
        r"\\begin\{document\}",  # \begin{document}
        r"\\end\{document\}",  # \end{document}
    ]

    # Apply each pattern to clean the text
    cleaned_text = latex_text
    for pattern in patterns:
        cleaned_text = re.sub(pattern, "", cleaned_text, flags=re.IGNORECASE)

    return cleaned_text


def get_formula_in_markdown(text: str) -> str:
    """
    Formats a string containing a formula into a standard Markdown block.

    Args:
        text (str): The input string, potentially containing a formula.

    Returns:
        str: The formatted string, ready for Markdown rendering.
    """
    # Remove leading/trailing whitespace
    text = text.strip()

    # Check if it's already enclosed in $$
    if text.startswith("$$") and text.endswith("$$"):
        text_new = text[2:-2].strip()
        if "$" not in text_new:
            return f"$$\n{text_new}\n$$"
        else:
            return text

    # Handle \[...\] format, convert to $$...$$
    if text.startswith("\\[") and text.endswith("\\]"):
        inner_content = text[2:-2].strip()
        return f"$$\n{inner_content}\n$$"

    # Check if it's enclosed in \[ \]
    if len(re.findall(r".*\\\[.*\\\].*", text)) > 0:
        return text

    # Handle inline formulas ($...$)
    pattern = r"\$([^$]+)\$"
    matches = re.findall(pattern, text)
    if len(matches) > 0:
        # It's an inline formula, return it as is
        return text

    # If no LaTeX markdown syntax is present, return directly
    if not has_latex_markdown(text):
        return text

    # Handle unnecessary LaTeX formatting like \usepackage
    if "usepackage" in text:
        text = clean_latex_preamble(text)

    if text[0] == "`" and text[-1] == "`":
        text = text[1:-1]

    # Enclose the final text in a $$ block with newlines
    text = f"$$\n{text}\n$$"
    return text


def clean_text(text: str) -> str:
    """
    Cleans text by removing extra whitespace.

    Args:
        text: The original text.

    Returns:
        str: The cleaned text.
    """
    if not text:
        return ""

    # Remove leading and trailing whitespace
    text = text.strip()

    # Replace multiple consecutive whitespace characters with a single space
    if text[:2] == "`$" and text[-2:] == "$`":
        text = text[1:-1]

    return text


def layoutjson2md(cells: list, text_key: str = "text", no_page_hf: bool = False) -> str:
    """
    Converts a layout JSON format to Markdown.

    In the layout JSON, formulas are LaTeX, tables are HTML, and text is Markdown.

    Args:
        cells: A list of dictionaries, each representing a layout cell.
        text_key: The key for the text field in the cell dictionary.
        no_page_header_footer: If True, skips page headers and footers.

    Returns:
        str: The text in Markdown format.
    """
    text_items = []

    for i, cell in enumerate(cells):
        text = cell.get(text_key, "")

        if no_page_hf and cell["category"] in ["Page-header", "Page-footer"]:
            continue

        if cell["category"] == "Formula":
            text_items.append(get_formula_in_markdown(text))
        else:
            text = clean_text(text)
            text_items.append(f"{text}")

    markdown_text = "\n\n".join(text_items)
    return markdown_text
