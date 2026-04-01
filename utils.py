def read_prompt(path: str, **kwargs) -> str:
    """
    Reads a prompt template from a text file and formats it with the given keyword arguments.

    Args:
        path (str): Path to the prompt .txt file.
        **kwargs: Key-value pairs to fill placeholders in the template.

    Returns:
        str: Formatted prompt string.
    """
    with open(path, "r", encoding="utf-8") as f:
        template = f.read()
    return template.format(**kwargs)
