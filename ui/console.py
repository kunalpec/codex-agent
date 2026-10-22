from rich.console import Console

from ui.theme import THEME


def create_console() -> Console:
    return Console(theme=THEME, highlight=False, soft_wrap=True)
