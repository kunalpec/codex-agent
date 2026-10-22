"""Static activity labels for the terminal UI."""

from rich.text import Text


class ActivityStatus(Text):
    """Display the current activity without a spinner or cycling animation."""

    def __init__(self, label: str = "Thinking") -> None:
        super().__init__(label, style="bold yellow")
