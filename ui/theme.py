from rich.theme import Theme

ACCENT = "cyan"
SUCCESS = "green"
WARNING = "yellow"
ERROR = "red"
MUTED = "grey50"
PROMPT = "bold bright_white"
CODE = "monokai"

ICONS = {"ok": "✓", "run": "▶", "error": "✗", "bullet": "•"}
ASCII_ICONS = {"ok": "*", "run": ">", "error": "x", "bullet": "+"}
LABEL_AGENT = "Agent"

THEME = Theme({
    "accent": ACCENT,
    "success": SUCCESS,
    "warning": WARNING,
    "error": ERROR,
    "muted": MUTED,
    "prompt": PROMPT,
})


def icon(console, name: str) -> str:
    glyph = ICONS[name]
    try:
        glyph.encode(console.encoding or "utf-8")
        return glyph
    except (UnicodeEncodeError, LookupError):
        return ASCII_ICONS[name]
