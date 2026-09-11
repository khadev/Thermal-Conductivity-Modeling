"""Utility helpers."""


def format_number(value, decimals=4):
    """Format a number for display."""
    if value is None or isinstance(value, str):
        return str(value)
    try:
        return f"{float(value):.{decimals}f}"
    except (ValueError, TypeError):
        return str(value)


def status_icon(r2):
    """Return a status indicator based on R2."""
    if r2 >= 0.99:
        return "Excellent"
    elif r2 >= 0.95:
        return "Good"
    elif r2 >= 0.80:
        return "Fair"
    else:
        return "Poor"
