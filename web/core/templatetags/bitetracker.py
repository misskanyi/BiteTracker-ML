from django import template

register = template.Library()


@register.filter
def get_item(mapping, key):
    """Dictionary lookup for keys with spaces (e.g. metrics CSV column names)."""
    return mapping.get(key) if hasattr(mapping, "get") else None


@register.filter
def humanize_key(key):
    """'expected_rainfall_mm' → 'Expected rainfall mm'."""
    text = str(key).replace("_", " ").strip()
    return text[:1].upper() + text[1:]
