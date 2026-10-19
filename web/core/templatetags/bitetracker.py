import math
from decimal import Decimal, InvalidOperation

from django import template
from django.contrib.humanize.templatetags.humanize import intcomma

register = template.Library()

# Placeholders the Google Sheet loader writes for fields the source does not provide.
_MISSING_TEXT = {"", "not provided", "nan", "none"}


@register.filter
def get_item(mapping, key):
    """Dictionary lookup for keys with spaces (e.g. metrics CSV column names)."""
    return mapping.get(key) if hasattr(mapping, "get") else None


@register.filter
def humanize_key(key):
    """'expected_rainfall_mm' → 'Expected rainfall mm'."""
    text = str(key).replace("_", " ").strip()
    return text[:1].upper() + text[1:]


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


@register.filter
def is_fractional(value):
    """True for a meal count that is not a whole number (preprocessing capped it at an outlier fence)."""
    number = _number(value)
    return number is not None and not number.is_integer()


@register.filter
def meals(value):
    """A meal count shown as recorded: whole numbers with separators, fractional values to at most
    three decimals (IQR fences land on multiples of 1/8 in the processed data), never truncated."""
    number = _number(value)
    if number is None:
        return "—"
    if number.is_integer():
        return intcomma(int(number))
    return intcomma(f"{number:.3f}".rstrip("0"))


@register.filter
def location(record):
    """'Sub-county, County' from a school record, skipping placeholder values; '' when nothing is known."""
    parts = []
    for key in ("sub_county", "county"):
        value = record.get(key) if hasattr(record, "get") else None
        if value is not None and str(value).strip().lower() not in _MISSING_TEXT:
            parts.append(str(value).strip())
    return ", ".join(parts)


@register.filter
def kes(value):
    """'KES 27,000' — money with separators; cents only when the amount is not whole."""
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return "—"
    if not amount.is_finite():
        return "—"
    text = f"{amount:,.0f}" if amount == amount.to_integral_value() else f"{amount:,.2f}"
    return f"KES {text}"
