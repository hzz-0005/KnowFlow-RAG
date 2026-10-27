"""Literal identity clues only; never a world-knowledge alias resolver."""

import re


def literal_name_in(name, text):
    return bool(re.search(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])", text, re.I))
