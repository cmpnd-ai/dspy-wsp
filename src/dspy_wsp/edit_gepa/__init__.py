"""GEPA population search over `edits` programs: a writer LM edits a small network's weights as text, no gradients.

    from dspy_wsp.edit_gepa import EditGEPA, language

See examples/seaquest_gepa/README.md for a recipe.
"""

from . import language
from .search import PROTOCOL, EditGEPA, EditGEPAResult, program_id

__all__ = ["EditGEPA", "EditGEPAResult", "PROTOCOL", "language", "program_id"]
