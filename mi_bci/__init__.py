"""
Public package interface.

This keeps the main import surface small:
    from mi_bci import FBCSPLDAModel, load_mat_data
"""

from .model import FBCSPLDAModel
from .data import load_mat_data
