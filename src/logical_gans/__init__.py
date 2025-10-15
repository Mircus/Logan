"""
logical_gans: logical constraints + EF games utilities.
"""
from .logic.ef_games import EFGameSimulator, ApproximateEFDistance  # re-export
from .logic.mso import MSOPropertyLibrary  # convenience accessor
__all__ = ["EFGameSimulator", "ApproximateEFDistance", "MSOPropertyLibrary"]
