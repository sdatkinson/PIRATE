# File: experiment.py
# File Created: Friday, 19th July 2019 4:32:17 pm
# Author: Steven Atkinson (212726320@ge.com)

"""
Classes for handling a complete experiment (left hand side + inhomogeneous term)
"""

from typing import Dict

import pandas as pd

from ..function import Function


class Experiment(object):
    """
    Class meant to handle all dataset & function objects associated with a 
    single "experiment" (aka particular solution of a differential equation),
    including terms to be manipulated on the left hand side, an (optional) 
    inhomogeneous term for the right hand side, etc. and the input data 
    associated with the experiment.

    We recognize that the Functions may have been trained on data at different 
    locations, but if we want to, say, evaluate a residual, we need to decide on
    common points to evaluate all functions at.  Provide these as `data`.
    """

    def __init__(
        self,
        left_hand_side: Dict[str, Function],
        data: pd.DataFrame,
        inhomogeneous: Function = None,
    ):
        self.left_hand_side = left_hand_side
        self.data = data
        self.inhomogeneous = inhomogeneous if inhomogeneous is not None else None
