# File: test_non_adaptive.py
# File Created: Thursday, 21st May 2020 10:20:13 pm
# Author: Steven Atkinson (212726320@ge.com)

from functools import partial
import importlib
from os.path import abspath, join, dirname
import sys

src_path = abspath(join(dirname(__file__), "..", ".."))
examples_path = abspath(join(dirname(__file__), "..", "..", "examples", "adaptive"))
for p in [src_path, examples_path]:
    if p not in sys.path:
        sys.path.append(p)


def t(name):
    m = importlib.import_module(name)
    m.main(testing=True)


test_ode = partial(t, "ode")
test_e_het = partial(t, "e_het")
test_e_nonlinear = partial(t, "e_nonlinear")
