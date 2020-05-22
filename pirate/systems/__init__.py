# File: __init__.py
# File Created: Wednesday, 2nd January 2019 2:34:44 pm
# Author: Steven Atkinson (212726320@ge.com)

"""
The 'systems' contains class definitions for the various physical systems that
we will apply our methods to.

Our main way of interacting with a system is by querying it to produce data.
For now, we assume that these are supervised data (i.e. input-output pairs).
We can specify certain inputs explicitly ("Please drive at speed X"), while 
others are inherently random and cannot be controlled ("What is the weather 
today?")

Given all inputs, the system responds in some way ("the car comes to a stop").  
We interact with this response through a set of observables (e.g. measuring the
distance traveled by the car before it stops.)

Systems may be either "real" physical/laboratory systems or computer
simulations.  In the former case, .sample() might be implemented by reading a 
stored data file...
"""

from . import base
from . import elliptic
from . import ode
