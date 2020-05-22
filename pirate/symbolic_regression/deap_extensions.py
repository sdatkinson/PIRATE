# File: deap_extensions.py
# File Created: Friday, 15th November 2019 4:19:53 pm
# Author: Steven Atkinson (212726320@ge.com)

"""
Extensions to deap's primitive tree class for symbolic regression on 
functions.
"""

from operator import attrgetter
import sys

import deap.gp
from deap.tools.selection import selRandom
import numpy as np
import torch
from torch.distributions import Normal

from ..function import Function, Constant


class PrimitiveSet(deap.gp.PrimitiveSet):
    """
    Extension to PrimitiveSet for parameters
    """

    def add_parameter(self, name):
        terminal = Constant
        ret_type = deap.gp.__type__

        symbolic = False
        if name is None and callable(terminal):
            name = terminal.__name__

        assert name not in self.context, (
            "Terminals are required to have a unique name. "
            "Consider using the argument 'name' to rename your "
            "second %s terminal." % (name,)
        )

        if name is not None:
            self.context[name] = terminal
            terminal = name
            symbolic = True
        elif terminal in (True, False):
            # To support True and False terminals with Python 2.
            self.context[str(terminal)] = terminal

        prim = Parameter(terminal, symbolic, ret_type)
        self._add(prim)
        self.terms_count += 1


class PrimitiveTree(deap.gp.PrimitiveTree):
    """
    Extends deap's PrimitiveTree class mainly with functionality centered around
    having parameters as terminals:
    
    1) By including a likelihood_std attribute.
    This is intended to correspond to the standard deviation (scale parameter) 
    of a univariate Gaussian likelihood.
    2) By allowing for primitives to be `Parameter` instances, and including 
    other helpful functionality (e.g. .has_parameters, .num_parameters)
    3) By extending `.stringify()` such that if Parameter instances are in the
    graph, we can constrol whether they appear as called functions (so that the 
    string can be `eval`'d as a callable with arguments that can be filled in 
    with parameter values) or not (which is easier to read).
    4) An attribute for holding the results of calibration.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Observation (residual) likelihood standard deviation
        self.likelihood_std = 1.0
        # type calibration.CalibrationResults
        # TODO fix circular import so that we can use type hint here.
        self.calibration_results = None

    def __str__(self):
        return self.stringify()

    @property
    def has_parameters(self) -> bool:
        """
        Returns whether there are any parameter leaves in the graph
        """

        return self.num_parameters > 0

    @property
    def num_parameters(self) -> int:
        """
        How many parameters are in this graph
        """

        return sum([int(isinstance(elem, Parameter)) for elem in self])

    def stringify(self, with_param_args: bool = False) -> str:
        """
        Convert the graph into a string expression that can be compiled using 
        eval() and a primitive set as context

        Keep in mind that the (tree) graph is stored as a depth-first list.

        :param with_args: If true, include call signatures for parameters so
        that you can fill them in as kwargs on the call signature for the 
        compiled function
        """

        return self._stringify(0, 0, with_param_args)[0]

    def _stringify(self, i_node, i_param, with_param_args):
        """
        Keeps track of indices so we can close argument lists and assign unique 
        args to parameters.

        :param i_node: Index of the node to consider now.
        :param i_param: How many parameters have been encountered so far (used 
        to assign unique args to input to each if with_param_args==True)
        :return: (str, int, int) 
            * assembled string
            * current list index
            * current # params seen
        """

        s = ""
        # Loop through nodes, which may be either operations (non-leaves) or
        # arguments (leaves).
        node = self[i_node]
        i_node += 1
        # Kinda hacky; see Teminal.format()
        s += node.value if hasattr(node, "value") else node.name
        if node.arity > 0:
            # Assemble operator args
            s += "("
            for j in range(node.arity):
                sj, i_node, i_param = self._stringify(i_node, i_param, with_param_args)
                s += sj
                if j < node.arity - 1:
                    s += ","
            s += ")"
        elif isinstance(node, Parameter):
            if with_param_args:
                s += "(theta_%i)" % i_param
            i_param += 1
        # else: some other terminal that we don't need to fuss about; name is all.

        return s, i_node, i_param


class Parameter(deap.gp.Terminal):
    """
    Special type of terminal for housing a pirate.function.Constant meant to be 
    given a value in the compiled graph's call signature
    """

    pass


def _compile_with_parameters(
    expr: PrimitiveTree, pset: deap.gp.PrimitiveSet
) -> Function:
    """
    Extension of `deap.gp.compile()` for expressions with parameters.

    If the expression has parameter leaves, then 

    :param expr: Expression to compile. It can either be a PrimitiveTree,
                 a string of Python code or any object that when
                 converted into string produced a valid Python code
                 expression.
    :param pset: Primitive set against which the expression is compile.
    :returns: a function if the primitive set has 1 or more arguments,
              or return the results produced by evaluating the tree.
    """

    # e.g. functions for operator
    assert len(pset.arguments) > 0, "Assume we've got arguments"
    code = expr.stringify(with_param_args=True)
    if len(pset.arguments) > 0:
        # This section is a stripped version of the lambdify
        # function of SymPy 0.6.6.
        args = ",".join(arg for arg in pset.arguments)
        # Also include parameter values!
        for i in range(expr.num_parameters):
            args += ", theta_%i=None" % i
        code = "lambda {args}: {code}".format(args=args, code=code)
    try:
        return eval(code, pset.context, {})
    except MemoryError:
        _, _, traceback = sys.exc_info()
        raise MemoryError(
            "DEAP : Error in tree evaluation :"
            " Python cannot evaluate a tree higher than 90. "
            "To avoid this problem, you should use bloat control on your "
            "operators. See the DEAP documentation for more information. "
            "DEAP will now abort."
        ).with_traceback(traceback)


def compile(expr: PrimitiveTree, pset: deap.gp.PrimitiveSet) -> Function:
    """
    Extension of deap's compile functionality so that we can incorporate parameters
    that can be initialized as the function is called.

    The call signature of the compiled function shall be:
    f(ps1, ..., psN, )
    """

    if not expr.has_parameters:
        # No parameters: make a wrapper to shed **kwargs
        _func = deap.gp.compile(expr, pset)

        def func(*args, **kwargs):
            return _func(*args)

        return func
    else:
        return _compile_with_parameters(expr, pset)
