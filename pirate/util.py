# File: util.py
# File Created: Wednesday, 13th March 2019 3:02:18 pm
# Author: Steven Atkinson (212726320@ge.com)


import types
import copy

import torch
import numpy as np
import scipy.stats
import sympy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def jitter_op(
    f: types.FunctionType,
    matrix: np.ndarray,
    max_tries: int = 10,
    initial_jitter: float = 1.0e-16,
    verbose: bool = True,
):
    """
    Linear algebra operation that might not succeed without a little jitter.
    """
    matrix_copy = None
    for i in range(max_tries):
        try:
            return f(matrix) if matrix_copy is None else f(matrix_copy)
        except:
            if verbose:
                print(
                    "Failed to do op {} (Try {} / {})".format(
                        f.__name__, i + 1, max_tries
                    )
                )
            diag_mean = np.diag(matrix).mean()
            jitter = diag_mean * 10.0 ** (i + 1)
            matrix_copy = matrix + jitter * np.eye(matrix.shape)

    # Exceeded max tries: throw an error
    raise RuntimeError("Failed to do op {}".format(f.__name__))


def kl_divergence_from_samples(
    log_prob_p: types.FunctionType, log_prob_q: types.FunctionType, x_from_p: np.ndarray
) -> float:
    """
    Estimate the KL divergence KL(p||q) using samples that are assumed to be 
    drawm from p(x)

    x's live in d dimensions.

    KL(p||q) = -int p(x) log(q(x)/p(x)) dx
    
    If N samples of x are drawn from p(x), then this is estimated as
    
    KL ~ -1/N sum_{i=1}^N (logq(x_i) - logp(x_i))

    :param log_prob_p: Function that takes as input x_from_p and outputs an 
        array of length N where each entry is the log probability of the 
        corresponding row of x_from_p.
    :param log_prob_q: Function that does the same for log q(x)
    :param x_from_p: an array (shape=Nxd) of samples assumed to be drawn from 
        p(x)
    """
    return -np.mean(log_prob_q(x_from_p) - log_prob_p(x_from_p))


def rms(x: np.ndarray) -> float:
    return np.sqrt(np.mean(x ** 2))


def rmse(x: np.ndarray, y: np.ndarray) -> float:
    return rms(x - y)


def _convert_inverse_prim(prim, args):
    """
    Convert inverse prims according to:
    [Dd]iv(a,b) -> Mul[a, 1/b]
    [Ss]ub(a,b) -> Add[a, -b]
    We achieve this by overwriting the corresponding format method of the sub and div prim.
    """
    prim = copy.copy(prim)
    # prim.name = re.sub(r'([A-Z])', lambda pat: pat.group(1).lower(), prim.name)    # lower all capital letters

    converter = {
        "sub": lambda *args_: "Add({}, Mul(-1,{}))".format(*args_),
        "protectedDiv": lambda *args_: "Mul({}, Pow({}, -1))".format(*args_),
        "mul": lambda *args_: "Mul({},{})".format(*args_),
        "add": lambda *args_: "Add({},{})".format(*args_),
        "neg": lambda *args_: "Mul(-1,{})".format(*args_),
        "div": lambda *args_: "Mul({}, Pow({}, -1))".format(*args_),
    }

    prim_formatter = converter.get(prim.name, prim.format)

    return prim_formatter(*args)


def _stringify_for_sympy(expression: types.FunctionType):
    """Return the expression in a human readable string.
    """
    string = ""
    stack = []
    for node in expression:
        stack.append((node, []))
        while len(stack[-1][1]) == stack[-1][0].arity:
            prim, args = stack.pop()
            string = _convert_inverse_prim(prim, args)
            if len(stack) == 0:
                break  # If stack is empty, all nodes should have been seen
            stack[-1][1].append(string)
    return string


def symbolic_print(expression: types.FunctionType, save_as: str = None):
    """
    simplify and export a png and latex file for the algebraic expressions
    """
    if isinstance(expression, str):
        symexpr = sympy.simplify(expression)
    else:
        stringify_symexpr = _stringify_for_sympy(expression)
        symexpr = sympy.simplify(stringify_symexpr)

    if save_as is not None:
        sympy.init_printing()
        print(symexpr)
        sympy.preview(symexpr, viewer="file", outputTexFile=save_as + ".tex")
        sympy.preview(symexpr, viewer="file", filename=save_as + ".png")
        img = mpimg.imread(save_as + ".png")
        plt.figure()
        plt.axis("off")
        plt.imshow(img)
    return symexpr


def negative_log_likelihood(r: np.ndarray) -> float:
    """
    Compute the negative log likelihood of a residual r under a Gaussian 
    likelihood whose variance is the MLE (i.e. the RMS of the residual)
    """

    return -scipy.stats.norm(scale=rms(r)).logpdf(r).sum()


def lchop(s, b):
    """
    Remove the beginning string "b" from the string "s"
    """
    return s[len(b) :] if s.startswith(b) else s
