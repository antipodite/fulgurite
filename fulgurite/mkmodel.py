"""
Mk model with a single parameter, the rate parameter Q.  This
parameter is used to generate a transition probability matrix such
that any pairwise transition between states has probability Q.  For
any Q matrix, the sum of all the elements in a row should be 0 as this
is a necessary condition for a transition rate matrix.
"""
import numpy
import scipy
from anytree.iterators.levelorderiter import LevelOrderIter


def make_transition_matrix(Q, k):
    """Generate Mk equal rate transition matrix.
    Q: rate parameter. The state transition probability.
    k: number of discrete states.
    """
    d = -(k - 1) * Q
    rows = []
    for i in range(k):
        row = [Q for i in range(k)]
        row[i] = d
        rows.append(row)
    return numpy.array(rows)


def node_state_likelihoods(node, matrix):
    P = scipy.linalg.expm(node.length * matrix)
    result = []
    for state in range(len(node.likelihoods)):
        state_L = sum([x * P[i][state] for i, x in enumerate(node.likelihoods)])
        result.append(state_L)
    return result


def combine_likelihoods(node, matrix):
    if node.children:
        lls, lrs = [node_state_likelihoods(c, matrix) for c in node.children]
        return [r * l for r, l in zip(lls, lrs)]
    return node.likelihoods


def subtree_likelihood(node, Q):
    # TODO Implement reverse level order traversal properly, as it's currently
    # at least O(n²) depending on how efficient reversed() is
    matrix = make_transition_matrix(Q, len(node.likelihoods))
    for n in node.reverse_level_walk():
        n.likelihoods = combine_likelihoods(n, matrix)
    prior = 1 / len(node.likelihoods) # Uniform prior
    return sum([prior * l for l in node.likelihoods])
