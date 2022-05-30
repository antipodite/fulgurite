from numpy import array, ndarray
from scipy.linalg.expm import expm


TEST_TREE = []


def trait_state_probs(t, Q):
    """Calculate probability distribution of trait states at time t from rate matrix Q."""
    return expm(Q * t)


def branch_likelihood(state, tip_state_probs, branch_len, Q):
    """Calculate state likelihood for a single branch from some node"""
    P = trait_state_probs(branch_len, Q)
    return sum([x * P[i][state] for i, x in enumerate(tip_state_probs)])


def node_likelihood(Q, left_len, right_len, left_state_p, right_state_p):
    """Calculate the likelihood for each possible state based on state likelihoods at
    left and right branches."""
    node_state_likelihoods = []
    for i in range(len(left_state_p)):
        LL = branch_likelihood(i, left_state_p, branch_len, Q)
        LR = branch_likelihood(i, right_state_p, branch_len, Q)
        node_state_likelihoods.append(LL * LR)
    return node_state_likelihoods


def tree_likelihood(Q, tree):
    """Felsenstein's (1973) pruning algorithm. Calculate the root likelihood of a tree"""
    # BFS, if node has likelihoods present do nothing, if not, do node likelihood on
    # left and right children.
    pass
