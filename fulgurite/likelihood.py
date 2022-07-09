import collections
import random
import math
import copy

import newick
import numpy
import scipy
import anytree


def trait_state_probs(t, Q):
    """Calculate probability distribution of trait states at time t from rate matrix Q."""
    return scipy.linalg.expm(Q * t)


def branch_likelihood(state, tip_state_probs, branch_len, Q):
    """Calculate state likelihood for a single branch from some node"""
    P = trait_state_probs(branch_len, Q)
    return sum([x * P[i][state] for i, x in enumerate(tip_state_probs)])


def node_likelihood(Q, left_len, right_len, left_state_p, right_state_p):
    """Calculate the likelihood for each possible state based on state likelihoods at
    left and right branches."""
    node_state_likelihoods = []
    for i in range(len(left_state_p)):
        LL = branch_likelihood(i, left_state_p, left_len, Q)
        LR = branch_likelihood(i, right_state_p, right_len, Q)
        node_state_likelihoods.append(LL * LR)
    return node_state_likelihoods


def tree_likelihood(Q, tree, tip_states):
    """Felsenstein's (1973) pruning algorithm. Calculate the root likelihood of a tree
    Q: state transition rate matrix
    tree: newick.Node instance containing root of tree
    tip_states: dict of {leaf_name : state_index}
    Assumes the tree is binary and in the Newick format where leaves are named
    and branch lengths are given.
    """
    # Attach tip likelihoods to leaf nodes. This is easy, the likelihood for state X at
    # some tip is 1 if the tip has that state, otherwise it's 0.
    tips = tree.get_leaves()
    n_states = max(tip_states.values()) + 1
    likelihoods = {} # For storing computed likelihoods at each node. Keyed by id(node)
    for label, state_index in tip_states.items():
        tip_likelihoods = [1 if i == state_index else 0 for i in range(n_states)]
        tip = tree.get_node(label)
        likelihoods[id(tip)] = tip_likelihoods

    # Traverse tree bottom up in breadth-first order, computing likelihood at each node
    queue = collections.deque(tips)
    while queue:
        node = queue.popleft()
        if node.ancestor and node.ancestor not in queue:
            queue.append(node.ancestor)
        if node.descendants:
            left, right = node.descendants
            if id(left) in likelihoods and id(right) in likelihoods:
                likelihoods[id(node)] = node_likelihood(
                    Q, left.length, right.length, likelihoods[id(left)], likelihoods[id(right)]
                )
    # Now calculate the likelihood across the whole tree. For the moment I will assume
    # the prior probability of each state is uniform.
    prior = 1 / n_states
    return sum([prior * l for l in likelihoods[id(tree)]])


def test():
    tl = tree_likelihood(TEST_Q, TEST_TREE, TEST_TIPS)
    print(tl)
    print(regraft(TEST_TREE))
