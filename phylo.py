import collections
import random
import math

import newick
import numpy
import scipy


# Tree from Harmon (2022), 8.7 Appendix - Felsenstein's pruning algorithm
TEST_TREE = newick.loads("((((A:1.0,B:1.0):0.5,C:1.5):1.0,(D:0.5,E:0.5)):2.0,F:2.5)")[0]
TEST_TIPS = {"A": 0, "B": 1, "C": 0, "D": 2, "E": 2, "F": 1}

# The Mk model, which assumes that all state transitions are equally probable
TEST_Q = numpy.array([
    [-2, 1, 1],
    [1, -2, 1],
    [1, 1, -2]
])


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


def unroot(tree):
    """Convert a rooted tree to an unrooted tree"""
    pass


def nearest_neighbour_interchange(tree):
    """The simplest algorithm for transforming a tree topology (Felsenstein 2004).
    An interior branch of a subtree and the two branches connected to that branch at
    each end are erased, so we're left with 4 subtrees. 4 subtrees can be joined into
    a tree in 3 possible ways, one of which is the original subtree, so there are two
    possibilities. 2(n - 3) 
    """
    # 1. Pick an internal node at random
    node = random.choice([n for n in tree.walk() if n.descendants and n.ancestor])
    # 2. It's got 4 connections: its ancestor, sibling, left child, right child
    

## Where next?
# The likelihood function is different for each model of evolution. I reckon stick with
# simple Mk and try doing metropolis hastings with the parameters being the tree topology
# and the branch lengths, so look into the algorithms which manipulate the tree to suggest
# new ones
# Also thinking about picking up Julia, so could rewrite once I get a bit more written

def test():
    tl = tree_likelihood(TEST_Q, TEST_TREE, TEST_TIPS)
    print(tl)

