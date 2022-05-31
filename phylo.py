from collections import deque

import newick
from numpy import array, ndarray
from scipy.linalg import expm


# Tree from Harmon (2022), 8.7 Appendix - Felsenstein's pruning algorithm
TEST_TREE = newick.loads("((((A:1.0,B:1.0):0.5,C:1.5):1.0,(D:0.5,E:0.5)):2.0,F:2.5)")[0]
TEST_Q = array([
    [-2, 1, 1],
    [1, -2, 1],
    [1, 1, -2]
])


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
    # Attach tip likelihoods to leaf nodes
    tips = tree.get_leaves()
    n_states = max(tip_states.values())
    for label, state_index in tip_states.items():
        likelihoods = [1 if i == state_index else 0 for i in range(n_states+1)]
        tree.get_node(label).likelihoods = likelihoods

    # Traverse tree bottom up in breadth-first order, computing likelihood at each node
    queue = deque(tips)
    while queue:
        print(queue)
        node = queue.popleft()
        if node.ancestor and node.ancestor not in queue:
            queue.append(node.ancestor)
        if node.descendants:
            left, right = node.descendants
            if hasattr(left, "likelihoods") and hasattr(right, "likelihoods"): # Fix
                node.likelihoods = node_likelihood(
                    Q, left.length, right.length, left.likelihoods, right.likelihoods
                )
    return tree.likelihoods
            

def tree_likelihood_dict(Q, tree, tip_states):
    """Felsenstein's (1973) pruning algorithm. Calculate the root likelihood of a tree
    Q: state transition rate matrix
    tree: newick.Node instance containing root of tree
    tip_states: dict of {leaf_name : state_index}
    Assumes the tree is binary and in the Newick format where leaves are named
    and branch lengths are given.
    """
    # Attach tip likelihoods to leaf nodes
    tips = tree.get_leaves()
    n_states = max(tip_states.values())
    likelihoods = {} # For storing computed likelihoods at each node
    for label, state_index in tip_states.items():
        tip_likelihoods = [1 if i == state_index else 0 for i in range(n_states+1)]
        tip = tree.get_node(label)
        likelihoods[id(tip)] = tip_likelihoods

    # Traverse tree bottom up in breadth-first order, computing likelihood at each node
    queue = deque(tips)
    while queue:
        print(queue)
        node = queue.popleft()
        if node.ancestor and node.ancestor not in queue:
            queue.append(node.ancestor)
        if node.descendants:
            left, right = node.descendants
            node_id, left_id, right_id = (id(node), id(left), id(right))
            if left_id in likelihoods and right_id in likelihoods:
                likelihoods[node_id] = node_likelihood(
                    Q, left.length, right.length, likelihoods[left_id], likelihoods[right_id]
                )
    return likelihoods[id(tree)]

