import anytree
import logging
import numpy
import random

from fulgurite.tree import PhyloNode, PhyloNodeError, PhyloTree

LOGGER = logging.getLogger(__name__)
# Tree from Harmon (2022), 8.7 Appendix - Felsenstein's pruning algorithm
NEWICK = "((((A:1.0,B:1.0):0.5,C:1.5):1.0,(D:0.5,E:0.5)):2.0,F:2.5)"
SWAPPED = "((((D:0.5,E:0.5),C:1.5):1.0,(A:1.0,B:1.0):0.5):2.0,F:2.5)"

# The Mk model, which assumes that all state transitions are equally probable. This
# rate matrix is for a character with 3 possible states
TEST_Q = numpy.array([
    [-2, 1, 1],
    [1, -2, 1],
    [1, 1, -2]
])
# Tip states for testing likelihood function
TEST_TIPS = {"A": 0, "B": 1, "C": 0, "D": 2, "E": 2, "F": 1}


def test_tree_creation():
    tree = PhyloNode.from_string(NEWICK, states=TEST_TIPS)
    LOGGER.debug("Tree with tips:\n" + str(anytree.RenderTree(tree)))
    assert tree.is_binary
    assert all([n.state != None and n.likelihoods for n in tree.leaves])


def test_attach():
    tree = PhyloNode.from_string(NEWICK)
    node_g = PhyloNode(label="G")
    tree.attach(node_g)
    assert tree.is_binary and all([n.label != None for n in tree.leaves])


def test_detach():
    tree = PhyloNode.from_string(NEWICK)
    node = random.choice([n for n in anytree.PreOrderIter(tree) if not n.is_root])
    node.detach()
    assert tree.is_binary
    tree = PhyloNode.from_string(NEWICK)
    for node in anytree.PreOrderIter(tree):
        if not node.is_root:
            node.detach()
            assert tree.is_binary


def test_from_string():
    tree = PhyloNode.from_string(NEWICK)
    assert tree.is_binary


def test_eq():
    tree1 = PhyloNode.from_string(NEWICK)
    tree2 = PhyloNode.from_string(NEWICK)
    assert tree1.equal(tree2) and tree1 == tree2


def test_likelihood():
    tree = PhyloNode.from_string(NEWICK, TEST_TIPS)
    L = tree.get_likelihood(1)
    assert round(L, 4) == 0.0015
    # So can have a look at the tree with computed likelihoods
    LOGGER.debug(anytree.RenderTree(tree))


def test_phylotree_creation():
    """Test wrapper class"""
    tree = PhyloTree.from_string(NEWICK, TEST_TIPS)
    assert tree.root == PhyloNode.from_string(NEWICK)
    LOGGER.debug(str(tree))


