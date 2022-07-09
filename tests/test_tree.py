import pytest
import anytree
import copy
import logging
import numpy

from fulgurite.tree import PhyloNode, PhyloNodeError

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
    assert tree.is_binary and all([n.state != None for n in tree.leaves])


def test_attach():
    tree = PhyloNode.from_string(NEWICK)
    node_g = PhyloNode(label="G")
    tree.attach(tree, node_g)
    assert tree.is_binary and all([n.label != None for n in tree.leaves])


def test_detach():
    tree = PhyloNode.from_string(NEWICK)
    for node in anytree.PreOrderIter(tree):
        if not node.is_root:
            fresh = copy.deepcopy(tree)
            detached = fresh.detach(node)
            assert detached.is_binary and all([n.label != None for n in detached.leaves])


def test_from_string():
    tree = PhyloNode.from_string(NEWICK)
    assert tree.is_binary


def test_eq():
    tree1 = PhyloNode.from_string(NEWICK)
    tree2 = PhyloNode.from_string(NEWICK)
    assert tree1.equal(tree2) and tree1 == tree2


def test_prune_regraft():
    tree = PhyloNode.from_string(NEWICK)
    LOGGER.debug("Tree:\n" + str(anytree.RenderTree(tree)))
    tree.prune_and_regraft()
    LOGGER.debug("Regrafted:\n" + str(anytree.RenderTree(tree)))


def test_swap():
    tree = PhyloNode.from_string(NEWICK)
    swapped = PhyloNode.from_string(SWAPPED)
    a = tree.children[0].children[0].children[0] # 
    b = tree.children[0].children[1]
    LOGGER.debug("Tree:\n" + str(anytree.RenderTree(tree)))
    tree.swap(a, b)
    LOGGER.debug("Swapped Tree:\n" + str(anytree.RenderTree(tree)))
    LOGGER.debug("Swapped compare:\n" + str(anytree.RenderTree(swapped)))
    assert tree.equal(swapped)


