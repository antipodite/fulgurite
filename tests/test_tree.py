import pytest
import anytree
import copy
import logging

from fulgurite.phylonode import PhyloNode, PhyloNodeError

LOGGER = logging.getLogger(__name__)
NEWICK = "((((A:1.0,B:1.0):0.5,C:1.5):1.0,(D:0.5,E:0.5)):2.0,F:2.5)"
SWAPPED = "((((D:0.5,E:0.5),C:1.5):1.0,(A:1.0,B:1.0):0.5):2.0,F:2.5)"


def test_tree_creation():
    tree = PhyloNode.from_string(NEWICK)
    assert tree.is_binary


def test_attach():
    tree = PhyloNode.from_string(NEWICK)
    node_f = PhyloNode(label="F")
    tree.attach(tree, node_f)
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
    print(a, b)
    print("Tree:\n", anytree.RenderTree(tree))
    tree.swap(a, b)
    print("Swapped Tree:\n", anytree.RenderTree(tree))
    print("Swapped compare:\n", anytree.RenderTree(swapped))
    assert tree.equal(swapped)

