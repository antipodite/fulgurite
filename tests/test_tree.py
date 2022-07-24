import anytree
import logging
import random
import pathlib
import csv

from distutils import dir_util
from fulgurite.tree import PhyloNode, PhyloTree
from fulgurite.mcmc import sample


LOGGER = logging.getLogger(__name__)
DATADIR = pathlib.Path(__file__).parent / "data"

# Tree from Harmon (2022), 8.7 Appendix - Felsenstein's pruning algorithm
NEWICK = "((((A:1.0,B:1.0):0.5,C:1.5):1.0,(D:0.5,E:0.5)):2.0,F:2.5)"
SWAPPED = "((((D:0.5,E:0.5),C:1.5):1.0,(A:1.0,B:1.0):0.5):2.0,F:2.5)"

# Tip states for testing likelihood function
TEST_TIPS = {"A": 0, "B": 1, "C": 0, "D": 2, "E": 2, "F": 1}


def load_squamate_data():
    """Load the squamate tree from Harmon (2022).
    Attach tip states for the "has limbs" character, this needs to be
    calculated from the forelimb and hindlimb lengths.
    """
    with open(DATADIR / "brandley_table.csv") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader]
    # Extract the "has limbs" character from the data. Need to count species
    # with "FLL" and "HLL" = 0  as limbless, others as limbed
    states = {}
    for row in rows:
        name = row["Species"].replace(" ", "_")
        is_limbless = int(row["HLL"] == row["FLL"] == "0")
        states[name] = is_limbless
    # Create the squamate tree with the generated tip states for the "has
    # limbs" character
    with open(DATADIR / "squamate.phy") as f:
        tree = PhyloTree.from_string(f.read(), states=states)
    return tree


def test_squamate_tree():
    """Confirm squamate data for "is limbless" character loaded correctly"""
    tree = load_squamate_data()
    LOGGER.debug(str(tree))
    assert len([x for x in tree.leaves if x.state == 1]) == 51
    assert tree.root.is_binary
    
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


def test_mcmc():
    tree = PhyloTree.from_string(NEWICK, TEST_TIPS)
    
