import pytest
import anytree
import copy
from fulgurite.phylonode import PhyloNode, PhyloNodeError

NEWICK = "((((A:1.0,B:1.0):0.5,C:1.5):1.0,(D:0.5,E:0.5)):2.0,F:2.5)"
SWAPPED = "((((D:0.5,E:0.5),C:1.5):1.0,(A:1.0,B:1.0):0.5):2.0,F:2.5)"


## Made it a function instead of var so pytest picks up any problems that might
## conceivably occur, maybe this is unnecessary?
newtree = lambda: PhyloNode(label="A", children=[
    PhyloNode(label="B", children=[
        PhyloNode(label="C"),
        PhyloNode(label="D")
    ]),
    PhyloNode(label="E")
])


def test_tree_creation():
    tree = newtree()
    assert tree.is_binary


def test_attach():
    tree = newtree()
    node_f = PhyloNode(label="F")
    tree.attach(node_f)
    assert tree.is_binary


def test_detach():
    tree = newtree()
    with pytest.raises(PhyloNodeError):
        tree.detach(tree)
    for node in anytree.PreOrderIter(tree):
        if not node.is_root:
            fresh = copy.deepcopy(tree)
            fresh.detach(node)
            assert fresh.is_binary


def test_properties():
    tree = newtree()
    assert tree.is_binary
    assert not tree.is_internal
    assert all([not n.is_internal for n in tree.leaves])


def test_from_string():
    tree = PhyloNode.from_string(NEWICK)
    assert tree.is_binary


def test_eq():
    tree1 = PhyloNode.from_string(NEWICK)
    tree2 = PhyloNode.from_string(NEWICK)
    assert tree1.equal(tree2)


def test_swap():
    tree = PhyloNode.from_string(NEWICK)
    swapped = PhyloNode.from_string(SWAPPED)
    a = tree.children[0].children[0].children[0]
    b = tree.children[0].children[1]
    print("Tree:\n", anytree.RenderTree(tree))
    tree.swap(a, b)
    print("Swapped Tree:\n", anytree.RenderTree(tree))
    print("Swapped compare:\n", anytree.RenderTree(swapped))
    assert tree.equal(swapped)
