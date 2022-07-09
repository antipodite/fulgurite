import pytest
import logging
from fulgurite.tree import PhyloNode

LOGGER = logging.getLogger(__name__)
# Tree from Harmon (2022), 8.7 Appendix - Felsenstein's pruning algorithm
NEWICK = "((((A:1.0,B:1.0):0.5,C:1.5):1.0,(D:0.5,E:0.5)):2.0,F:2.5)"

def test_likelihood():
    tree = PhyloNode.from_string(NEWICK)
    assert tree.get_likelihood == 0.00150
