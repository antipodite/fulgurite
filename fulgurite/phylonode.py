import newick
import anytree
import random


class PhyloNodeError(Exception):
    pass


class PhyloNode(anytree.NodeMixin):
    ## Walking the tree every time a node needs to be selected, etc will be inefficient,
    ## so there should be a hash table of all nodes in the tree or something
    """Represents a node in a binary-branching phylogenetic tree"""
    def __init__(self, parent=None, children=None, branch_length=None, label=None):
        self.parent = parent
        if children:
            self.children = children
        # Branch length between this node and its parent
        self.branch_length = branch_length
        self.label = label
        self.likelihoods = list()

    @classmethod
    def from_string(cls, newick_str):
        """Build a PhyloNode tree from a Newick format string"""
        fromnode = lambda n, p: PhyloNode(parent=p, branch_length=n.length, label=n.name)
        newick_tree = newick.loads(newick_str)[0]
        phylo_tree = fromnode(newick_tree, None)
        stack = [(newick_tree, phylo_tree)]
        while stack:
            newick_n, phylo_n = stack.pop()
            for c in newick_n.descendants:
                stack.append((c, fromnode(c, phylo_n)))
        return phylo_tree

    def attach(self, subtree):
        """Attach subtree to this node, making necessary changes to maintain
        binary branching structure.
        """
        if self.is_leaf:
            raise PhyloNodeError("Can't attach subtree to tip {}".format(self.label))
        self.children = [PhyloNode(children=self.children), subtree]

    def detach(self, subtree):
        """Detach a node, maintaining binary structure by collapsing remaining
        unary branch.
        """
        if subtree.is_root:
            raise PhyloNodeError("Can't detach root of tree")
        parent = subtree.parent
        subtree.parent = None
        unary = parent.children[0]
        parent.children = unary.children

    def swap(self, a, b):
        """Swap the positions of two nodes."""
        if a.is_root or b.is_root:
            raise PhyloNodeError("Can't swap position of root node")
        if a.parent == b.parent:
            raise PhyloNodeError("Attempt to swap siblings")
        # Save child order
        indices = (a.parent.children.index(a), b.parent.children.index(b))
        oldparent = b.parent
        b.parent = a.parent
        a.parent = oldparent
        # Make sure children in same order. This is much easier than trying to
        # write a recursive equality function that doesn't depend on child order
        for node, index in zip((b,a), indices):
            if node.parent.children.index(node) != index:
                node.parent.children = node.parent.children[::-1]

    def prune_and_regraft(self):
        """Generate a new tree with a randomly chosen subtree pruned and reattached
        at another randomly chosen node. How do you handle branch lengths in this
        procedure?
        """
        pass

    def equal(self, tree):
        """Return True if this tree and other tree have same nodes and topology.
        """
        stack = [(self, tree)] # Need to control traversal myself here as order matters
        while stack:
            this, that = stack.pop()
            if not this == that or not this.children == that.children:
                return False
            stack.extend([(ca, cb) for ca, cb in zip(this.children, that.children)])
        return True

    @property
    def is_binary(self):
        """Return True if this is a proper binary tree"""
        return all([len(n.children) == 2 for n in anytree.PreOrderIter(self) if not n.is_leaf])

    @property
    def is_internal(self):
        """Return True is this node is neither the root nor a tip"""
        return not self.is_root and not self.is_leaf

    def __eq__(self, node):
        return all([self.label == node.label, self.branch_length == node.branch_length])

    def __repr__(self):
        return "PhyloNode({}, {})".format(self.label, self.branch_length)

