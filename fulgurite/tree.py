import newick
import anytree
import random


class PhyloNodeError(Exception):
    pass


class PhyloNode(anytree.NodeMixin):
    ## Walking the tree every time a node needs to be selected, etc will be inefficient,
    ## so there should be a hash table of all nodes in the tree or something
    """Represents a node in a binary-branching phylogenetic tree"""
    def __init__(self, parent=None, children=None, length=None, label=None, state=None):
        self.parent = parent
        if children:
            self.children = children
        # Branch length between this node and its parent
        self.length = length
        self.label = label
        self.state = state
        self.likelihoods = list()


    @classmethod
    def from_string(cls, newick_str, states=None):
        """Build a PhyloNode tree from a Newick format string"""
        fromnode = lambda n, p: PhyloNode(parent=p, length=n.length, label=n.name)
        newick_tree = newick.loads(newick_str)[0]
        phylo_tree = fromnode(newick_tree, None)
        stack = [(newick_tree, phylo_tree)]
        while stack:
            newick_n, phylo_n = stack.pop()
            for c in newick_n.descendants:
                stack.append( (c, fromnode(c, phylo_n)) )
        # Attach states
        if states:
            for node in anytree.PreOrderIter(phylo_tree):
                if node.label in states:
                    node.state = states[node.label]
        return phylo_tree


    def attach(self, a, b):
        """Attach subtree b to node a, making necessary changes to maintain
        binary branching structure.
        """
        # if self.is_leaf:
        #     raise PhyloNodeError("Can't attach subtree to tip {}".format(self.label))
        a.children = [PhyloNode(children=a.children), b]
        return self


    def detach(self, subtree):
        """Detach a node, maintaining binary structure by collapsing remaining
        unary branch.
        """
        if subtree.is_root:
            raise PhyloNodeError("Can't detach root of tree")
        sibling = subtree.siblings[0]
        parent = subtree.parent
        grandparent = parent.parent
        subtree.parent, parent.parent = (None, None)
        sibling.parent = grandparent
        if grandparent:
            new_root = grandparent
        else: # As a result of the special case of removing a first-order branch
            new_root = sibling
        return new_root

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
        return self


    def prune_and_regraft(self):
        """Prune and regraft tree manipulation.
        Detach a randomly selected subtree and reattach it at at
        another randomly chosen edge.
        """
        subtree = random.choice([n for n in anytree.PreOrderIter(self) if not n.is_root])
        detached = self.detach(subtree)
        rejoin_at = random.choice([n for n in anytree.PreOrderIter(detached)])
        return detached.attach(rejoin_at, subtree)


    def equal(self, tree):
        """Recursive version of __eq__.
        Return True if this tree and other tree have same nodes and topology.
        """
        for a, b in zip(anytree.PreOrderIter(self), anytree.PreOrderIter(tree)):
            if not a == b:
                return False
        return True


    @property
    def is_binary(self):
        """Return True if this is a proper binary tree"""
        return all([len(n.children) == 2 for n in anytree.PreOrderIter(self) if not n.is_leaf])


    def __eq__(self, node):
        return all([
            self.children == node.children,
            self.label == node.label,
            self.length == node.length
        ])


    def __repr__(self):
        return "PhyloNode({}, {}, state={})".format(self.label, self.length, self.state)

