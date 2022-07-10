import newick
import anytree
import random
import copy
import fulgurite.likelihood as likelihood


class PhyloNodeError(Exception):
    pass


class PhyloTree:
    """Container class for a tree of PhyloNodes.
    This allows manipulation of the tree without needing to keep track of the root node
    externally when attaching and detaching nodes. Also allows storing info which is
    relevant for the whole tree such as a list of nodes so selection can be O(1), the
    list of states, etc."""
    def __init__(self, root, states, nodes=None):
        self.root = root
        self.states = states
        if not nodes:
            self.nodes = [n for n in anytree.PreOrderIter(self.root)]
        else:
            self.nodes = nodes

    @classmethod
    def from_string(cls, newickstr, states):
        root = PhyloNode.from_string(newickstr, states=states)
        return PhyloTree(root, states)

    def attach(self, subtree, loc):
        if loc not in self.nodes:
            raise PhyloNodeError("Node not in tree!")
        loc.attach(loc, subtree) # Yuck, fix
        return self

    def detach(self, subtree):
        if subtree not in self.nodes:
            raise PhyloNodeError("Node not in tree!")
        self.root = self.root.detach(subtree)
        return self

    def regraft(self, subtree, loc):
        self.detach(subtree)
        self.attach(subtree, loc)
        return self

    @property
    def leaves(self):
        return self.root.leaves

    def __str__(self):
        info = "PhyloTree with {} nodes and {} states\n".format(
            len(self.nodes), len(set(self.states.values()))
        )
        treestruc = str(anytree.RenderTree(self.root))
        return info + treestruc


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
        newick_root = newick.loads(newick_str)[0]
        phylo_root = fromnode(newick_root, None)
        stack = [(newick_root, phylo_root)]
        while stack:
            newick_n, phylo_n = stack.pop()
            for c in newick_n.descendants:
                stack.append( (c, fromnode(c, phylo_n)) )
        # Attach states
        if states:
            for node in anytree.PreOrderIter(phylo_root):
                if node.label in states:
                    node.state = states[node.label]
        return phylo_root


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

    ## TODO: This is why I need a wrapper PhyloTree class (as well as being able to
    ## refer to root node without the hacky system above..), the wrapper can store
    ## the total number and names etc of states which are already stored in tree
    ## TODO: Convert likelihood implementation to use the likelihood slots in the
    ## tree. This means when the tree changes topology we don't need to recalculate
    ## the likelihood for the whole tree, further speeding up the calculation
    ## TODO: Use log likelihoods. This is important according to the big brains as
    ## often the likelihood values are very small and you can have numerical problems.
    ## Although according to https://stackoverflow.com/a/48189386 this doesn't matter
    ## with Python (and implies with doesn't matter with any language running on modern
    ## hardware which uses IEEE-754):
    ## >> If the machine you are running on happens not to be using
    ## >> IEEE-754, it is still likely that computing x/y directly will
    ## >> produce a better result than np.exp(np.log(x)-np.log(y)). The
    ## >> former is a single operation computing a basic function in
    ## >> hardware that was likely reasonably designed. The latter is
    ## >> several operations computing complicated functions in software
    ## >> that is difficult to make accurate using common hardware
    ## >> operations.
    def get_likelihood(self, Q, tip_states):
        """Calculate the likelihood of subtree from this node given rate matrix Q."""
        return likelihood.tree_likelihood(Q, self, tip_states)


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

