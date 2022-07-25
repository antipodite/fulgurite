import newick
import anytree
import random
import collections
import fulgurite.mkmodel as mkmodel


def reverselevelorder(node):
    """Walk the tree in reverse level order, in O(n) time"""
    leaves = sorted(node.leaves, key=lambda n: n.height)
    queue = collections.deque([node.root])
    result = collections.deque()
    while queue:
        this = queue.popleft()
        result.appendleft(this)
        queue.extend(this.children)
    return result


class PhyloNodeError(Exception):
    pass


class PhyloTree:
    """Container class for a tree of PhyloNodes.
    This allows manipulation of the tree without needing to keep track of the root node
    externally when attaching and detaching nodes. Also allows storing info which is
    relevant for the whole tree such as a list of nodes so selection can be O(1), the
    list of states, etc."""
    def __init__(self, root, states=dict(), nodes=None):
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
        loc.attach(subtree) # Yuck, fix
        return self

    def detach(self, subtree):
        if subtree not in self.nodes:
            raise PhyloNodeError("Node not in tree!")
        self.root = subtree.detach()
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
    def from_string(cls, newick_str, states=dict()):
        """Build a PhyloNode tree from a Newick format string
        newick_str: a tree defined as a Newick formatted string
        states: a dict of {label: state} where state is an integer from 0 -> n states - 1
        TODO: Move this code into the PhyloTree wrapper class
        """
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
            n_states = max(states.values()) + 1
            for node in anytree.PreOrderIter(phylo_root):
                if node.label in states:
                    this_state = states[node.label]
                    node.state = this_state
                    # Preset likelihoods: likelihood for this state at this node is 1,
                    # the rest are 0
                    likelihoods = [0 for i in range(n_states)]
                    likelihoods[this_state] = 1
                    node.likelihoods = likelihoods
                else:
                    node.likelihoods = [None for i in range(n_states)]
        return phylo_root


    def attach(self, subtree):
        """Attach subtree b to node a, making necessary changes to maintain
        binary branching structure.
        """
        # if self.is_leaf:
        #     raise PhyloNodeError("Can't attach subtree to tip {}".format(self.label))
        self.children = [PhyloNode(children=self.children), subtree]
        return self

    def detach(self):
        """Detach this node.
        We maintain binary structure by collapsing the remaining unary branch.
        """
        if self.is_root:
            raise PhyloNodeError("Can't detach root node")
        sibling = self.siblings[0]
        parent = self.parent
        grandparent = parent.parent
        self.parent, parent.parent = (None, None)
        sibling.parent = grandparent
        if grandparent:
            new_root = grandparent
        else: # As a result of the special case of removing a first-order branch
            new_root = sibling
        return new_root

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
    def get_likelihood(self, Q):
        """Calculate the likelihood of subtree from this node given rate matrix Q."""
        return mkmodel.subtree_likelihood(self, Q)
    
    def equal(self, subtree):
        """Recursive version of __eq__.
        Return True if this tree and other tree have same nodes and topology.
        """
        for a, b in zip(anytree.PreOrderIter(self), anytree.PreOrderIter(subtree)):
            if not a == b:
                return False
        return True

    def reverse_level_walk(self):
        for n in reverselevelorder(self):
            yield n

    @property
    def is_binary(self):
        """Return True if this is a proper binary subtree"""
        return all([len(n.children) == 2 for n in anytree.PreOrderIter(self) if not n.is_leaf])

    def __eq__(self, node):
        return all([
            self.children == node.children,
            self.label == node.label,
            self.length == node.length
        ])

    def __repr__(self):
        template = "PhyloNode({}, {}, state={}, likelihoods={})"
        return template.format(self.label, self.length, self.state, self.likelihoods)

