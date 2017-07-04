class BayesianNetworkNode(object):

    def __init__(self, name, parents, probability):
        """
        The name of this node and from the parents must coincide with the probability table column names.
        Children can be added after creation.
        :param name: node name string
        :param parents: collection of parent nodes; must be a BayesianNetworkNode
        :param probability: ai_algorithms.probabilities.tables.Table
        """
        self.name = name
        self.parents = set(parents)
        self.probability = probability
        self.children = set()

    def __repr__(self):
        """
        Shows the name of this node.
        :return: string
        """
        return self.name

    def add_child(self, child):
        """
        Add a child node.
        :param child: BayesianNetworkNode
        """
        self.children.add(child)

    def is_child_of(self, parent):
        """
        Tells if the given node is a parent of this node.
        :param parent: BayesianNetworkNode
        :return: True if the given node is a parent of this node, False otherwise
        """
        return parent in self.parents

    def is_parent_of(self, child):
        """
        Tells if the given node is a child of this node.
        :param child: BayesianNetworkNode
        :return: True if the given node is a child of this node, False otherwise
        """
        return child in self.children

    def neighbors(self):
        """
        Gives the collection of neighbors of this node (parents and children).
        :return: collection of BayesianNetworkNode
        """
        return self.children + self.parents
