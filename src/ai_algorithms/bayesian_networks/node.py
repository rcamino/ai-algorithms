class BayesianNetworkNode(object):

    def __init__(self, name, parents, probability):
        self.name = name
        self.parents = set(parents)
        self.probability = probability
        self.children = set()

    def __repr__(self):
        return self.name

    def add_child(self, child):
        self.children.add(child)

    def is_child_of(self, parent):
        return parent in self.parents

    def is_parent_of(self, child):
        return child in self.children

    def neighbors(self):
        return self.children + self.parents
