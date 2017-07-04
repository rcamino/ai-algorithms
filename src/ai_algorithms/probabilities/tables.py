class Table(object):
    """
    This object works like a multidimensional dictionary of probabilities,
    but have the names of the probability table columns and the possible values for each column.
    Also includes some extra functions for creating queries.
    """

    def __init__(self, children_by_value, names, values_by_name):
        """
        :param children_by_value: probabilities for the first column; multidimensional dictionary of probabilities
        :param names: name of every column; list of strings
        :param values_by_name: possible values for every column; dictionary of strings to string lists
        """
        self.children_by_value = children_by_value
        self.names = names
        self.values_by_name = values_by_name

    def __getitem__(self, value):
        """
        Probability table for a value in the fist column.
        Implemented for compatibility with dictionaries.
        :param value: value in the fist column; must be a hashable object
        :return: ai_algorithms.probabilities.tables.Table or float
        """
        return self.children_by_value[value]

    def __len__(self):
        """
        Amount of values for the first column.
        Implemented for compatibility with dictionaries.
        :return: int
        """
        return len(self.children_by_value)

    def keys(self):
        """
        Values for the first column.
        Implemented for compatibility with dictionaries.
        :return: string list
        """
        return self.children_by_value.keys()

    def create_query(self, names):
        """
        Creates a sequence of booleans indicating for every column if it must be fetched.
        The columns are sorted according to the order used in the table creation.
        :param names: selected variable names
        :return: list of booleans
        """
        selected_names = set(names)
        return [name in selected_names for name in self.names]

    def create_evidence(self, observations_by_name):
        """
        Creates a sequence that indicates for every column if it has an observation or not.
        The columns are sorted according to the order used in the table creation.
        :param observations_by_name: dictionary of column name to value
        :return: list of values, some positions can be None
        """
        evidence = []
        for name in self.names:
            if name in observations_by_name:
                evidence.append(observations_by_name[name])
            else:
                evidence.append(None)
        return evidence


def from_tuple_dictionary(tuple_probability, names):
    """
    Create a probability table from a dictionary of tuples to probabilities.

    Example: P(A | B) with A, B in {0, 1}

    tuple_probability = {
        (0, 0): 0.6,
        (1, 0): 0.4,
        (0, 1): 0.2,
        (1, 1): 0.8,
    }

    table = from_tuple_dictionary(tuple_probability, ["A", "B"])

    :param tuple_probability: dictionary with tuples of hashable objects as keys and float as values
    :param names: string list
    :return: ai_algorithms.probabilities.tables.Table
    """
    size = len(names)
    probability = {}
    for values, p in tuple_probability.items():
        assert len(values) == size
        query = probability
        for i in xrange(size - 1):
            value = values[i]
            if value not in query:
                query[value] = {}
            query = query[value]
        last_value = values[size - 1]
        query[last_value] = p
    return from_dictionary(probability, names)


def from_dictionary(probability, names):
    """
    Create a probability table from a multidimensional dictionary of probabilities.

    Example: P(A | B) with A, B in {0, 1}

    probability = {
        0: {0: 0.6, 1: 0.4}
        1: {0: 0.2, 1: 0.8}
    }

    table = from_dictionary(probability, ["A", "B"])

    :param probability: multidimensional dictionary of probabilities
    :param names: string list
    :return: ai_algorithms.probabilities.tables.Table
    """
    if len(names) == 0:
        return probability
    children_by_value = {}
    values = probability.keys()
    values_by_name = {names[0]: values}
    children_have_names = len(names) > 1
    for value in values:
        child = from_dictionary(probability[value], names[1:])
        children_by_value[value] = child
        if children_have_names:
            for child_name, child_values in child.values_by_name.items():
                if child_name not in values_by_name:
                    values_by_name[child_name] = child_values
                else:
                    assert values_by_name[child_name] == child_values
    return Table(children_by_value, names, values_by_name)
