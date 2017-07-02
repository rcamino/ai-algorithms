class Table(object):

    def __init__(self, children_by_value, names, values_by_name):
        self.children_by_value = children_by_value
        self.names = names
        self.values_by_name = values_by_name

    def __getitem__(self, value):
        return self.children_by_value[value]

    def __len__(self):
        return len(self.children_by_value)

    def keys(self):
        return self.children_by_value.keys()

    def create_query(self, names):
        selected_names = set(names)
        return [name in selected_names for name in self.names]

    def create_evidence(self, observations_by_name):
        evidence = []
        for name in self.names:
            if name in observations_by_name:
                evidence.append(observations_by_name[name])
            else:
                evidence.append(None)
        return evidence


def from_tuple_dictionary(tuple_probability, names):
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
