from collections import deque


def marginalize(joint_probability, query):
    """
    Marginalize a joint probability over a list of variables.
    :param joint_probability: multidimensional dictionary
    :param query: indicates for every variable if it must be marginalized; can be any sequential structure of booleans
    :return: marginal probability; multidimensional dictionary
    """
    query_queue = deque()
    count = 0
    for marginalize_variable in query:
        if marginalize_variable:
            count += 1
        query_queue.append(marginalize_variable)
    return marginalize_recursion(joint_probability, query_queue, count)


def marginalize_recursion(joint_probability, query, remaining):
    """
    Recursively marginalize a joint probability over a list of variables.
    :param joint_probability: multidimensional dictionary
    :param query: indicates for every variable if it must be marginalized; collections.deque of booleans
    :param remaining: remaining True elements in query; int >= 0
    :return: marginal probability; multidimensional dictionary
    """
    if len(query) == 0:
        return joint_probability
    else:
        marginalize_variable = query.popleft()

        if marginalize_variable:
            child_joint_probabilities = []
            for value in joint_probability.keys():
                child_joint_probabilities.append(marginalize_recursion(joint_probability[value], query, remaining - 1))

            marginal_probability = sum_joint_probabilities(child_joint_probabilities, len(query) - (remaining - 1))
        else:
            marginal_probability = {}
            for value in joint_probability.keys():
                marginal_probability[value] = marginalize_recursion(joint_probability[value], query, remaining)

        query.appendleft(marginalize_variable)

        return marginal_probability


def sum_joint_probabilities(joint_probabilities, size):
    """
    Sums a list of joint probability tables.
    :param joint_probabilities: list of joint probability tables; every table is a multidimensional dictionary
    :param size: amount of variables of the joint probability table; int >= 0
    :return: joint probability table (or entry in base case); multidimensional dictionary (or float in base case)
    """
    if size == 0:
        return sum(joint_probabilities)
    else:
        first_joint_probability = joint_probabilities[0]
        aggregated_probability = {}
        for value in first_joint_probability.keys():
            aggregated_probability[value] = sum_joint_probabilities(
                [joint_probability[value] for joint_probability in joint_probabilities], size - 1)
        return aggregated_probability
