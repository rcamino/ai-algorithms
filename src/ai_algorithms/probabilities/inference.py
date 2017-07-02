from collections import deque

from ai_algorithms.probabilities.marginal import marginalize


def inference_by_enumeration(joint_probability, query, evidence):
    """
    Infer a joint probability over a list of variables given some evidence.
    :param joint_probability: multidimensional dictionary
    :param query: indicates for every variable if it must be inferred; can be any sequential structure of booleans
    :param evidence: indicates for every variable if it has an observation (or None); can be any sequential structure
    :return: inferred probability; multidimensional dictionary
    """
    filtered_joint_probability = observe(joint_probability, evidence)

    hidden_variables = deque()
    for include_in_query, evidence_value in zip(query, evidence):
        if evidence_value is None:
            hidden_variables.append(not include_in_query)
    return marginalize(filtered_joint_probability, hidden_variables)


def observe(joint_probability, evidence):
    """
    Fix values in the joint probability table according to the evidence.
    :param joint_probability: multidimensional dictionary
    :param evidence: indicates for every variable if it has an observation (or None); can be any sequential structure
    :return: multidimensional dictionary
    """
    return observe_recursion(joint_probability, deque(evidence))


def observe_recursion(joint_probability, evidence):
    """
    Fix values in the joint probability table according to the evidence.
    :param joint_probability: multidimensional dictionary
    :param evidence: indicates for every variable if it has an observation (or None); collections.deque
    :return: multidimensional dictionary
    """
    if len(evidence) == 0:
        return joint_probability
    else:
        evidence_value = evidence.popleft()

        if evidence_value is not None:
            probability = observe_recursion(joint_probability[evidence_value], evidence)
        else:
            probability = {}
            for value in joint_probability.keys():
                probability[value] = observe_recursion(joint_probability[value], evidence)

        evidence.appendleft(evidence_value)
        return probability
