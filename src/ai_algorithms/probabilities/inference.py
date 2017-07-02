from collections import deque


def inference_by_enumeration(joint_probability, query, evidence):
    assert len(query) == len(evidence)

    filtered_joint_probability = filter_evidence(joint_probability, deque(evidence))

    filtered_query = deque()
    for include_in_query, evidence_value in zip(query, evidence):
        if evidence_value is None:
            filtered_query.append(include_in_query)
    collapsed_probability, _ = collapse_hidden(filtered_joint_probability, filtered_query)
    return collapsed_probability


def filter_evidence(joint_probability, evidence):
    if len(evidence) == 0:
        return joint_probability
    else:
        evidence_value = evidence.popleft()

        if evidence_value is not None:
            probability = filter_evidence(joint_probability[evidence_value], evidence)
        else:
            probability = {}
            for value in joint_probability.keys():
                probability[value] = filter_evidence(joint_probability[value], evidence)

        evidence.appendleft(evidence_value)
        return probability


def collapse_hidden(joint_probability, query):
    if len(query) == 0:
        return joint_probability, 0
    else:
        include_in_query = query.popleft()

        if include_in_query:
            collapsed_probability = {}
            for value in joint_probability.keys():
                collapsed_probability[value], times_collapsed = collapse_hidden(joint_probability[value], query)
        else:
            sub_joint_probabilities = []
            for value in joint_probability.keys():
                sub_joint_probability, times_collapsed = collapse_hidden(joint_probability[value], query)
                sub_joint_probabilities.append(sub_joint_probability)

            collapsed_probability = collapse(sub_joint_probabilities, len(query) - times_collapsed)
            times_collapsed += 1

        query.appendleft(include_in_query)

        return collapsed_probability, times_collapsed


def collapse(joint_probabilities, size):
    if size == 0:
        return sum(joint_probabilities)
    else:
        first_joint_probability = joint_probabilities[0]
        collapsed_probability = {}
        for key in first_joint_probability.keys():
            collapsed_probability[key] = collapse([table[key] for table in joint_probabilities], size - 1)
        return collapsed_probability
