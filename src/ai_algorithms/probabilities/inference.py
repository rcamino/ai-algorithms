from collections import deque

from ai_algorithms.probabilities.marginal import marginalize


def inference_by_enumeration(joint_probability, query, evidence):
    filtered_joint_probability = filter_evidence(joint_probability, deque(evidence))

    hidden_variables = deque()
    for include_in_query, evidence_value in zip(query, evidence):
        if evidence_value is None:
            hidden_variables.append(not include_in_query)
    return marginalize(filtered_joint_probability, hidden_variables)


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
