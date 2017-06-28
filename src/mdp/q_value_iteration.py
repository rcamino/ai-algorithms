def create_empty_q_value(model):
    q_0 = {}
    for s in model.states():
        q_0[s] = {}
        for a in model.actions_from(s):
            q_0[s][a] = 0
    return q_0


def q_value_iteration_step(model, q_k, discount=1.0):
    q_k_plus_1 = create_empty_q_value(model)
    for s1 in model.states():
        for a in model.actions_from(s1):
            q = 0.0
            for s2 in model.states_from(s1, a):
                max_a = 0
                for a2 in model.actions_from(s2):
                    max_a = max(max_a, q_k[s2][a2])
                q += model.probability(s1, a, s2) * (model.reward(s1, a, s2) + discount * max_a)
            q_k_plus_1[s1][a] = q
    return q_k_plus_1


def q_value_iteration(model, q_0=None, discount=1.0, iterations=1000, tolerance=1e-6):
    if q_0 is None:
        q_k = create_empty_q_value(model)
    else:
        q_k = q_0
    i = 0
    while i < iterations:
        q_k_plus_1 = q_value_iteration_step(model, q_k, discount)
        max_change = 0
        for s in q_k.keys():
            for a in q_k[s].keys():
                max_change = max(max_change, abs(q_k_plus_1[s][a] - q_k[s][a]))
        if max_change < tolerance:
            break
        q_k = q_k_plus_1
        i += 1
    return q_k


def policy_from_q_values(model, q_values):
    pi = {}
    for s in model.states():
        max_q_value = 0.0
        max_action = None
        for a in model.actions_from(s):
            q_value = q_values[s][a]
            if max_action is None or q_value > max_q_value:
                max_action = a
                max_q_value = q_value
        if max_action is not None:
            pi[s] = max_action
    return pi
