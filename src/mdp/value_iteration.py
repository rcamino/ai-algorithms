def create_empty_value(model):
    v_0 = {}
    for state in model.states():
        v_0[state] = 0.0
    return v_0


def value_iteration_step(model, v_k, discount=1.0):
    v_k_plus_1 = create_empty_value(model)
    for s1 in model.states():
        max_v = 0
        max_a = None
        for a in model.actions_from(s1):
            v = 0.0
            for s2 in model.states_from(s1, a):
                v += model.probability(s1, a, s2) * (model.reward(s1, a, s2) + discount * v_k[s2])
            if max_a is None or v > max_v:
                max_a = a
                max_v = v
        if max_a is not None:
            v_k_plus_1[s1] = max_v
    return v_k_plus_1


def value_iteration(model, v_0=None, discount=1.0, iterations=1000, tolerance=1e-6):
    if v_0 is None:
        v_k = create_empty_value(model)
    else:
        v_k = v_0
    i = 0
    while i < iterations:
        v_k_plus_1 = value_iteration_step(model, v_k, discount)
        max_change = 0
        for s in v_k.keys():
            max_change = max(max_change, abs(v_k_plus_1[s] - v_k[s]))
        if max_change < tolerance:
            break
        v_k = v_k_plus_1
        i += 1
    return v_k
