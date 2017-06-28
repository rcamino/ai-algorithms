def create_empty_value(mdp):
    v_0 = {}
    for state in mdp.states():
        v_0[state] = 0.0
    return v_0


def value_iteration_step(mdp, v_k, gamma=1.0):
    v_k_plus_1 = create_empty_value(mdp)
    for s1 in mdp.states():
        max_v = 0
        max_a = None
        for a in mdp.actions_from(s1):
            v = 0.0
            for s2 in mdp.states_from(s1, a):
                v += mdp.probability(s1, a, s2) * (mdp.reward(s1, a, s2) + gamma * v_k[s2])
            if max_a is None or v > max_v:
                max_a = a
                max_v = v
        if max_a is not None:
            v_k_plus_1[s1] = max_v
    return v_k_plus_1


def value_iteration(mdp, v_0=None, gamma=1.0, iterations=1000, tolerance=1e-6):
    if v_0 is None:
        v_k = create_empty_value(mdp)
    else:
        v_k = v_0
    i = 0
    while i < iterations:
        v_k_plus_1 = value_iteration_step(mdp, v_k, gamma)
        max_change = 0
        for s in v_k.keys():
            max_change = max(max_change, abs(v_k_plus_1[s] - v_k[s]))
        if max_change < tolerance:
            break
        v_k = v_k_plus_1
        i += 1
    return v_k
