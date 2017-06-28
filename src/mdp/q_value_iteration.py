def create_empty_q_value(mdp):
    q_0 = {}
    for s in mdp.states():
        q_0[s] = {}
        for a in mdp.actions_from(s):
            q_0[s][a] = 0
    return q_0


def q_value_iteration_step(mdp, q_k, gamma=1.0):
    q_k_plus_1 = create_empty_q_value(mdp)
    for s1 in mdp.states():
        for a in mdp.actions_from(s1):
            q = 0.0
            for s2 in mdp.states_from(s1, a):
                max_a = 0
                for a2 in mdp.actions_from(s2):
                    max_a = max(max_a, q_k[s2][a2])
                q += mdp.probability(s1, a, s2) * (mdp.reward(s1, a, s2) + gamma * max_a)
            q_k_plus_1[s1][a] = q
    return q_k_plus_1


def q_value_iteration(mdp, q_0=None, gamma=1.0, iterations=1000, tolerance=1e-6):
    if q_0 is None:
        q_k = create_empty_q_value(mdp)
    else:
        q_k = q_0
    i = 0
    while i < iterations:
        q_k_plus_1 = q_value_iteration_step(mdp, q_k, gamma)
        max_change = 0
        for s in q_k.keys():
            for a in q_k[s].keys():
                max_change = max(max_change, abs(q_k_plus_1[s][a] - q_k[s][a]))
        if max_change < tolerance:
            break
        q_k = q_k_plus_1
        i += 1
    return q_k
