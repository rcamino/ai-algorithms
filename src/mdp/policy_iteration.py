import random

from value_iteration import create_empty_value


def create_random_policy(model, rnd=None):
    if rnd is None:
        rnd = random
    pi_0 = {}
    for s in model.states():
        actions = model.actions_from(s)
        if len(actions) > 0:
            pi_0[s] = rnd.choice(actions)
    return pi_0


def policy_evaluation(model, pi, v_pi_0=None, gamma=1.0, iterations=1000, tolerance=1e-6):
    if v_pi_0 is None:
        v_pi_k = create_empty_value(model)
    else:
        v_pi_k = v_pi_0
    i = 0
    while i < iterations:
        v_pi_k_plus_1 = create_empty_value(model)
        for s1 in pi.keys():
            a = pi[s1]
            v = 0
            for s2 in model.states_from(s1, a):
                v += model.probability(s1, a, s2) * (model.reward(s1, a, s2) + gamma * v_pi_k[s2])
            v_pi_k_plus_1[s1] = v
        max_change = 0
        for s in pi.keys():
            max_change = max(max_change, abs(v_pi_k_plus_1[s] - v_pi_k[s]))
        if max_change < tolerance:
            break
        v_pi_k = v_pi_k_plus_1
        i += 1
    return v_pi_k


def policy_improvement(model, v_pi_i, gamma=1.0):
    pi_i_plus_1 = {}
    for s1 in model.states():
        max_a = None
        max_v_a = 0
        for a in model.actions_from(s1):
            v_a = 0
            for s2 in model.states_from(s1, a):
                v_a += model.probability(s1, a, s2) * (model.reward(s1, a, s2) + gamma * v_pi_i[s2])
            if max_a is None or v_a > max_v_a:
                max_a = a
                max_v_a = v_a
        if max_a is not None:
            pi_i_plus_1[s1] = max_a
    return pi_i_plus_1


def policy_iteration(model, pi_0=None, gamma=1.0, iterations=1000, tolerance=1e-6):
    if pi_0 is None:
        pi_i = create_random_policy(model)
    else:
        pi_i = pi_0
    i = 0
    while i < iterations:
        v_pi_i = policy_evaluation(model, pi_i, gamma=gamma, iterations=iterations, tolerance=tolerance)
        pi_i_plus_1 = policy_improvement(model, v_pi_i, gamma)
        if all([pi_i_plus_1[s] == pi_i[s] for s in pi_i.keys()]):
            break
        pi_i = pi_i_plus_1
        i += 1
    return pi_i
