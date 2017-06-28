import random

from value_iteration import create_initial_values


def create_random_policy(model, random_state=None):
    if random_state is None:
        random_state = random.Random()
    random_policy = {}
    for state in model.states():
        actions = model.actions_from(state)
        if len(actions) > 0:
            random_policy[state] = random_state.choice(actions)
    return random_policy


def policy_evaluation(model, policy, initial_values=None, discount=1.0, iterations=1000, tolerance=1e-6):
    if initial_values is None:
        values = create_initial_values(model)
    else:
        values = initial_values
    iteration = 0
    while iteration < iterations:
        next_values = create_initial_values(model)
        for state_from in policy.keys():
            action = policy[state_from]
            value = 0
            for state_to in model.states_from(state_from, action):
                value += model.probability(state_from, action, state_to) \
                         * (model.reward(state_from, action, state_to) + discount * values[state_to])
            next_values[state_from] = value
        max_change = 0
        for state in policy.keys():
            max_change = max(max_change, abs(next_values[state] - values[state]))
        if max_change < tolerance:
            break
        values = next_values
        iteration += 1
    return values


def policy_improvement(model, values, discount=1.0):
    next_policy = {}
    for state_from in model.states():
        max_action = None
        max_value = 0
        for action in model.actions_from(state_from):
            value = 0
            for state_to in model.states_from(state_from, action):
                value += model.probability(state_from, action, state_to) \
                         * (model.reward(state_from, action, state_to) + discount * values[state_to])
            if max_action is None or value > max_value:
                max_action = action
                max_value = value
        if max_action is not None:
            next_policy[state_from] = max_action
    return next_policy


def policy_iteration(model, initial_policy=None, discount=1.0, iterations=1000, tolerance=1e-6):
    if initial_policy is None:
        policy = create_random_policy(model)
    else:
        policy = initial_policy
    iteration = 0
    while iteration < iterations:
        values = policy_evaluation(model, policy, discount=discount, iterations=iterations, tolerance=tolerance)
        next_policy = policy_improvement(model, values, discount)
        if all([next_policy[state] == policy[state] for state in policy.keys()]):
            break
        policy = next_policy
        iteration += 1
    return policy
