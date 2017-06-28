def create_initial_values(model):
    initial_values = {}
    for state in model.states():
        initial_values[state] = 0.0
    return initial_values


def value_iteration_step(model, values, discount=1.0):
    next_values = create_initial_values(model)
    for state_from in model.states():
        max_value = 0
        max_action = None
        for action in model.actions_from(state_from):
            value = 0.0
            for state_to in model.states_from(state_from, action):
                value += model.probability(state_from, action, state_to) \
                         * (model.reward(state_from, action, state_to) + discount * values[state_to])
            if max_action is None or value > max_value:
                max_action = action
                max_value = value
        if max_action is not None:
            next_values[state_from] = max_value
    return next_values


def value_iteration(model, initial_values=None, discount=1.0, iterations=1000, tolerance=1e-6):
    if initial_values is None:
        values = create_initial_values(model)
    else:
        values = initial_values
    iteration = 0
    while iteration < iterations:
        next_values = value_iteration_step(model, values, discount)
        max_change = 0
        for s in values.keys():
            max_change = max(max_change, abs(next_values[s] - values[s]))
        if max_change < tolerance:
            break
        values = next_values
        iteration += 1
    return values
