def create_initial_q_values(model):
    """
    Returns a dictionary that gives zero reward for every valid state and action pair.
    :param model: must implement ai_algorithms.model.{Model,CompleteSpace,Stochastic,RewardAware}
    :return: dictionary of dictionaries of values for every state and action
    """
    initial_q_values = {}
    for state in model.states():
        initial_q_values[state] = {}
        for action in model.actions_from(state):
            initial_q_values[state][action] = 0
    return initial_q_values


def q_value_iteration_step(model, q_values, discount=1.0):
    """
    Creates new q-values from previous ones.
    :param model: must implement ai_algorithms.model.{Model,CompleteSpace,Stochastic,RewardAware}
    :param q_values: result from previous step
    :param discount: factor in [0, 1] to indicate how much of the value should be retained from the previous step
    :return: dictionary of dictionaries of values for every state and action
    """
    next_q_values = create_initial_q_values(model)
    for state_from in model.states():
        for action in model.actions_from(state_from):
            q_value = 0.0
            for state_to in model.states_from(state_from, action):
                max_q_value = 0
                for second_action in model.actions_from(state_to):
                    max_q_value = max(max_q_value, q_values[state_to][second_action])
                q_value += model.probability(state_from, action, state_to) \
                           * (model.reward(state_from, action, state_to) + discount * max_q_value)
            next_q_values[state_from][action] = q_value
    return next_q_values


def q_value_iteration(model, initial_q_values=None, discount=1.0, iterations=1000, tolerance=1e-6):
    """
    Returns the optimum reward value that can be obtained from every state and action pair.
    :param model: must implement ai_algorithms.model.{Model,CompleteSpace,Stochastic,RewardAware}
    :param initial_q_values: initial estimation of the result
    :param discount: factor in [0, 1] to indicate how much of the value should be retained between iterations
    :param iterations: maximum number of iterations
    :param tolerance: if the maximum difference between all the values is less than this number the algorithm stops
    :return: dictionary of dictionaries of values for every state and action
    """
    if initial_q_values is None:
        q_values = create_initial_q_values(model)
    else:
        q_values = initial_q_values
    iteration = 0
    while iteration < iterations:
        next_q_values = q_value_iteration_step(model, q_values, discount)
        max_change = 0
        for state in q_values.keys():
            for action in q_values[state].keys():
                max_change = max(max_change, abs(next_q_values[state][action] - q_values[state][action]))
        if max_change < tolerance:
            break
        q_values = next_q_values
        iteration += 1
    return q_values


def policy_from_q_values(q_values):
    """
    Calculates the optimal policy from q-values.
    :param q_values: the optimum reward value that can be obtained from every state and action pair
    :return: dictionary of state to action
    """
    policy = {}
    for state in q_values.keys():
        max_q_value = 0.0
        max_action = None
        for action in q_values[state].keys():
            q_value = q_values[state][action]
            if max_action is None or q_value > max_q_value:
                max_action = action
                max_q_value = q_value
        if max_action is not None:
            policy[state] = max_action
    return policy
