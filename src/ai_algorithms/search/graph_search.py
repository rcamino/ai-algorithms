from collections import namedtuple, deque


Node = namedtuple("Node", ["state", "action", "previous", "cost"])


def reconstruct_path(last_node):
    """
    Build a sequence of actions going backwards to the start state from a given state.
    :param last_node: last visited state node
    :return: action list
    """
    action = last_node.action
    node = last_node.previous
    action_stack = deque()
    while node is not None:
        action_stack.append(action)
        action = node.action
        node = node.previous
    actions = []
    while len(action_stack) > 0:
        actions.append(action_stack.pop())
    return actions


def graph_search(model, strategy):
    """
    Finds in a graph the path of actions between the starting state and a goal using the lowest accumulated cost.
    :param model: must implement ai_algorithms.model.{Model,Deterministic,CostAware}
    :param strategy: must implement ai_algorithms.search.strategy.Strategy
    :return: action sequence or None if the problem cannot be solved
    """
    closed = set()
    candidates = strategy.create_candidates()
    start_node = Node(state=model.start_state(), action=None, previous=None, cost=0)
    strategy.add_to_candidates(start_node, candidates)
    while candidates.has_candidates():
        node = candidates.next_candidate()
        if model.is_goal_state(node.state):
            return reconstruct_path(node)
        if node.state not in closed:
            closed.add(node.state)
            for action in model.actions_from(node.state):
                child_state = model.next_state(node.state, action)
                cost = node.cost + model.cost(node.state, action, child_state)
                child_node = Node(state=child_state, action=action, previous=node, cost=cost)
                strategy.add_to_candidates(child_node, candidates)
    return None
