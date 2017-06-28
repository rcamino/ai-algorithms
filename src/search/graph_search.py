from collections import namedtuple, deque


Node = namedtuple("Node", ["state", "action", "previous", "cost"])


def reconstruct_path(last_node):
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


def graph_search(problem, strategy):
    closed = set()
    candidates = strategy.create_candidates()
    start_node = Node(state=problem.start_state(), action=None, previous=None, cost=0)
    strategy.add_to_candidates(start_node, candidates)
    while candidates.has_candidates():
        node = candidates.next_candidate()
        if problem.is_goal_state(node.state):
            return reconstruct_path(node)
        if node.state not in closed:
            closed.add(node.state)
            for action, child_state, cost in problem.transitions(node.state):
                child_node = Node(state=child_state, action=action, previous=node, cost=node.cost + cost)
                strategy.add_to_candidates(child_node, candidates)
    return None
