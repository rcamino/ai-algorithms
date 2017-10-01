import math
import random

from ai_algorithms.environment.agent_strategies.agent_strategy import AgentStrategy


class Node(object):
    """
    Internal structure for the Monte Carlo Tree Search.
    """

    def __init__(self, state, agent, pending_actions, parent=None):
        """
        :param state: current state; must be a hashable object
        :param agent: agent taking the next action; must implement ai_algorithms.environment.agent.Agent
        :param pending_actions: actions that were not expanded yet; list of hashable objects
        :param parent: Node
        """
        self.state = state
        self.agent = agent
        self.pending_actions = pending_actions

        self.parent = parent

        self.children_by_action = {}
        self.visits = 0
        self.score = 0


class MonteCarloTreeSearch(AgentStrategy):
    """
    Monte Carlo Tree Search (MCTS) implemented with Upper Confidence bounds applied to Trees (UCT).
    """

    def __init__(self, iterations=1000, exploration=None, random_state=None):
        """
        :param iterations: maximum number of iterations
        :param exploration: balance between taking the best known action and exploring others; default sqrt(2)
        :param random_state: random.RandomState; if None, default random state will be used
        """
        self.iterations = iterations

        if exploration is None:
            exploration = math.sqrt(2.0)
        self.exploration = exploration

        if random_state is None:
            random_state = random.Random()
        self.random_state = random_state

    def next_action(self, environment, state, agent):
        """
        Decides the next action for the agent to chose from the possible actions in the environment.
        :param environment: must implement ai_algorithms.environment.models.Environment
        :param state: current state; must be a hashable object
        :param agent: must implement ai_algorithms.environment.agent.Agent
        :return: chosen action; must be a hashable object
        """
        root = Node(state, agent, environment.actions_from(state))

        for iteration in range(self.iterations):
            node = self.selection(root)
            if not environment.is_goal_state(node.state):
                node = self.expansion(environment, node)
            winners = self.simulation(environment, node.state, node.agent)
            self.backpropagation(agent, node, winners)

        return self.best_action(root)

    def selection(self, root):
        """
        Selects the next tree node to expand from another tree node.
        :param root: Node
        :return: Node
        """
        node = root
        while len(node.pending_actions) == 0 and len(node.children_by_action) > 0:
            node = sorted(node.children_by_action.values(),
                          key=lambda child: self.upper_confidence_bound(node, child))[-1]
        return node

    def upper_confidence_bound(self, node, child):
        """
        Upper Confidence Bound for the score exploring the indicated child node.
        :param node: origin of the selection; Node
        :param child: candidate child; Node
        :return: float
        """
        return child.score / child.visits + self.exploration + math.sqrt(math.log(node.visits) / child.visits)

    def expansion(self, environment, node):
        """
        Expand the node taking one of its pending actions.
        :param environment: must implement ai_algorithms.environment.models.Environment
        :param node: Node
        :return: the expanded child; Node
        """
        action = self.random_state.choice(node.pending_actions)
        child_state = environment.react(node.state, node.agent, action)
        node.pending_actions.remove(action)
        child = Node(
            child_state,
            environment.next_agent(child_state),
            environment.actions_from(child_state),
            parent=node
        )
        node.children_by_action[action] = child
        return child

    def simulation(self, environment, state, agent):
        """
        Roll out of the environment taking random actions until reaching a goal state.
        :param environment: must implement ai_algorithms.environment.models.Environment
        :param state: current state; must be a hashable object
        :param agent: starting agent; must implement ai_algorithms.environment.agent.Agent
        :return: winning agents; set of ai_algorithms.environment.agent.Agent
        """
        while not environment.is_goal_state(state):
            action = self.random_state.choice(environment.actions_from(state))
            state = environment.react(state, agent, action)
            agent = environment.next_agent(state)
        return self.best_agents(environment, state)

    def best_agents(self, environment, state):
        """
        Collects the agent with the highest score.
        Could be more than one in case of a tie.
        :param environment: must implement ai_algorithms.environment.models.Environment
        :param state: current state; must be a hashable object
        :return: set of ai_algorithms.environment.agent.Agent
        """
        winners = set()
        max_score = 0
        for agent in environment.agents(state):
            score = environment.evaluate(state, agent)
            if len(winners) == 0 or score > max_score:
                winners = {agent}
                max_score = score
            elif len(winners) > 0 or score == max_score:
                winners.add(agent)
        return winners

    def backpropagation(self, agent, node, winners):
        """
        Updates the scores and visits of the last simulated branch.
        :param agent: the agent using this strategy; must implement ai_algorithms.environment.agent.Agent
        :param node: the simulated node; Node
        :param winners: winners of the simulation; set of ai_algorithms.environment.agent.Agent
        """
        if agent in winners:
            score = 1.0 / len(winners)
        else:
            score = 0.0
        while node is not None:
            node.visits += 1
            # if the agent that took an action to lead to the current node is the agent using this strategy,
            # then add the corresponding score
            if node.parent is not None and node.parent.agent == agent:
                node.score += score
            node = node.parent

    def best_action(self, root):
        """
        Select the action with the highest calculated score.
        In case of a tie, select one at random.
        :param root: Node
        :return: selected action; must be a hashable object
        """
        actions = []
        max_score = 0
        for action, child in root.children_by_action.items():
            if len(actions) == 0 or child.score > max_score:
                actions = [action]
                max_score = child.score
            elif len(actions) > 0 or child.score == max_score:
                actions.append(action)
        return self.random_state.choice(actions)
