from collections import deque


def in_path_node(node, path_node):
    while path_node is not None:
        parent, path_node = path_node
        if parent == node:
            return True
    return False


def paths_between(node_a, node_b):
    path_nodes = []
    queue = deque()
    queue.append((node_a, None))
    while len(queue) > 0:
        path_node = queue.popleft()
        node, parent_path_node = path_node
        for neighbor in node.neighbors():
            if neighbor == node_b:
                path_nodes.append((neighbor, path_node))
            else:
                if not in_path_node(neighbor, parent_path_node):
                    queue.append((neighbor, path_node))
    paths = []
    for path_node in path_nodes:
        path = deque()
        while path_node is not None:
            node, path_node = path_node
            path.appendleft(node)
        paths.append(path)
    return paths


def arcs_to_edges(adjacency_list):
    new_adjacency_list = {}
    for node_from, neighbors in adjacency_list.items():
        for node_to in neighbors:
            if node_from not in new_adjacency_list:
                new_adjacency_list[node_from] = set()
            if node_to not in new_adjacency_list:
                new_adjacency_list[node_to] = set()
            new_adjacency_list[node_from].add(node_to)
            new_adjacency_list[node_to].add(node_from)
    return new_adjacency_list


def d_separated_triple(a, b, c, observed_nodes):
    # causal chain: a -> b -> c and b is given => c is independent from a given b
    if a.is_parent_of(b) and b.is_parent_of(c) and b in observed_nodes:
        return True
    # causal chain: a <- b <- c and b is given => a is independent from c given b
    if a.is_child_of(b) and b.is_child_of(c) and b in observed_nodes:
        return True
    # common cause: a <- b -> c and b is given => c is independent from a given b
    if a.is_child_of(b) and b.is_parent_of(c) and b in observed_nodes:
        return True
    # common effect: a -> b <- c
    if a.is_parent_of(b) and b.is_child_of(c):
        # check if b or a node reachable from b is given
        queue = deque()
        queue.append(b)
        visited = set()
        while len(queue) > 0:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                if node in observed_nodes:
                    return False
                for child in node.children:
                    queue.append(child)
        # b is not given and no node reachable from b is given => c is independent from a
        return True
    # no case was found so it is not d-separable
    return False


def d_separated_path(path, observed_nodes):
    i = 0
    while i < len(path) - 2:
        if not d_separated_triple(path[i], path[i+1], path[i+2], observed_nodes):
            # if at least one triple is not d-separable then the path is not d-separable
            return False
        i += 1
    return True


def d_separated(node_a, node_b, observed_nodes):
    for path in paths_between(node_a, node_b):
        if not d_separated_path(path, observed_nodes):
            # if at least one path is not d-separable then the node pair is not d-separable
            return False
    return True
