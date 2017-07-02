from collections import deque


def in_path_node(node, path_node):
    while path_node is not None:
        parent, path_node = path_node
        if parent == node:
            return True
    return False


def paths_between(node_a, node_b, adjacency_list):
    path_nodes = []
    queue = deque()
    queue.append((node_a, None))
    while len(queue) > 0:
        path_node = queue.popleft()
        node, parent_path_node = path_node
        for neighbor in adjacency_list[node]:
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


def d_separated_triple(a, b, c, adjacency_list, conditions):
    # causal chain a -> b -> c, c _||_ a | b
    if b in adjacency_list[a] and c in adjacency_list[b] and b in conditions:
        return True
    # causal chain a <- b <- c, a _||_ c | b
    if b in adjacency_list[c] and a in adjacency_list[b] and b in conditions:
        return True
    # common cause a <- b -> c, c _||_ a | b
    if a in adjacency_list[b] and c in adjacency_list[b] and b in conditions:
        return True
    # common effect a -> b <- c, c _||_ a
    if b in adjacency_list[a] and b in adjacency_list[c]:
        # check if b or a node reachable from b is conditioned
        queue = deque()
        queue.append(b)
        visited = set()
        while len(queue) > 0:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                if node in conditions:
                    return False
                for neighbor in adjacency_list[node]:
                    queue.append(neighbor)
        # if no node was found then it is d-separable
        return True
    # no case was found so it is not d-separable
    return False


def d_separated_path(path, adjacency_list, conditions):
    i = 0
    while i < len(path) - 2:
        if not d_separated_triple(path[i], path[i+1], path[i+2], adjacency_list, conditions):
            # if at least one triple is not d-separable then the path is not d-separable
            return False
        i += 1
    return True


def d_separated(node_a, node_b, adjacency_list, conditions):
    undirected_adjacency_list = arcs_to_edges(adjacency_list)
    for path in paths_between(node_a, node_b, undirected_adjacency_list):
        if not d_separated_path(path, adjacency_list, conditions):
            # if at least one path is not d-separable then the node pair is not d-separable
            return False
    return True
