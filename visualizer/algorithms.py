import heapq
import math

def bfs(graph, start, goal):
    visited = set()
    queue = [[start]]
    visited_order = []

    if start == goal:
        return [start], [start]

    while queue:
        path = queue.pop(0)
        node = path[-1]

        if node not in visited:
            visited_order.append(node)
            neighbors = graph.get(node, [])
            for neighbor in neighbors:
                new_path = list(path)
                new_path.append(neighbor)
                queue.append(new_path)

                if neighbor == goal:
                    visited_order.extend(new_path[len(path):])
                    return visited_order, new_path
            visited.add(node)
    return visited_order, []

def dfs(graph, start, goal):
    visited = set()
    stack = [[start]]
    visited_order = []

    if start == goal:
        return [start], [start]

    while stack:
        path = stack.pop()
        node = path[-1]

        if node not in visited:
            visited_order.append(node)
            if node == goal:
                return visited_order, path
            visited.add(node)
            neighbors = graph.get(node, [])
            for neighbor in reversed(neighbors):
                if neighbor not in visited:
                    new_path = list(path)
                    new_path.append(neighbor)
                    stack.append(new_path)
    return visited_order, []


def astar(graph, start, goal, positions, rules=None):
    def heuristic(node1, node2):
        pos1 = positions.get(node1)
        pos2 = positions.get(node2)
        if not pos1 or not pos2: return 0
        return math.sqrt((pos1['x'] - pos2['x']) ** 2 + (pos1['y'] - pos2['y']) ** 2)

    avoid_nodes = set()
    if rules:
        for rule in rules.replace(" ", "").split(','):
            if rule.upper().startswith("AVOID("):
                node_to_avoid = rule[6:-1]
                avoid_nodes.add(node_to_avoid)

    pq = [(0, [start])]
    visited = set()
    visited_order = []
    g_costs = {start: 0}

    while pq:
        cost, path = heapq.heappop(pq)
        node = path[-1]

        if node in visited:
            continue

        visited.add(node)
        visited_order.append(node)

        if node == goal:
            return visited_order, path

        for neighbor in graph.get(node, []):
            if neighbor in visited or neighbor in avoid_nodes:
                continue

            new_g_cost = g_costs[node] + heuristic(node, neighbor)

            if neighbor not in g_costs or new_g_cost < g_costs[neighbor]:
                g_costs[neighbor] = new_g_cost
                f_cost = new_g_cost + heuristic(neighbor, goal)
                new_path = path + [neighbor]
                heapq.heappush(pq, (f_cost, new_path))

    return visited_order, []

def dijkstra(graph, start, goal, positions):
    def get_weight(node1, node2):
        pos1 = positions.get(node1)
        pos2 = positions.get(node2)
        if not pos1 or not pos2: return 1  # Default weight if no position
        return math.sqrt((pos1['x'] - pos2['x']) ** 2 + (pos1['y'] - pos2['y']) ** 2)

    pq = [(0, [start])]
    visited = set()
    visited_order = []
    min_distances = {node: float('inf') for node in graph}
    min_distances[start] = 0

    while pq:
        dist, path = heapq.heappop(pq)
        node = path[-1]

        if node in visited:
            continue

        visited.add(node)
        visited_order.append(node)

        if node == goal:
            return visited_order, path

        if dist > min_distances[node]:
            continue

        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                weight = get_weight(node, neighbor)
                new_dist = dist + weight

                if new_dist < min_distances[neighbor]:
                    min_distances[neighbor] = new_dist
                    new_path = path + [neighbor]
                    heapq.heappush(pq, (new_dist, new_path))

    return visited_order, []


def greedy_bfs(graph, start, goal, positions):
    def heuristic(node, goal_node):
        pos1 = positions.get(node)
        pos2 = positions.get(goal_node)
        if not pos1 or not pos2: return 0
        return math.sqrt((pos1['x'] - pos2['x']) ** 2 + (pos1['y'] - pos2['y']) ** 2)

    pq = [(heuristic(start, goal), [start])]
    visited = set()
    visited_order = []

    while pq:
        _, path = heapq.heappop(pq)
        node = path[-1]

        if node in visited:
            continue

        visited.add(node)
        visited_order.append(node)

        if node == goal:
            return visited_order, path

        for neighbor in sorted(graph.get(node, [])):
            if neighbor not in visited:
                new_path = path + [neighbor]
                heapq.heappush(pq, (heuristic(neighbor, goal), new_path))

    return visited_order, []

def dls(graph, start, goal, depth_limit):
    visited = set()
    stack = [(start, [start], 0)]  # (node, path, depth)
    visited_order = []

    while stack:
        node, path, depth = stack.pop()

        if node not in visited:
            visited.add(node)
            visited_order.append(node)

            if node == goal:
                return visited_order, path

            if depth < depth_limit:
                for neighbor in reversed(graph.get(node, [])):
                    if neighbor not in visited:
                        stack.append((neighbor, path + [neighbor], depth + 1))

    return visited_order, []


def iddfs(graph, start, goal):
    visited_order_total = []
    for depth_limit in range(len(graph) + 1):
        visited_this_run = set()
        stack = [(start, [start], 0)]

        while stack:
            node, path, depth = stack.pop()

            if node not in visited_this_run:
                visited_this_run.add(node)
                if node not in visited_order_total:
                    visited_order_total.append(node)

                if node == goal:
                    return visited_order_total, path

                if depth < depth_limit:
                    for neighbor in reversed(graph.get(node, [])):
                        if neighbor not in visited_this_run:
                            stack.append((neighbor, path + [neighbor], depth + 1))

    return visited_order_total, []

def hill_climbing(graph, start, node_values):
    path = [start]
    current_node = start

    while True:
        neighbors = graph.get(current_node, [])
        if not neighbors:
            break

        best_neighbor = None
        current_value = node_values[current_node]['value']
        for neighbor in neighbors:
            if node_values[neighbor]['value'] > current_value:
                if best_neighbor is None or node_values[neighbor]['value'] > node_values[best_neighbor]['value']:
                    best_neighbor = neighbor

        if best_neighbor is None:
            break
        else:
            path.append(best_neighbor)
            current_node = best_neighbor
    return path, [path[-1]]