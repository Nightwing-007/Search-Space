# visualizer/views.py
from django.shortcuts import render
from django.http import JsonResponse
import json
from . import algorithms

def index(request):
    return render(request, 'visualizer/index.html')

def run_search(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            graph_data = data.get('graph_data')
            algorithm = data.get('algorithm')
            start_node = data.get('start_node')
            goal_node = data.get('goal_node')
            rules = data.get('rules')

            if not all([graph_data, algorithm, start_node, goal_node]):
                return JsonResponse({'status': 'error', 'message': 'Missing required data.'}, status=400)

            # Convert edge list to an adjacency list for the algorithms
            adjacency_list = {node: [] for node in graph_data['nodes']}
            for edge in graph_data['edges']:
                u, v = edge
                if u in adjacency_list and v in adjacency_list:
                    adjacency_list[u].append(v)
                    adjacency_list[v].append(u)

            visited_order = []
            path_to_goal = []

            if algorithm == 'bfs':
                visited_order, path_to_goal = algorithms.bfs(adjacency_list, start_node, goal_node)
            elif algorithm == 'dfs':
                visited_order, path_to_goal = algorithms.dfs(adjacency_list, start_node, goal_node)
            elif algorithm == 'astar':
                positions = graph_data['nodes']
                visited_order, path_to_goal = algorithms.astar(adjacency_list, start_node, goal_node, positions, rules)

            return JsonResponse({
                'status': 'success',
                'visited_order': visited_order,
                'path_to_goal': path_to_goal
            })
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)


# visualizer/views.py
# ... (imports and index view are unchanged) ...

def run_search(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            # ... (data extraction code is unchanged) ...
            graph_data = data.get('graph_data')
            algorithm = data.get('algorithm')
            start_node = data.get('start_node')
            goal_node = data.get('goal_node')
            rules = data.get('rules')

            if not all([graph_data, algorithm, start_node, goal_node]):
                return JsonResponse({'status': 'error', 'message': 'Missing required data.'}, status=400)

            # ... (adjacency list creation is unchanged) ...
            adjacency_list = {node: [] for node in graph_data['nodes']}
            for edge in graph_data['edges']:
                u, v = edge
                if u in adjacency_list and v in adjacency_list:
                    adjacency_list[u].append(v)
                    adjacency_list[v].append(u)

            visited_order = []
            path_to_goal = []
            positions = graph_data['nodes']  # Needed by more algorithms now

            if algorithm == 'bfs':
                visited_order, path_to_goal = algorithms.bfs(adjacency_list, start_node, goal_node)
            elif algorithm == 'dfs':
                visited_order, path_to_goal = algorithms.dfs(adjacency_list, start_node, goal_node)
            elif algorithm == 'astar':
                visited_order, path_to_goal = algorithms.astar(adjacency_list, start_node, goal_node, positions, rules)

            # --- START: ADD THESE NEW CONDITIONS ---
            elif algorithm == 'dijkstra':
                visited_order, path_to_goal = algorithms.dijkstra(adjacency_list, start_node, goal_node, positions)
            elif algorithm == 'greedy':
                visited_order, path_to_goal = algorithms.greedy_bfs(adjacency_list, start_node, goal_node, positions)
            # --- END: ADD THESE NEW CONDITIONS ---

            return JsonResponse({
                'status': 'success',
                'visited_order': visited_order,
                'path_to_goal': path_to_goal
            })
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)