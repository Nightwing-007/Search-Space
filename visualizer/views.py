# visualizer/views.py

from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
import json
import numpy as np
from sklearn.cluster import KMeans

from . import algorithms
from .models import Graph


# --- Main App View ---

@login_required
def index(request):
    """
    Renders the main application page.
    This view is protected and requires a user to be logged in.
    """
    return render(request, 'visualizer/index.html')


# --- API Views ---

@login_required
def run_search(request):
    """
    Handles all algorithm execution requests from the frontend.
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            graph_data = data.get('graph_data')
            algorithm = data.get('algorithm')
            start_node = data.get('start_node')
            goal_node = data.get('goal_node')
            rules = data.get('rules')
            depth_limit = data.get('depth_limit')

            # Smart validation:
            # 1. Check for data required by ALL algorithms
            if not all([graph_data, algorithm, start_node]):
                return JsonResponse({'status': 'error', 'message': 'Missing graph, algorithm, or start node.'},
                                    status=400)
            # 2. Check for goal_node only if the algorithm needs it
            if algorithm != 'hill_climbing' and not goal_node:
                return JsonResponse({'status': 'error', 'message': 'Goal node is required for this algorithm.'},
                                    status=400)

            # Convert edge list to an adjacency list for the algorithms
            adjacency_list = {node: [] for node in graph_data['nodes']}
            for edge in graph_data['edges']:
                u, v = edge
                if u in adjacency_list and v in adjacency_list:
                    adjacency_list.setdefault(u, []).append(v)
                    adjacency_list.setdefault(v, []).append(u)

            visited_order = []
            path_to_goal = []
            positions = graph_data['nodes']
            node_values = graph_data['nodes']  # For hill climbing

            # Algorithm router
            if algorithm == 'bfs':
                visited_order, path_to_goal = algorithms.bfs(adjacency_list, start_node, goal_node)
            elif algorithm == 'dfs':
                visited_order, path_to_goal = algorithms.dfs(adjacency_list, start_node, goal_node)
            elif algorithm == 'dls':
                limit = int(depth_limit) if depth_limit else len(adjacency_list)
                visited_order, path_to_goal = algorithms.dls(adjacency_list, start_node, goal_node, limit)
            elif algorithm == 'iddfs':
                visited_order, path_to_goal = algorithms.iddfs(adjacency_list, start_node, goal_node)
            elif algorithm == 'dijkstra':
                visited_order, path_to_goal = algorithms.dijkstra(adjacency_list, start_node, goal_node, positions)
            elif algorithm == 'greedy':
                visited_order, path_to_goal = algorithms.greedy_bfs(adjacency_list, start_node, goal_node, positions)
            elif algorithm == 'astar':
                visited_order, path_to_goal = algorithms.astar(adjacency_list, start_node, goal_node, positions, rules)
            elif algorithm == 'hill_climbing':
                visited_order, path_to_goal = algorithms.hill_climbing(adjacency_list, start_node, node_values)

            return JsonResponse({'status': 'success', 'visited_order': visited_order, 'path_to_goal': path_to_goal})

        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)


@login_required
def find_clusters(request):
    """
    Handles K-Means clustering requests.
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            graph_data = data.get('graph_data')
            k = int(data.get('k', 3))

            if not graph_data:
                return JsonResponse({'status': 'error', 'message': 'Graph data is required.'}, status=400)

            nodes = graph_data.get('nodes', {})
            node_ids = sorted(nodes.keys())  # Ensure consistent node order
            coordinates = np.array([[nodes[nid]['x'], nodes[nid]['y']] for nid in node_ids])

            if len(coordinates) < k:
                return JsonResponse({'status': 'error', 'message': 'More clusters than nodes.'}, status=400)

            kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto').fit(coordinates)
            labels = kmeans.labels_

            # Add cluster_id back to each node
            for i, node_id in enumerate(node_ids):
                graph_data['nodes'][node_id]['cluster_id'] = int(labels[i])

            return JsonResponse({'status': 'success', 'graph_data': graph_data})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)


# --- Graph Save/Load API Views ---

@login_required
def save_graph(request):
    """
    Saves a graph to the database, linked to the logged-in user.
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            graph_name = data.get('name')
            graph_data = data.get('graph_data')

            if not graph_name or not graph_data:
                return JsonResponse({'status': 'error', 'message': 'Graph name and data are required.'}, status=400)

            Graph.objects.create(
                name=graph_name,
                graph_data=graph_data,
                owner=request.user
            )
            return JsonResponse({'status': 'success', 'message': f"Graph '{graph_name}' saved."})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)


@login_required
def get_graphs(request):
    """
    Retrieves all graphs owned by the logged-in user.
    """
    graphs = Graph.objects.filter(owner=request.user).order_by('-created_at')
    graph_list = [{'id': graph.id, 'name': graph.name, 'graph_data': graph.graph_data} for graph in graphs]
    return JsonResponse({'graphs': graph_list})


# --- User Authentication Views ---

def register_view(request):
    """
    Handles user registration.
    """
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('index')
    else:
        form = UserCreationForm()
    return render(request, 'visualizer/register.html', {'form': form})


def logout_view(request):
    """
    Logs the user out and redirects to the login page.
    """
    logout(request)
    return redirect('login')