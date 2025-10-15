# visualizer/views.py

from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
import json
import numpy as np
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from . import algorithms
from .models import Graph


# --- Main App View ---
@login_required
def index(request):
    return render(request, 'visualizer/index.html')


# --- API Views ---
@login_required
def run_search(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            graph_data, algorithm, start_node = data.get('graph_data'), data.get('algorithm'), data.get('start_node')
            goal_node, rules, depth_limit = data.get('goal_node'), data.get('rules'), data.get('depth_limit')

            if not all([graph_data, algorithm, start_node]):
                return JsonResponse({'status': 'error', 'message': 'Missing graph, algorithm, or start node.'},
                                    status=400)
            if algorithm not in ['hill_climbing'] and not goal_node:
                return JsonResponse({'status': 'error', 'message': 'Goal node is required for this algorithm.'},
                                    status=400)

            adjacency_list = {node: [] for node in graph_data['nodes']}
            for edge in graph_data['edges']:
                u, v = edge
                if u in adjacency_list and v in adjacency_list:
                    adjacency_list.setdefault(u, []).append(v)
                    adjacency_list.setdefault(v, []).append(u)

            visited_order, path_to_goal = [], []
            positions, node_values = graph_data['nodes'], graph_data['nodes']

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
            elif algorithm == 'hill_climbing':
                visited_order, path_to_goal = algorithms.hill_climbing(adjacency_list, start_node, node_values)
            elif algorithm == 'astar':
                model = None
                if data.get('heuristic_type') == 'nn' and 'heuristic_model_weights' in request.session:
                    if request.session.get('goal_node_for_model') != goal_node:
                        return JsonResponse({'status': 'error',
                                             'message': f"Model trained for goal '{request.session.get('goal_node_for_model')}'. Retrain for goal '{goal_node}'."},
                                            status=400)

                    model = keras.Sequential(
                        [layers.Dense(32, activation='relu', input_shape=[2]), layers.Dense(32, activation='relu'),
                         layers.Dense(1)])
                    # This dummy build step is essential before loading weights
                    model.build(input_shape=(None, 2))

                    weights_as_lists = request.session['heuristic_model_weights']
                    weights_as_np = [np.array(w) for w in weights_as_lists]
                    model.set_weights(weights_as_np)

                visited_order, path_to_goal = algorithms.astar(adjacency_list, start_node, goal_node, positions, rules,
                                                               model=model)

            return JsonResponse({'status': 'success', 'visited_order': visited_order, 'path_to_goal': path_to_goal})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)


@login_required
def train_heuristic_model(request):
    if request.method == 'POST':
        try:
            print("INFO: Starting heuristic model training...")
            data = json.loads(request.body)
            graph_data, goal_node = data.get('graph_data'), data.get('goal_node')
            if not graph_data or not goal_node:
                return JsonResponse({'status': 'error', 'message': 'Graph and goal node required.'}, status=400)

            adjacency_list = {node: [] for node in graph_data['nodes']}
            for edge in graph_data['edges']:
                u, v = edge
                if u in adjacency_list and v in adjacency_list:
                    adjacency_list.setdefault(u, []).append(v)
                    adjacency_list.setdefault(v, []).append(u)

            print(f"INFO: Generating training data for goal node '{goal_node}'...")
            _, true_costs = algorithms.dijkstra(adjacency_list, goal_node, None, graph_data['nodes'],
                                                return_all_costs=True)
            X_train, y_train = [], []
            for node_id, cost in true_costs.items():
                if cost != float('inf'):
                    X_train.append([graph_data['nodes'][node_id]['x'], graph_data['nodes'][node_id]['y']])
                    y_train.append(cost)

            if len(X_train) < 2: return JsonResponse({'status': 'error', 'message': 'Not enough data to train.'},
                                                     status=400)

            print("INFO: Building and training Keras model...")
            model = keras.Sequential(
                [layers.Dense(32, activation='relu', input_shape=[2]), layers.Dense(32, activation='relu'),
                 layers.Dense(1)])
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(np.array(X_train), np.array(y_train), epochs=100, batch_size=8, verbose=0)
            print("INFO: Model training complete.")

            print("INFO: Saving model weights to session...")
            weights = [w.tolist() for w in model.get_weights()]
            request.session['heuristic_model_weights'] = weights
            request.session['goal_node_for_model'] = goal_node
            print("INFO: Weights saved successfully.")

            return JsonResponse({'status': 'success', 'message': f"Model trained for goal '{goal_node}'."})
        except Exception as e:
            print(f"ERROR in train_heuristic_model: {e}")  # This will print the error to your terminal
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    return JsonResponse({'status': 'error', 'message': 'Invalid request'}, status=400)


# --- (All other views: find_clusters, save_graph, get_graphs, register_view, logout_view remain unchanged) ---
@login_required
def find_clusters(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            graph_data, k = data.get('graph_data'), int(data.get('k', 3))
            if not graph_data: return JsonResponse({'status': 'error', 'message': 'Graph data is required.'},
                                                   status=400)
            nodes, node_ids = graph_data.get('nodes', {}), sorted(graph_data['nodes'].keys())
            coordinates = np.array([[nodes[nid]['x'], nodes[nid]['y']] for nid in node_ids])
            if len(coordinates) < k: return JsonResponse({'status': 'error', 'message': 'More clusters than nodes.'},
                                                         status=400)
            kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto').fit(coordinates)
            for i, node_id in enumerate(node_ids):
                graph_data['nodes'][node_id]['cluster_id'] = int(kmeans.labels_[i])
            return JsonResponse({'status': 'success', 'graph_data': graph_data})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)


@login_required
def save_graph(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            graph_name, graph_data = data.get('name'), data.get('graph_data')
            if not graph_name or not graph_data: return JsonResponse(
                {'status': 'error', 'message': 'Name and data required.'}, status=400)
            Graph.objects.create(name=graph_name, graph_data=graph_data, owner=request.user)
            return JsonResponse({'status': 'success', 'message': f"Graph '{graph_name}' saved."})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)


@login_required
def get_graphs(request):
    graphs = Graph.objects.filter(owner=request.user).order_by('-created_at')
    graph_list = [{'id': graph.id, 'name': graph.name, 'graph_data': graph.graph_data} for graph in graphs]
    return JsonResponse({'graphs': graph_list})


def register_view(request):
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
    logout(request)
    return redirect('login')