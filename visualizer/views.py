# visualizer/views.py
from django.shortcuts import render

def index(request):
    return render(request, 'visualizer/index.html')