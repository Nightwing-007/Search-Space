# visualizer/models.py
from django.db import models
from django.contrib.auth.models import User

class Graph(models.Model):
    name = models.CharField(max_length=100)
    graph_data = models.JSONField()
    owner = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"'{self.name}' by {self.owner.username}"