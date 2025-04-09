from django.urls import path
from main.views import api

urlpatterns = [
    path("api/", api.urls),  # This will mount your API at /api/
]
