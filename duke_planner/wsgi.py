"""
WSGI config for duke_planner project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application
from django.db import connections

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'duke_planner.settings')

application = get_wsgi_application()

# Embed any new courses before serving requests, so semantic search covers them.
from catalog.utils.embeddings import embed_missing_courses  # noqa: E402

embed_missing_courses()
# Gunicorn forks workers after loading this module (--preload); they mustn't share its connection.
connections.close_all()
