#!/bin/sh
set -e

# The database lives in Postgres, which isn't reachable at image build time,
# so schema migrations and the catalog fixtures are applied on each start.
# loaddata upserts by primary key, so reloading leaves user data untouched;
# load_catalog wraps it so existing course embeddings survive the reload.
python manage.py migrate --noinput
python manage.py load_catalog catalog/fixtures/*.json

# Embed new or changed courses in the background so the server isn't held up.
python manage.py embed_courses &

exec "$@"
