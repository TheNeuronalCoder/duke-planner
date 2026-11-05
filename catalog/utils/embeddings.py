from functools import lru_cache

from django.db import connection, transaction
from pgvector.django import CosineDistance

from catalog.models import Course

MODEL_NAME = 'BAAI/bge-small-en-v1.5'
BATCH_SIZE = 64
# bge models expect this prefix on short queries, but not on the passages they're matched against.
QUERY_INSTRUCTION = 'Represent this sentence for searching relevant passages: '


@lru_cache(maxsize=1)
def _model():
    # Imported lazily: loading torch is slow and only needed when there's work to do.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


def embed_missing_courses():
    """Fill Course.embedding for every course that doesn't have one yet."""
    courses = list(Course.objects.filter(embedding__isnull=True))
    if not courses:
        return 0

    model = _model()
    for start in range(0, len(courses), BATCH_SIZE):
        batch = courses[start:start + BATCH_SIZE]
        # Normalized so cosine distance (the HNSW index opclass) behaves as expected.
        vectors = model.encode([c.embedding_text() for c in batch], normalize_embeddings=True)
        for course, vector in zip(batch, vectors):
            course.embedding = vector.tolist()
        Course.objects.bulk_update(batch, ['embedding'])
    return len(courses)


def embed_query(text):
    return _model().encode(QUERY_INSTRUCTION + text, normalize_embeddings=True).tolist()


def semantic_search(query, courses, limit):
    """The `limit` courses of the `courses` queryset closest in meaning to `query`."""
    courses = courses.filter(embedding__isnull=False).order_by(
        CosineDistance('embedding', embed_query(query))
    )
    with transaction.atomic():
        # Without this, an HNSW scan filtered by other conditions (subject, Gen Ed)
        # stops after ef_search candidates and can return too few rows (pgvector 0.8+).
        with connection.cursor() as cursor:
            cursor.execute("SET LOCAL hnsw.iterative_scan = 'strict_order'")
        return list(courses[:limit])
