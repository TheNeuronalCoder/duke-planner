from catalog.models import Course

MODEL_NAME = 'BAAI/bge-small-en-v1.5'
BATCH_SIZE = 64


def embed_missing_courses():
    """Fill Course.embedding for every course that doesn't have one yet."""
    courses = list(Course.objects.filter(embedding__isnull=True))
    if not courses:
        return 0

    # Imported lazily: loading torch is slow and only needed when there's work to do.
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME)
    for start in range(0, len(courses), BATCH_SIZE):
        batch = courses[start:start + BATCH_SIZE]
        # Normalized so cosine distance (the HNSW index opclass) behaves as expected.
        vectors = model.encode([c.embedding_text() for c in batch], normalize_embeddings=True)
        for course, vector in zip(batch, vectors):
            course.embedding = vector.tolist()
        Course.objects.bulk_update(batch, ['embedding'])
    return len(courses)
