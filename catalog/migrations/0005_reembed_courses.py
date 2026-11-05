from django.db import migrations


def clear_embeddings(apps, schema_editor):
    # Embeddings now come from fastembed's ONNX build of the model, whose vectors
    # don't exactly match the sentence-transformers ones; embed_courses (run on
    # each start) recomputes them all.
    apps.get_model('catalog', 'Course').objects.update(embedding=None)


class Migration(migrations.Migration):

    dependencies = [
        ('catalog', '0004_alter_course_embedding'),
    ]

    operations = [
        migrations.RunPython(clear_embeddings, migrations.RunPython.noop),
    ]
