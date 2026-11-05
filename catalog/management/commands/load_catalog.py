from django.core.management import call_command
from django.core.management.base import BaseCommand
from django.db import transaction

from catalog.models import Course


class Command(BaseCommand):
    help = (
        'Run loaddata on the given fixtures, keeping existing course embeddings. '
        'loaddata saves every field, so courses would otherwise lose their embedding; '
        'it is kept for each course whose embedding text is unchanged.'
    )

    def add_arguments(self, parser):
        parser.add_argument('fixtures', nargs='+')

    def handle(self, *args, fixtures, **options):
        with transaction.atomic():
            existing = {
                course.pk: (course.embedding_text(), course.embedding)
                for course in Course.objects.filter(embedding__isnull=False)
            }

            call_command('loaddata', *fixtures, verbosity=options['verbosity'])

            restored = []
            for course in Course.objects.filter(pk__in=existing, embedding__isnull=True):
                text, embedding = existing[course.pk]
                if course.embedding_text() == text:
                    course.embedding = embedding
                    restored.append(course)
            Course.objects.bulk_update(restored, ['embedding'], batch_size=500)

        stale = len(existing) - len(restored)
        self.stdout.write(f'Kept {len(restored)} course embedding(s); {stale} need re-embedding.')
