from django.core.management.base import BaseCommand

from catalog.utils.embeddings import embed_missing_courses


class Command(BaseCommand):
    help = 'Compute Course.embedding for every course that does not have one yet.'

    def handle(self, *args, **options):
        count = embed_missing_courses()
        self.stdout.write(f'Embedded {count} course(s).')
