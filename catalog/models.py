from django.db import models
from pgvector.django import HnswIndex, VectorField


class Subject(models.Model):
    code = models.CharField(max_length=8, primary_key=True)
    name = models.CharField(max_length=100)

    class Meta:
        ordering = ['code']

    def __str__(self):
        return f'{self.code} - {self.name}'


class Course(models.Model):
    prefix = models.ForeignKey(
        Subject, on_delete=models.PROTECT, related_name='courses', db_column='prefix'
    )
    # Not purely numeric: e.g. "499A", "196L".
    number = models.CharField(max_length=8)
    title = models.CharField(max_length=200)
    # Null when the course has a variable credit range (see min/max_credits).
    credits = models.PositiveSmallIntegerField(null=True, blank=True)
    min_credits = models.PositiveSmallIntegerField()
    max_credits = models.PositiveSmallIntegerField()
    description = models.TextField(null=True, blank=True)
    prerequisites = models.TextField(null=True, blank=True)
    corequisites = models.TextField(null=True, blank=True)
    # For semantic search; null until the course has been embedded.
    embedding = VectorField(dimensions=384, null=True, blank=True)

    class Meta:
        ordering = ['prefix', 'number']
        constraints = [
            models.UniqueConstraint(fields=['prefix', 'number'], name='unique_course_code'),
        ]
        indexes = [
            HnswIndex(
                name='course_embedding_hnsw',
                fields=['embedding'],
                m=16,
                ef_construction=64,
                opclasses=['vector_cosine_ops'],
            ),
        ]

    @property
    def code(self):
        return f'{self.prefix_id} {self.number}'

    def embedding_text(self):
        """The text that Course.embedding is computed from."""
        if self.description:
            return f'{self}\n{self.description}'
        return str(self)

    def __str__(self):
        return f'{self.code}: {self.title}'


class Program(models.Model):
    class Level(models.TextChoices):
        UNDERGRADUATE = 'undergraduate', 'Undergraduate'
        GRADUATE = 'graduate', 'Graduate'

    slug = models.SlugField(max_length=128, primary_key=True)
    name = models.CharField(max_length=200)
    level = models.CharField(max_length=16, choices=Level.choices)
    # Degree/credential type, e.g. "B.S.", "minor", "certificate".
    type = models.CharField(max_length=64)
    url = models.URLField(max_length=300)
    requirements_pdf_url = models.URLField(max_length=500)
    description = models.TextField(null=True, blank=True)  # HTML
    keywords = models.JSONField(default=list, blank=True)

    class Meta:
        ordering = ['name']

    def __str__(self):
        return self.name


class Track(models.Model):
    program = models.ForeignKey(Program, on_delete=models.CASCADE, related_name='tracks')
    name = models.CharField(max_length=200)

    class Meta:
        ordering = ['program', 'name']
        constraints = [
            models.UniqueConstraint(fields=['program', 'name'], name='unique_track_per_program'),
        ]

    def __str__(self):
        return f'{self.program.name} - {self.name}'


class ProgramRequirement(models.Model):
    program = models.ForeignKey(
        Program, on_delete=models.CASCADE, related_name='requirements'
    )
    title = models.CharField(max_length=200)
    subsection = models.CharField(max_length=200, null=True, blank=True)
    # Set when the requirement only applies to one track/concentration.
    track = models.ForeignKey(
        Track, on_delete=models.CASCADE, null=True, blank=True, related_name='requirements'
    )
    cluster = models.CharField(max_length=128, null=True, blank=True)
    credits_min = models.PositiveSmallIntegerField(null=True, blank=True)
    credits_max = models.PositiveSmallIntegerField(null=True, blank=True)

    class Meta:
        ordering = ['program', 'pk']

    def __str__(self):
        return f'{self.program_id}: {self.title}'


class RequirementGroup(models.Model):
    """A node in a requirement's rule tree. Each ProgramRequirement has one root group."""

    class Kind(models.TextChoices):
        ALL = 'all', 'All of'
        CHOOSE = 'choose', 'Choose'
        PLACEHOLDER = 'placeholder', 'Placeholder'
        NOTE = 'note', 'Note'

    class EachUnit(models.TextChoices):
        COUNT = 'count', 'Courses'
        CREDITS = 'credits', 'Credits'

    program_requirement = models.ForeignKey(
        ProgramRequirement, on_delete=models.CASCADE, related_name='groups'
    )
    parent_group = models.ForeignKey(
        'self', on_delete=models.CASCADE, null=True, blank=True, related_name='children'
    )
    position = models.PositiveIntegerField(default=0)
    kind = models.CharField(max_length=16, choices=Kind.choices)
    min_count = models.PositiveSmallIntegerField(null=True, blank=True)
    max_count = models.PositiveSmallIntegerField(null=True, blank=True)
    min_credits = models.PositiveSmallIntegerField(null=True, blank=True)
    max_credits = models.PositiveSmallIntegerField(null=True, blank=True)
    # Per-child limits (e.g. "at least 3 credits from each area"), measured in each_unit.
    each_min = models.PositiveSmallIntegerField(null=True, blank=True)
    each_max = models.PositiveSmallIntegerField(null=True, blank=True)
    each_unit = models.CharField(
        max_length=8, choices=EachUnit.choices, null=True, blank=True
    )
    condition = models.TextField(null=True, blank=True)
    shareable = models.BooleanField(default=False)
    text = models.TextField(null=True, blank=True)
    needs_review = models.BooleanField(default=False)

    class Meta:
        ordering = ['program_requirement', 'parent_group_id', 'position']

    def __str__(self):
        return f'{self.get_kind_display()} group #{self.pk} ({self.program_requirement})'


class CourseRequirement(models.Model):
    """A specific course listed within a requirement group."""

    group = models.ForeignKey(
        RequirementGroup, on_delete=models.CASCADE, related_name='course_requirements'
    )
    course = models.ForeignKey(
        Course, on_delete=models.CASCADE, related_name='requirement_entries'
    )
    # Course code as written in the catalog, e.g. "COB 191".
    course_code = models.CharField(max_length=16)
    position = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ['group', 'position']

    def __str__(self):
        return f'{self.course_code} in group #{self.group_id}'


class CourseFilter(models.Model):
    """A rule matching a set of courses for a 'choose' group, e.g. "ACTG 500-699".

    Any of subject / number range / course may be set; exclude=True removes matches.
    """

    group = models.ForeignKey(
        RequirementGroup, on_delete=models.CASCADE, related_name='course_filters'
    )
    subject = models.ForeignKey(
        Subject,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name='course_filters',
    )
    number_min = models.PositiveSmallIntegerField(null=True, blank=True)
    number_max = models.PositiveSmallIntegerField(null=True, blank=True)
    course = models.ForeignKey(
        Course,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name='course_filters',
    )
    course_code = models.CharField(max_length=16, null=True, blank=True)
    exclude = models.BooleanField(default=False)

    def __str__(self):
        if self.course_code:
            target = self.course_code
        else:
            rng = ''
            if self.number_min is not None or self.number_max is not None:
                rng = f" {self.number_min or ''}-{self.number_max or ''}"
            target = f"{self.subject_id or 'any'}{rng}"
        return f"{'exclude ' if self.exclude else ''}{target} (group #{self.group_id})"
