"""The signed-in account's plan as plain data, shared by the student views and
the assistant."""

from collections import defaultdict

from django.db.models import F, Q, Sum
from django.db.models.functions import Coalesce

from account.validation import errors, validate_account
from catalog.utils import gened as gen_ed
from catalog.utils.courses import course_json
from catalog.utils.requirements import bucket, evaluate_program, requirements_of
from catalog.utils.status import IN_PROGRESS, PLANNED, SATISFIED, combine
from .models import Semester


def _semester_key(semester):
    """(year, Semester.Term) for a TERM_YEAR string like "FALL_2026"."""
    term, _, year = semester.partition('_')
    return int(year), Semester.Term[term.upper()]


def _status(year, term, current):
    key = (year, term)
    if key < current:
        return 'completed'
    if key == current:
        return 'in progress'
    return 'planned'


def semesters_data(account, current_semester):
    """The `classes` of GET /student/requirements/ (see student.views.requirements).

    `current_semester` is the semester in progress as TERM_YEAR (e.g.
    "FALL_2026"), which sets each semester's `status`.
    """
    current = _semester_key(current_semester)
    classes = account.classes.select_related('course').order_by('course__prefix_id', 'course__number')

    by_semester = defaultdict(list)
    for c in classes:
        course = c.course
        by_semester[c.semester_id].append({
            'id': c.pk,
            **course_json(course, details=True),
            'credits': c.credits,
            'official_credits': (
                course.credits if course.credits is not None
                else [course.min_credits, course.max_credits]
            ),
            'credits_earned': c.credits_earned,
        })

    semesters = account.semesters.annotate(
        attempted=Coalesce(Sum('classes__credits'), 0),
        earned=Coalesce(Sum('classes__credits', filter=Q(classes__credits_earned=True)), 0),
    ).order_by('year', 'term')

    return {
        'semesters': [
            {
                'id': s.pk,
                'term': s.get_term_display(),
                'year': s.year,
                'status': _status(s.year, s.term, current),
                'credits_attempted': s.attempted,
                'credits_earned': s.earned,
                'courses': by_semester[s.pk],
            }
            for s in semesters
        ],
        'unassigned': by_semester[None],
    }


def _evaluate(curricula, taken):
    """Evaluates the account's programs and General Education against `taken`
    (a dict of pk -> Course)."""
    gened = None

    def get_gened():
        nonlocal gened
        if gened is None:
            gened = gen_ed.evaluate(taken)
        return gened

    results = []
    used_by = defaultdict(list)
    requires_gened = False
    for curriculum in curricula:
        result, codes, program_gened = evaluate_program(
            curriculum.program, curriculum.track_id, taken, get_gened
        )
        results.append(result)
        requires_gened = requires_gened or program_gened
        for code in codes:
            used_by[code].append(curriculum.program_id)

    return {
        'programs': results,
        'courses': [course_json(c) for c in taken.values()],
        # Only present if some program requires General Education.
        'general_education': gened[0] if requires_gened else None,
        'shared_courses': {
            code: slugs for code, slugs in sorted(used_by.items())
            if len(slugs) > 1
        },
    }


def evaluate_with(account, courses):
    """The account's programs and General Education evaluated as if its passed
    classes and `courses` (Course objects, e.g. a draft plan) were all taken,
    shaped like _evaluate's result. The account must be valid."""
    curricula = list(account.programs.select_related('program').order_by('pk'))
    taken = {}
    for c in account.classes.filter(credits_earned=True).select_related('course'):
        c.course.credits = c.credits
        taken.setdefault(c.course.pk, c.course)
    for course in courses:
        taken.setdefault(course.pk, course)
    return _evaluate(curricula, taken)


def _mark(node, earlier, status):
    """Changes every `status` in `node` (an evaluation) that is satisfied there
    but not in `earlier` (the same evaluation counting fewer classes) to
    `status`. Lists are matched up by `id` if their items have one, otherwise
    by position.
    """
    if isinstance(node, dict) and isinstance(earlier, dict):
        if node.get('status') == SATISFIED and earlier.get('status') != SATISFIED:
            node['status'] = status
        for key, value in node.items():
            if key in earlier:
                _mark(value, earlier[key], status)
    elif isinstance(node, list) and isinstance(earlier, list):
        if all(isinstance(x, dict) and 'id' in x for x in node + earlier):
            earlier_by_id = {x['id']: x for x in earlier}
            pairs = [(x, earlier_by_id.get(x['id'])) for x in node]
        else:
            pairs = zip(node, earlier) if len(node) == len(earlier) else []
        for x, e in pairs:
            _mark(x, e, status)


def _mark_data(data, earlier, status):
    """_mark for a whole _evaluate result, re-bucketing each program's
    requirements."""
    for program, earlier_program in zip(data['programs'], earlier['programs']):
        requirements = requirements_of(program)
        _mark(requirements, requirements_of(earlier_program), status)
        _mark(program['degree_requirements'], earlier_program['degree_requirements'], status)
        program.update(bucket(requirements))
        program['status'] = combine([r['status'] for r in requirements])
    _mark(data['general_education'], earlier['general_education'], status)


def requirements_data(account, current_semester):
    """The body of GET /student/requirements/ (see student.views.requirements).

    `current_semester` is the semester in progress as TERM_YEAR (e.g.
    "FALL_2026"). Passed classes of earlier semesters, and ones not placed in a
    semester, count as completed. A requirement that the classes of the
    current semester would also need to meet is "in progress", and one that
    classes of later semesters would also need to meet is "planned".

    Returns (data, None), or (None, issues) with the account's ERROR-level
    validation issues if it isn't set up and valid.
    """
    issues = errors(validate_account(account))
    if issues:
        return None, issues

    current = _semester_key(current_semester)
    curricula = list(account.programs.select_related('program').order_by('pk'))
    # In the order taken (classes not placed in a semester first), each course
    # once, as first taken, so requirements show the class that met them first.
    taken, through_current, completed = {}, {}, {}
    # Failed classes don't count toward any requirement.
    classes = account.classes.filter(credits_earned=True).select_related('course', 'semester').order_by(
        F('semester__year').asc(nulls_first=True), 'semester__term', 'pk'
    )
    for c in classes:
        if c.course.pk in taken:
            continue
        status = 'completed' if c.semester is None else _status(c.semester.year, c.semester.term, current)
        # Count the credits the student took, which can differ from the catalog's
        # for variable-credit courses (course_credits reads `credits` first).
        c.course.credits = c.credits
        taken[c.course.pk] = c.course
        if status != 'planned':
            through_current[c.course.pk] = c.course
        if status == 'completed':
            completed[c.course.pk] = c.course

    data = _evaluate(curricula, taken)
    # Planned first: what's still satisfied after is met by this semester.
    if len(through_current) < len(taken):
        _mark_data(data, _evaluate(curricula, through_current), PLANNED)
    if len(completed) < len(through_current):
        _mark_data(data, _evaluate(curricula, completed), IN_PROGRESS)
    return data, None
