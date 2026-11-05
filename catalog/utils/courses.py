import re

from django.db.models import Case, IntegerField, Q, Value, When

from ..models import Course


def course_credits(course):
    return course.credits if course.credits is not None else course.min_credits


def course_number(course):
    """Leading numeric part of a course number, e.g. "499A" -> 499."""
    match = re.match(r'\d+', course.number)
    return int(match.group()) if match else None


def normalize_code(code):
    return ' '.join(code.upper().split())


def course_json(course, details=False):
    """`details` adds the description, prerequisites and corequisites."""
    data = {
        'code': course.code,
        'title': course.title,
        'credits': course_credits(course),
        'min_credits': course.min_credits,
        'max_credits': course.max_credits,
    }
    if details:
        data['description'] = course.description
        data['prerequisites'] = course.prerequisites
        data['corequisites'] = course.corequisites
    return data


def all_terms(fields, terms):
    """Q matching rows where every term appears (case-insensitively) in some field."""
    q = Q()
    for term in terms:
        any_field = Q()
        for field in fields:
            any_field |= Q(**{f'{field}__icontains': term})
        q &= any_field
    return q


def search_courses(query, subjects=None):
    """Courses matching `query` as a code ("COB 191", "cob19", "COB") or as words
    that must all appear in the title, ranked exact code, then code prefix, then
    title match. An empty query matches every course."""
    courses = Course.objects.all()
    if subjects:
        courses = courses.filter(prefix__in=subjects)
    if not query:
        return courses

    code_match = re.fullmatch(r'([A-Za-z]+)\s*(\d\w*)?', query)
    exact = prefix = Q(pk__in=[])
    if code_match:
        subject, number = code_match.group(1).upper(), code_match.group(2)
        if number:
            exact = Q(prefix=subject, number__iexact=number)
            prefix = Q(prefix=subject, number__istartswith=number)
        else:
            prefix = Q(prefix=subject)
    title = all_terms(['title'], query.split())

    return courses.filter(exact | prefix | title).annotate(
        rank=Case(
            When(exact, then=Value(0)),
            When(prefix, then=Value(1)),
            When(title__istartswith=query, then=Value(2)),
            default=Value(3),
            output_field=IntegerField(),
        )
    ).order_by('rank', 'prefix', 'number')


def lookup_courses(codes):
    """Map course codes like "COB 191" to Course objects; returns (found, unknown)."""
    wanted = {}
    unknown = []
    for code in codes:
        match = re.fullmatch(r'([A-Za-z]+)\s*(\S+)', code)
        if not match:
            unknown.append(code)
            continue
        wanted[(match.group(1).upper(), match.group(2).upper())] = code

    found = {}
    prefixes = {prefix for prefix, _ in wanted}
    for course in Course.objects.filter(prefix__in=prefixes):
        if (course.prefix_id, course.number.upper()) in wanted:
            found[course.pk] = course
    found_keys = {(c.prefix_id, c.number.upper()) for c in found.values()}
    unknown.extend(code for key, code in wanted.items() if key not in found_keys)
    return found, unknown
