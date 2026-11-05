"""Catalog tools Duke AI can call instead of having courses stuffed into its prompt.

TOOLS holds the schemas sent to Groq and run() executes a call, returning the
JSON string sent back as the tool message. Results are kept small for Groq's
tokens-per-minute limit.
"""

import json
import re

from catalog.models import Course, Subject
from catalog.utils import gened
from catalog.utils.courses import course_json, lookup_courses, normalize_code, search_courses
from catalog.utils.embeddings import semantic_search

MAX_RESULTS = 10
MAX_DESCRIPTION = 800

TOOLS = [
    {
        'type': 'function',
        'function': {
            'name': 'search_courses',
            'description': (
                'Search the JMU course catalog by course code ("CS 345", "CS 3", "MATH") '
                'or by topic ("machine learning", "intro to drawing"), which finds the '
                'courses closest in meaning. Returns at most '
                f'{MAX_RESULTS} courses with their Gen Ed areas, best match first.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {
                        'type': 'string',
                        'description': 'Course code or topic. May be empty if '
                                       'subject or gen_ed is given.',
                    },
                    'subject': {
                        'type': 'string',
                        'description': 'Only courses with this subject prefix, e.g. "CS".',
                    },
                    'gen_ed': {
                        'type': 'string',
                        'description': 'Only courses counting toward this Gen Ed cluster or '
                                       'area, by code ("C2", "C3QR") or name ("Literature").',
                    },
                },
                'required': ['query'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'get_course',
            'description': 'Look up one course by code: title, credits, description, '
                           'prerequisites, corequisites and Gen Ed areas.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'code': {'type': 'string', 'description': 'Course code, e.g. "CS 345".'},
                },
                'required': ['code'],
            },
        },
    },
]


def _with_gen_ed(course, data):
    areas = gened.areas_by_course().get(normalize_code(course.code))
    if areas:
        data['gen_ed'] = areas
    return data


def _is_code(query):
    """Whether `query` is a course code or code prefix ("CS 345", "cs3") or a
    subject ("MATH") rather than a topic."""
    if re.fullmatch(r'[A-Za-z]+\s*\d\w*', query):
        return True
    return Subject.objects.filter(code=query.upper()).exists()


def search(query='', subject=None, gen_ed=None):
    query = (query or '').strip()
    if not query and not subject and not gen_ed:
        return {'error': 'Give a query, subject or gen_ed.'}
    courses = Course.objects.all()
    if subject:
        courses = courses.filter(prefix=subject.strip().upper())
    if gen_ed:
        codes = gened.course_codes_for(gen_ed)
        if codes is None:
            return {'error': f'No Gen Ed cluster or area named "{gen_ed}".'}
        found, _ = lookup_courses(codes)
        courses = courses.filter(pk__in=found)

    if query and not _is_code(query):
        results = semantic_search(query, courses, MAX_RESULTS)
        return {'results': [_with_gen_ed(c, course_json(c)) for c in results]}

    courses = search_courses(query).filter(pk__in=courses)
    results = list(courses[:MAX_RESULTS + 1])
    data = {'results': [_with_gen_ed(c, course_json(c)) for c in results[:MAX_RESULTS]]}
    if len(results) > MAX_RESULTS:
        data['more'] = True
    return data


def get_course(code):
    found, _ = lookup_courses([code])
    if not found:
        return {'error': f'No course {code} in the catalog.'}
    course = next(iter(found.values()))
    data = _with_gen_ed(course, course_json(course, details=True))
    if data['description'] and len(data['description']) > MAX_DESCRIPTION:
        data['description'] = data['description'][:MAX_DESCRIPTION - 1] + '…'
    return data


HANDLERS = {'search_courses': search, 'get_course': get_course}


def run(name, arguments):
    """Run the tool call `name` with JSON `arguments`; returns the result as JSON.
    Bad calls return an error the model can read and correct."""
    handler = HANDLERS.get(name)
    if handler is None:
        return json.dumps({'error': f'Unknown tool {name}.'})
    try:
        args = json.loads(arguments or '{}')
        result = handler(**args)
    except (ValueError, TypeError) as e:
        result = {'error': f'Bad arguments for {name}: {e}'}
    return json.dumps(result, ensure_ascii=False)
