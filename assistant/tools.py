"""Catalog and plan tools Duke AI can call instead of having courses and the
student's plan stuffed into its prompt.

TOOLS holds the schemas sent to Groq and run() executes a call, returning the
JSON string sent back as the tool message. Results are kept small for Groq's
tokens-per-minute limit.
"""

import json
import logging
import re
from functools import partial

from catalog.models import Course, Subject
from catalog.utils import gened
from catalog.utils.courses import course_json, lookup_courses, normalize_code, search_courses
from catalog.utils.embeddings import semantic_search

from . import context

logger = logging.getLogger(__name__)

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
    {
        'type': 'function',
        'function': {
            'name': 'get_requirements',
            'description': (
                "The student's requirements by program: which are satisfied, in progress, "
                'planned, unsatisfied or unverified, with the rules and course options '
                'still missing for unsatisfied ones. Includes General Education.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'program': {
                        'type': 'string',
                        'description': 'Only this program, by name as in the plan overview '
                                       '("Computer Science") or "General Education".',
                    },
                    'status': {
                        'type': 'string',
                        'enum': list(context.BUCKETS),
                        'description': 'Only requirements with this status.',
                    },
                },
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'get_semesters',
            'description': (
                "The student's semesters, past, current and planned, with their "
                'classes, status and credits.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'term': {'type': 'string', 'description': 'Only this term, e.g. "Fall".'},
                    'year': {'type': 'string', 'description': 'Only this year, e.g. "2026".'},
                },
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
# These take the student's context.Plan first.
PLAN_HANDLERS = {'get_requirements': context.requirements, 'get_semesters': context.semesters}


def run(name, arguments, plan):
    """Run the tool call `name` with JSON `arguments` for `plan` (a
    context.Plan); returns the result as JSON. Bad calls return an error the
    model can read and correct."""
    handler = partial(PLAN_HANDLERS[name], plan) if name in PLAN_HANDLERS else HANDLERS.get(name)
    if handler is None:
        return json.dumps({'error': f'Unknown tool {name}.'})
    try:
        args = json.loads(arguments or '{}')
        # Every parameter is a string; the model sometimes sends null for ones it skips.
        if not isinstance(args, dict):
            raise TypeError('arguments must be an object')
        args = {k: v for k, v in args.items() if v is not None}
        if not all(isinstance(v, str) for v in args.values()):
            raise TypeError('every argument must be a string')
        result = handler(**args)
    except (ValueError, TypeError) as e:
        result = {'error': f'Bad arguments for {name}: {e}'}
    except Exception:
        # A failed lookup (e.g. the embedding model or database) shouldn't fail the whole chat.
        logger.exception('Duke AI tool %s failed', name)
        result = {'error': f'{name} failed. Answer without it or try another search.'}
    return json.dumps(result, ensure_ascii=False)


def append_calls(messages, message, results):
    """Appends `message`, a model reply with tool calls, to `messages`, then
    the result of each call, given by `results(call)` as a string."""
    messages.append({
        'role': 'assistant',
        'content': message.content or '',
        'tool_calls': [
            {
                'id': call.id,
                'type': 'function',
                'function': {'name': call.function.name, 'arguments': call.function.arguments},
            }
            for call in message.tool_calls
        ],
    })
    for call in message.tool_calls:
        messages.append({'role': 'tool', 'tool_call_id': call.id, 'content': results(call)})
