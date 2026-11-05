"""Graduation plans drafted by Duke AI.

The model gets the student's unmet requirements and the terms left to plan,
looks courses up with the catalog tools and submits a draft with submit_plan.
check() then drops what can't be saved (unknown, repeated or over-cap courses,
terms that aren't open) and evaluates the student's requirements with the
draft added. Problems go back to the model for one revision; what remains is
returned as warnings. Nothing is saved: the client applies a draft with POST
/student/semesters/. Prerequisites aren't checked; the model orders courses
by their prerequisite text.
"""

import json
import time

import groq
from django.conf import settings

from catalog.utils.courses import course_credits, course_json, lookup_courses, normalize_code
from catalog.utils.status import UNSATISFIED
from student.models import Semester
from student.summary import _semester_key, evaluate_with

from . import context, tools

DEFAULT_CREDITS = 15
MIN_CREDITS = 12
MAX_CREDITS = 18
DEGREE_CREDITS = 120
# How many years past the current one a plan may reach (validation allows 10).
MAX_YEARS = 6
MAX_PREFERENCES = 500
MAX_FILLS = 3
# Model calls per draft; the last one has to submit.
MAX_ROUNDS = 8
# Seconds after which the next call has to submit and no revision is asked for.
TIME_LIMIT = 50

SYSTEM_PROMPT = """You are Duke AI, an academic planner for James Madison University \
(JMU) students. Draft the semesters this student has left until graduation and \
submit them with submit_plan.

Rules:
- Plan only the terms under "Terms to plan", earliest first. Leave out terms that \
aren't needed; graduating early is fine.
- Each term may have at most {max_credits} credits.
- Cover every unsatisfied requirement listed. Where a requirement lists options, \
pick from them; for open-ended ones (electives, Gen Ed areas) use search_courses.
- Use get_course to check a course's credits and prerequisites, and put every course \
in a later term than the courses its prerequisites name.
- Never plan a course under "Already taken or planned".
- Prefer courses that count toward more than one requirement.
- If the student's degree needs {degree_credits} credits, add electives to reach it.
- Only use course codes from the requirements or tool results; never invent courses.
- In each course's `fills`, name the requirement(s) it counts toward as titled in \
the requirements, e.g. "Computer Science: Core courses" or "General Education: Cluster 3".
- In `notes`, tell the student in one or two sentences anything they should know, \
e.g. a requirement you couldn't fit or a prerequisite to watch.
"""

SUBMIT_PLAN = {
    'type': 'function',
    'function': {
        'name': 'submit_plan',
        'description': 'Submit the finished plan: the courses for each term to plan.',
        'parameters': {
            'type': 'object',
            'properties': {
                'semesters': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'term': {'type': 'string', 'enum': list(Semester.Term.labels)},
                            'year': {'type': 'integer'},
                            'courses': {
                                'type': 'array',
                                'items': {
                                    'type': 'object',
                                    'properties': {
                                        'code': {'type': 'string', 'description': 'e.g. "CS 345"'},
                                        'fills': {'type': 'array', 'items': {'type': 'string'}},
                                    },
                                    'required': ['code'],
                                },
                            },
                        },
                        'required': ['term', 'year', 'courses'],
                    },
                },
                'notes': {'type': 'string'},
            },
            'required': ['semesters'],
        },
    },
}
PLAN_TOOLS = [
    t for t in tools.TOOLS if t['function']['name'] in ('search_courses', 'get_course')
] + [SUBMIT_PLAN]
FORCE_SUBMIT = {'type': 'function', 'function': {'name': 'submit_plan'}}


class PlanError(Exception):
    """A request the planner can't serve, with a message for the student."""


class InvalidAccount(Exception):
    def __init__(self, issues):
        self.issues = issues


def parse_options(body):
    """The planner options from a request body; raises PlanError if malformed."""
    body = body if isinstance(body, dict) else {}
    max_credits = body.get('max_credits', DEFAULT_CREDITS)
    if not isinstance(max_credits, int) or not MIN_CREDITS <= max_credits <= MAX_CREDITS:
        raise PlanError(f'max_credits must be a whole number from {MIN_CREDITS} to {MAX_CREDITS}.')
    include_summer = body.get('include_summer', False)
    if not isinstance(include_summer, bool):
        raise PlanError('include_summer must be true or false.')
    preferences = body.get('preferences') or ''
    if not isinstance(preferences, str) or len(preferences) > MAX_PREFERENCES:
        raise PlanError(f'preferences must be text of at most {MAX_PREFERENCES} characters.')

    graduate_by = body.get('graduate_by')
    if graduate_by is not None:
        term = Semester.Term.__members__.get(str(
            graduate_by.get('term', '') if isinstance(graduate_by, dict) else ''
        ).upper())
        year = graduate_by.get('year') if isinstance(graduate_by, dict) else None
        if term is None or not isinstance(year, int):
            raise PlanError('graduate_by must be {"term": "spring"|"summer"|"fall"|"winter", "year": 2029}.')
        graduate_by = (year, term)

    return {
        'max_credits': max_credits,
        'include_summer': include_summer,
        'graduate_by': graduate_by,
        'preferences': preferences.strip(),
    }


def open_terms(account, options):
    """The (year, Semester.Term) the plan may fill, in order: those after the
    current semester, through `graduate_by`, that the account doesn't have yet.
    Winters, and summers unless asked for, are skipped."""
    current = _semester_key(settings.CURRENT_SEMESTER)
    allowed = [Semester.Term.SPRING, Semester.Term.FALL]
    if options['include_summer']:
        allowed.append(Semester.Term.SUMMER)
    last = options['graduate_by'] or (current[0] + MAX_YEARS, Semester.Term.WINTER)
    existing = set(account.semesters.values_list('year', 'term'))
    return [
        (year, term)
        for year in range(current[0], current[0] + MAX_YEARS + 1)
        for term in sorted(allowed)
        if current < (year, term) <= last and (year, term) not in existing
    ]


def _term_name(key):
    year, term = key
    return f'{Semester.Term(term).label} {year}'


def _needs_degree_credits(account):
    return account.programs.filter(program__level='undergraduate', program__type__startswith='B.').exists()


def _unsatisfied(account, courses):
    """What the account would still be missing with `courses` added: each
    unsatisfied requirement as "Program: title", and the credit total if short."""
    data = evaluate_with(account, courses)
    missing = [
        f'{p["program"]["name"]}: {r["title"]}' for p in data['programs'] for r in p['unsatisfied']
    ]
    gened = data['general_education']
    if gened:
        missing += [
            f'{context.GENERAL_EDUCATION}: {c["name"]}'
            for c in gened['clusters'] if c['status'] == UNSATISFIED
        ]
    total = sum(c['credits'] or 0 for c in data['courses'])
    if total < DEGREE_CREDITS and _needs_degree_credits(account):
        missing.append(f'Total credits: {total} of the {DEGREE_CREDITS} needed')
    return missing


def _taken_codes(plan):
    """Codes of the plan's passed or planned classes, as normalize_code gives them."""
    data = plan.semesters
    classes = [c for s in data['semesters'] for c in s['courses']] + data['unassigned']
    return {normalize_code(c['code']) for c in classes if c['credits_earned']}


def check(account, arguments, terms, max_credits, taken):
    """Checks a submit_plan call's JSON `arguments` against the open `terms`,
    `max_credits` per term and the `taken` codes.

    Returns {"semesters", "notes", "problems", "unsatisfied"}: the semesters
    without the courses and terms that can't be saved, which `problems`
    explains, and what the account would still be missing with them added.
    """
    try:
        args = json.loads(arguments or '{}')
    except ValueError:
        args = None
    if not isinstance(args, dict) or not isinstance(args.get('semesters'), list):
        return {'semesters': [], 'notes': '', 'unsatisfied': _unsatisfied(account, []),
                'problems': ['submit_plan needs {"semesters": [...]}.']}

    problems = []
    planned = set(taken)
    seen_terms = set()
    semesters = []
    for s in args['semesters']:
        if not isinstance(s, dict):
            continue
        term = Semester.Term.__members__.get(str(s.get('term', '')).upper())
        key = (s.get('year'), term)
        name = f'{s.get("term")} {s.get("year")}'
        if key not in terms:
            problems.append(f"{name} isn't one of the terms to plan.")
            continue
        if key in seen_terms:
            problems.append(f'{name} is listed twice.')
            continue
        seen_terms.add(key)

        entries = [c for c in s.get('courses') or [] if isinstance(c, dict) and isinstance(c.get('code'), str)]
        found, _ = lookup_courses([c['code'] for c in entries])
        by_code = {normalize_code(c.code): c for c in found.values()}
        credits, courses = 0, []
        for entry in entries:
            code = normalize_code(entry['code'])
            course = by_code.get(code)
            if course is None:
                problems.append(f'{entry["code"]} is not in the catalog.')
            elif code in planned:
                problems.append(f'{code} is already taken or planned.')
            elif credits + course_credits(course) > max_credits:
                problems.append(f'{name} would be over {max_credits} credits, so {code} was left out.')
            else:
                planned.add(code)
                credits += course_credits(course)
                fills = [f[:80] for f in entry.get('fills') or [] if isinstance(f, str)][:MAX_FILLS]
                courses.append((course, fills))
        if courses:
            semesters.append((key, credits, courses))

    semesters.sort(key=lambda s: s[0])
    notes = args.get('notes')
    return {
        'semesters': [
            {
                'term': Semester.Term(key[1]).label,
                'year': key[0],
                'credits': credits,
                'courses': [{**course_json(c), 'fills': fills} for c, fills in courses],
            }
            for key, credits, courses in semesters
        ],
        'notes': notes.strip() if isinstance(notes, str) else '',
        'problems': problems,
        'unsatisfied': _unsatisfied(account, [c for _, _, courses in semesters for c, _ in courses]),
    }


def _prompt(plan, terms, options, taken):
    """The user message that sets out what to plan."""
    unmet = context.requirements(plan, status=UNSATISFIED)
    parts = [
        context.build_context(plan),
        'Unsatisfied requirements:\n' + json.dumps(unmet, ensure_ascii=False),
        'Already taken or planned: ' + (', '.join(sorted(taken)) or 'nothing'),
        'Terms to plan: ' + ', '.join(_term_name(t) for t in terms),
        f'At most {options["max_credits"]} credits per term.',
    ]
    if options['preferences']:
        parts.append("The student's preferences: " + options['preferences'])
    return '\n\n'.join(parts)


def _submission(client, messages, plan, start):
    """Runs the model, executing its catalog lookups, until it calls
    submit_plan; returns (its message, the submit_plan call)."""
    for i in range(MAX_ROUNDS):
        last = i == MAX_ROUNDS - 1 or time.monotonic() - start > TIME_LIMIT
        message = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=messages,
            tools=PLAN_TOOLS,
            tool_choice=FORCE_SUBMIT if last else 'required',
            temperature=0.2,
            max_tokens=4000,
            reasoning_effort='low',
        ).choices[0].message
        calls = message.tool_calls or []
        submitted = next((c for c in calls if c.function.name == 'submit_plan'), None)
        if submitted:
            return message, submitted
        if not calls:
            messages.append({'role': 'assistant', 'content': message.content or ''})
            messages.append({'role': 'user', 'content': 'Submit the plan with submit_plan.'})
            continue
        tools.append_calls(
            messages, message, lambda call: tools.run(call.function.name, call.function.arguments, plan)
        )
    raise PlanError("Duke AI couldn't finish a plan. Try again.")


def generate(account, options):
    """Drafts a graduation plan for `account` with `options` (see
    parse_options); returns the body of POST /assistant/plan/.

    Raises InvalidAccount if the account isn't valid, PlanError if there's
    nothing to plan, and groq errors as they come.
    """
    plan = context.Plan(account)
    _, issues = plan.requirements
    if issues:
        raise InvalidAccount(issues)
    if not _unsatisfied(account, []):
        return {'semesters': [], 'notes': 'Your classes already cover every requirement.',
                'warnings': [], 'unsatisfied': []}
    terms = open_terms(account, options)
    if not terms:
        raise PlanError('There are no terms left to plan. Pick a later graduation term '
                        "or remove planned semesters you don't need.")

    taken = _taken_codes(plan)
    prompt = SYSTEM_PROMPT.format(max_credits=options['max_credits'], degree_credits=DEGREE_CREDITS)
    messages = [
        {'role': 'system', 'content': prompt},
        {'role': 'user', 'content': _prompt(plan, terms, options, taken)},
    ]
    client = groq.Groq(api_key=settings.GROQ_API_KEY, timeout=30, max_retries=1)
    start = time.monotonic()
    message, call = _submission(client, messages, plan, start)
    draft = check(account, call.function.arguments, terms, options['max_credits'], taken)
    if (draft['problems'] or draft['unsatisfied']) and time.monotonic() - start <= TIME_LIMIT:
        feedback = json.dumps({
            'problems': draft['problems'],
            'still_unsatisfied': draft['unsatisfied'],
            'instruction': 'Fix what you can and call submit_plan again with the whole plan.',
        }, ensure_ascii=False)
        tools.append_calls(messages, message, lambda c: feedback if c is call else tools.run(
            c.function.name, c.function.arguments, plan
        ))
        _, call = _submission(client, messages, plan, start)
        draft = check(account, call.function.arguments, terms, options['max_credits'], taken)

    return {
        'semesters': draft['semesters'],
        'notes': draft['notes'],
        'warnings': draft['problems'],
        'unsatisfied': draft['unsatisfied'],
    }
