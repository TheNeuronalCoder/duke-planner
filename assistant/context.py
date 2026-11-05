"""The student's plan as compact text for Duke AI's system prompt.

Groq's free tier limits tokens per minute, so this summarizes the plan rather
than sending the JSON the student views return. Requirement options are listed
by code only; the model looks up details with the tools in tools.py.
"""

from django.conf import settings

from catalog.utils.status import IN_PROGRESS, PLANNED, SATISFIED, UNSATISFIED
from student.summary import requirements_data, semesters_data

MAX_CHARS = 4000
MAX_OPTIONS = 8
MAX_DEPTH = 3


def _course(c):
    """A course as "CS 149 Intro to Programming", so the model never has to guess titles."""
    title = c['title'] if len(c['title']) <= 40 else c['title'][:39] + '…'
    return f'{c["code"]} {title}'


def _codes(codes, limit=MAX_OPTIONS):
    codes = list(codes)
    more = f' (+{len(codes) - limit} more)' if len(codes) > limit else ''
    return ', '.join(codes[:limit]) + more


def _open_rules(node, depth=0):
    """Lines describing the unmet parts of a requirement tree."""
    if depth > MAX_DEPTH or node.get('status') in (SATISFIED, IN_PROGRESS, PLANNED, None):
        return []
    indent = '  ' * (depth + 2)
    line = node.get('text') or ''
    if node.get('min_count'):
        line += f' [choose {node["min_count"]}]'
    elif node.get('min_credits'):
        line += f' [{node["min_credits"]} cr]'
    options = [c['code'] for c in node.get('courses', []) if not c['taken']]
    if options:
        line += f' options: {_codes(options)}'
    if node.get('satisfied_by'):
        line += f' (have: {_codes(node["satisfied_by"])})'
    lines = [indent + '- ' + line.strip()] if line.strip() else []
    for child in node.get('children', []):
        lines += _open_rules(child, depth + 1)
    return lines


def _semesters(account):
    data = semesters_data(account, settings.CURRENT_SEMESTER)
    lines = ['Semesters:']
    for s in data['semesters']:
        courses = [
            _course(c) + ('' if c['credits_earned'] else ' (failed)') for c in s['courses']
        ]
        lines.append(
            f'  {s["term"]} {s["year"]} ({s["credits_attempted"]} cr): '
            + (', '.join(courses) or 'no courses')
        )
    if data['unassigned']:
        lines.append('  Unassigned: ' + ', '.join(_course(c) for c in data['unassigned']))
    earned = sum(s['credits_earned'] for s in data['semesters'])
    lines.append(f'  Total credits earned: {earned}')
    return lines


def _requirements(account):
    data, issues = requirements_data(account, settings.CURRENT_SEMESTER)
    if issues:
        return ['Account issues (requirements could not be checked):'] + [
            f'  - {i.message}' for i in issues
        ]

    lines = []
    for p in data['programs']:
        prog = p['program']
        lines.append(f'Program: {prog["name"]} ({prog["type"]}), overall {p["status"]}')
        lines.append(f'  Satisfied: {", ".join(r["title"] for r in p["satisfied"]) or "none"}')
        for bucket in ('in_progress', 'planned', 'unsatisfied', 'unverified'):
            for r in p[bucket]:
                lines.append(f'  {bucket.replace("_", " ").upper()}: {r["title"]}')
                if bucket == 'unsatisfied':
                    lines += _open_rules(r['rules'])
        electives = p.get('university_electives')
        if electives:
            lines.append(f'  University elective credits: {electives["credits_earned"]}')

    gened = data.get('general_education')
    if gened:
        lines.append(f'General Education: {gened["status"]}')
        for cluster in gened['clusters']:
            if cluster['status'] == UNSATISFIED:
                lines.append(f'  Missing: {cluster["name"]}')

    if data['shared_courses']:
        lines.append('Courses counted by more than one program: ' + ', '.join(data['shared_courses']))
    return lines


def build_context(account):
    lines = _semesters(account) + _requirements(account)
    text = '\n'.join(lines)
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS] + '\n[truncated]'
    return text
