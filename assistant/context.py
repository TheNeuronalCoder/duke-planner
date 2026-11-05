"""The student's plan for Duke AI: a short overview for its system prompt, and
the details its plan tools (see tools.py) return.

Groq's free tier limits tokens per minute, so the prompt only summarizes each
program's status; the model looks up semesters and unmet requirements with
get_semesters and get_requirements. Requirement options are listed by code
only; the model looks up course details with the catalog tools.
"""

from functools import cached_property

from django.conf import settings

from catalog.utils.status import IN_PROGRESS, PLANNED, SATISFIED, UNSATISFIED
from student.summary import requirements_data, semesters_data

MAX_OPTIONS = 8
MAX_DEPTH = 3
BUCKETS = ('satisfied', 'in_progress', 'planned', 'unsatisfied', 'unverified')
GENERAL_EDUCATION = 'General Education'


class Plan:
    """An account's plan, evaluated at most once per chat request however many
    tools ask for it."""

    def __init__(self, account):
        self.account = account

    @cached_property
    def semesters(self):
        return semesters_data(self.account, settings.CURRENT_SEMESTER)

    @cached_property
    def requirements(self):
        """(data, issues), as requirements_data returns."""
        return requirements_data(self.account, settings.CURRENT_SEMESTER)


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
    lines = ['  ' * depth + line.strip()] if line.strip() else []
    for child in node.get('children', []):
        lines += _open_rules(child, depth + 1)
    return lines


def _issues(issues):
    return 'Account issues (requirements could not be checked): ' + '; '.join(
        i.message for i in issues
    )


def _program_name(program):
    return f'{program["name"]} ({program["type"]})'


def build_context(plan):
    """The overview of `plan` for the system prompt."""
    semesters = plan.semesters['semesters']
    earned = sum(s['credits_earned'] for s in semesters)
    lines = [f'Total credits earned: {earned}.']
    if semesters:
        first, last = semesters[0], semesters[-1]
        lines[0] += (f' {len(semesters)} semesters, {first["term"]} {first["year"]} to '
                     f'{last["term"]} {last["year"]}.')
    current = [s for s in semesters if s['status'] == IN_PROGRESS]
    if current:
        lines.append(f'Current semester: {current[0]["term"]} {current[0]["year"]}.')

    data, issues = plan.requirements
    if issues:
        return '\n'.join(lines + [_issues(issues)])
    for p in data['programs']:
        counts = ', '.join(
            f'{len(p[b])} {b.replace("_", " ")}' for b in BUCKETS if p[b]
        )
        lines.append(f'Program: {_program_name(p["program"])}, overall {p["status"]}'
                     + (f' ({counts})' if counts else ''))
    gened = data.get('general_education')
    if gened:
        missing = [c['name'] for c in gened['clusters'] if c['status'] == UNSATISFIED]
        lines.append(f'{GENERAL_EDUCATION}: {gened["status"]}'
                     + (f', missing {", ".join(missing)}' if missing else ''))
    return '\n'.join(lines)


def semesters(plan, term=None, year=None):
    """The plan's semesters and their classes, optionally only those of `term`
    ("Fall") and/or `year`."""
    data = plan.semesters
    result = []
    for s in data['semesters']:
        if term and s['term'].lower() != term.strip().lower():
            continue
        if year and str(s['year']) != year.strip():
            continue
        result.append({
            'term': s['term'],
            'year': s['year'],
            'status': s['status'],
            'credits_attempted': s['credits_attempted'],
            'credits_earned': s['credits_earned'],
            'courses': [
                _course(c) + ('' if c['credits_earned'] else ' (failed)') for c in s['courses']
            ],
        })
    if not result and (term or year):
        return {'error': f'No semester matching {" ".join(filter(None, [term, year]))} in the plan.'}
    response = {'semesters': result}
    if data['unassigned'] and not (term or year):
        response['unassigned'] = [_course(c) for c in data['unassigned']]
    return response


def _requirement(r, bucket):
    data = {'title': r['title']}
    if bucket == 'unsatisfied':
        data['missing'] = _open_rules(r['rules'])
    return data


def _gened(gened, status):
    clusters = []
    for cluster in gened['clusters']:
        if status and cluster['status'] != status:
            continue
        data = {'name': cluster['name'], 'status': cluster['status']}
        if cluster['status'] == UNSATISFIED:
            data['missing'] = [
                f'{a["code"]} {a["name"]} [choose {a["choose"]}]'
                + (f' (have: {_codes(a["satisfied_by"])})' if a['satisfied_by'] else '')
                for a in cluster['areas'] if a['status'] == UNSATISFIED
            ]
        clusters.append(data)
    return {'name': GENERAL_EDUCATION, 'status': gened['status'], 'clusters': clusters}


def requirements(plan, program=None, status=None):
    """The plan's requirements by program, optionally only `program` (name or
    slug, or "General Education") and only those with `status`."""
    data, issues = plan.requirements
    if issues:
        return {'error': _issues(issues)}
    if status:
        status = status.strip().lower().replace(' ', '_')
        if status not in BUCKETS:
            return {'error': f'status must be one of {", ".join(BUCKETS)}.'}
    wanted = program.strip().lower() if program else None

    result = []
    for p in data['programs']:
        names = {p['program']['slug'], p['program']['name'].lower(),
                 _program_name(p['program']).lower()}
        if wanted and wanted not in names:
            continue
        item = {'program': _program_name(p['program']), 'status': p['status']}
        for b in BUCKETS:
            if p[b] and (not status or status == b):
                item[b] = [_requirement(r, b) for r in p[b]]
        electives = p.get('university_electives')
        if electives:
            item['university_elective_credits'] = electives['credits_earned']
        result.append(item)

    gened = data.get('general_education')
    if gened and (not wanted or wanted == GENERAL_EDUCATION.lower()):
        # Clusters are only ever satisfied or unsatisfied.
        if not status or status in (SATISFIED, UNSATISFIED):
            result.append(_gened(gened, status))

    if not result and wanted:
        names = [_program_name(p['program']) for p in data['programs']]
        if gened:
            names.append(GENERAL_EDUCATION)
        return {'error': f'No program "{program}". The student has: {", ".join(names)}.'}
    response = {'programs': result}
    if data['shared_courses'] and not wanted:
        response['shared_courses'] = list(data['shared_courses'])
    return response
