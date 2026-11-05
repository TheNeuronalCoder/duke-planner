"""General Education ("The Human Community") evaluation against taken courses.

The requirements live in data/gened.yaml: clusters made of areas, each area
requiring `choose` courses from its list.
"""
import re
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import yaml

from .courses import course_credits, normalize_code
from .status import SATISFIED, UNSATISFIED

DATA_PATH = Path(__file__).resolve().parent.parent / 'data' / 'gened.yaml'

# Placeholder groups in program requirements that stand for the whole Gen Ed
# program, e.g. "General Education" or "General Education requirements" (but not
# "Philosophy course (in addition to General Education courses)").
PLACEHOLDER_RE = re.compile(
    r'\s*general education(\s+(courses|requirements?))?\s*', re.IGNORECASE
)


def is_gened_placeholder(text):
    return bool(text) and PLACEHOLDER_RE.fullmatch(text) is not None


@lru_cache(maxsize=1)
def load():
    with open(DATA_PATH) as f:
        return yaml.safe_load(f)


def cluster_id_for(text):
    """The id of the cluster a placeholder names, e.g. "Arts and Humanities" -> "C2"."""
    wanted = (text or '').strip().lower()
    for cluster in load()['clusters']:
        if cluster['name'].lower() == wanted:
            return cluster['id']
    return None


@lru_cache(maxsize=1)
def areas_by_course():
    """Normalized course code -> codes of the Gen Ed areas it counts toward."""
    areas = {}
    for cluster in load()['clusters']:
        for area in cluster['areas']:
            for c in area['courses']:
                areas.setdefault(normalize_code(c['code']), []).append(area['code'])
    return areas


def course_codes_for(key):
    """Codes of the courses in the Gen Ed cluster or area `key` names, by id or
    code ("C2", "C2L") or name ("Literature"), or None if nothing matches."""
    wanted = key.strip().lower()
    for cluster in load()['clusters']:
        if wanted in (cluster['id'].lower(), cluster['name'].lower()):
            return [c['code'] for area in cluster['areas'] for c in area['courses']]
        for area in cluster['areas']:
            if wanted in (area['code'].lower(), area['name'].lower()):
                return [c['code'] for c in area['courses']]
    return None


def _assign(areas, candidates, exclusive):
    """Pick which taken courses count toward each area of one cluster.

    Each course counts toward at most one area, and at most one course from each
    `exclusive` set may be used. Maximizes the number of satisfied areas, then the
    number of courses used. Returns a list of chosen code tuples, one per area.
    """
    exclusive = [set(map(normalize_code, group)) for group in exclusive]

    def options(area, cands):
        k = min(area['choose'], len(cands))
        for size in range(k, -1, -1):
            yield from combinations(cands, size)

    best = (None, (-1, -1))

    def search(i, used, chosen, score):
        nonlocal best
        if i == len(areas):
            if score > best[1]:
                best = (list(chosen), score)
            return
        area = areas[i]
        cands = [c for c in candidates[i] if c not in used]
        for combo in options(area, cands):
            trial = used | set(combo)
            if any(len(trial & group) > 1 for group in exclusive):
                continue
            satisfied = len(combo) >= area['choose']
            search(
                i + 1,
                trial,
                chosen + [combo],
                (score[0] + satisfied, score[1] + len(combo)),
            )

    search(0, frozenset(), [], (0, 0))
    return best[0]


def evaluate(taken):
    """Evaluate Gen Ed against `taken` (a dict of pk -> Course, in the order
    taken). Earlier courses are preferred, and an area's `satisfied_by` lists
    the courses counted toward it followed by its other taken options no area
    of the cluster counts, all in the order taken.

    Returns (json, courses_used) where courses_used are the Course objects counted
    toward Gen Ed (not the extra options).
    """
    data = load()
    taken_by_code = {normalize_code(c.code): c for c in taken.values()}
    order = {code: i for i, code in enumerate(taken_by_code)}

    clusters = []
    all_used = []
    for cluster in data['clusters']:
        areas = cluster['areas']
        candidates = [
            sorted(
                {
                    code
                    for code in (normalize_code(c['code']) for c in area['courses'])
                    if code in taken_by_code
                },
                key=order.get,
            )
            for area in areas
        ]
        chosen = _assign(areas, candidates, cluster.get('exclusive', []))
        used_codes = {code for combo in chosen for code in combo}
        # Taken options no area counts, listed under the first area they fit.
        extras = [[] for _ in areas]
        listed = set(used_codes)
        for i, cands in enumerate(candidates):
            for code in cands:
                if code not in listed:
                    extras[i].append(code)
                    listed.add(code)

        # Courses with an embedded lab also satisfy a "lab experience" area.
        embedded_labs = sorted(
            {
                normalize_code(c['code'])
                for area in areas
                for c in area['courses']
                if c.get('includes_lab') and normalize_code(c['code']) in used_codes
            },
            key=order.get,
        )

        area_nodes = []
        for area, combo, extra in zip(areas, chosen, extras):
            satisfied_by = sorted(combo, key=order.get) + extra
            status = SATISFIED if len(combo) >= area['choose'] else UNSATISFIED
            if status == UNSATISFIED and area.get('satisfied_by_embedded_lab') and embedded_labs:
                status = SATISFIED
                satisfied_by = embedded_labs
            area_nodes.append(
                {
                    'code': area['code'],
                    'name': area['name'],
                    'choose': area['choose'],
                    'notes': area['notes'],
                    'status': status,
                    'satisfied_by': satisfied_by,
                    'courses': [
                        {
                            'code': c['code'],
                            'title': c['title'],
                            'taken': normalize_code(c['code']) in taken_by_code,
                            **({'note': c['note']} if 'note' in c else {}),
                        }
                        for c in area['courses']
                    ],
                }
            )

        all_used.extend(taken_by_code[code] for code in used_codes)
        clusters.append(
            {
                'id': cluster['id'],
                'name': cluster['name'],
                'notes': cluster['notes'],
                'status': (
                    SATISFIED
                    if all(a['status'] == SATISFIED for a in area_nodes)
                    else UNSATISFIED
                ),
                'areas': area_nodes,
            }
        )

    node = {
        'name': data['name'],
        'catalog_year': data['catalog_year'],
        'source': data['source'],
        'total_credits': data['total_credits'],
        'credits_earned': sum(course_credits(c) for c in all_used),
        'status': (
            SATISFIED
            if all(c['status'] == SATISFIED for c in clusters)
            else UNSATISFIED
        ),
        'satisfied_by': sorted(c.code for c in all_used),
        'clusters': clusters,
    }
    return node, all_used
