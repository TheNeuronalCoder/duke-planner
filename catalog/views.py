import re

from django.db.models import Case, IntegerField, TextField, Value, When
from django.db.models.functions import Cast
from django.http import JsonResponse
from rest_framework.decorators import (
    api_view, authentication_classes, parser_classes, permission_classes,
)
from rest_framework.parsers import JSONParser
from rest_framework.permissions import AllowAny

from .models import Program
from .serializers import CourseSearchSerializer, CoursesSerializer, ProgramSearchSerializer
from .utils.courses import all_terms, course_json, lookup_courses, search_courses as find_courses


@api_view(['GET'])
@parser_classes([JSONParser])
@authentication_classes([])
@permission_classes([AllowAny])
def courses(request):
    """GET /catalog/courses?codes=<code>[,<code>...]

    Returns {code: course} for each requested code, keyed exactly as given, with
    course details included. Codes that don't match a course are omitted.
    """
    serializer = CoursesSerializer(data=request.query_params)
    serializer.is_valid(raise_exception=True)
    codes = serializer.validated_data['codes']

    found, _ = lookup_courses(codes)
    by_key = {(c.prefix_id, c.number.upper()): c for c in found.values()}

    results = {}
    for code in codes:
        match = re.fullmatch(r'([A-Za-z]+)\s*(\S+)', code)
        course = match and by_key.get((match.group(1).upper(), match.group(2).upper()))
        if course:
            results[code] = course_json(course, details=True)
    return JsonResponse(results)


@api_view(['GET'])
@parser_classes([JSONParser])
@authentication_classes([])
@permission_classes([AllowAny])
def search_courses(request):
    """GET /catalog/search/course?q=<text>[&subject=<code>[,<code>...]][&limit=<n>]

    Matches course codes ("COB 191", "cob19", "COB") and titles. Every word of a
    title search must appear in the title. Results are ranked exact code, then
    code prefix, then title match, and capped at `limit` (default 20, max 100).
    """
    serializer = CourseSearchSerializer(data=request.query_params)
    serializer.is_valid(raise_exception=True)
    params = serializer.validated_data
    query = params['q']

    courses = find_courses(query, params['subject'])
    return JsonResponse({'results': [course_json(c) for c in courses[:params['limit']]]})


@api_view(['GET'])
@parser_classes([JSONParser])
@authentication_classes([])
@permission_classes([AllowAny])
def search_programs(request):
    """GET /catalog/search/program?q=<text>[&level=<level>][&type=<type>[,<type>...]][&limit=<n>]

    Every word of `q` must appear in the program's name or keywords. Results are
    ranked name-prefix, then name, then keyword-only matches, and capped at
    `limit` (default 20, max 100). Each result lists its tracks.
    """
    serializer = ProgramSearchSerializer(data=request.query_params)
    serializer.is_valid(raise_exception=True)
    params = serializer.validated_data
    query = params['q']

    programs = Program.objects.annotate(
        keywords_text=Cast('keywords', TextField())
    ).filter(all_terms(['name', 'keywords_text'], query.split()))
    if 'level' in params:
        programs = programs.filter(level=params['level'])
    if params['type']:
        programs = programs.filter(type__in=params['type'])
    programs = (
        programs.annotate(
            rank=Case(
                When(name__istartswith=query, then=Value(0)),
                When(name__icontains=query, then=Value(1)),
                When(all_terms(['name'], query.split()), then=Value(2)),
                default=Value(3),
                output_field=IntegerField(),
            )
        )
        .order_by('rank', 'name')
        .prefetch_related('tracks')
    )

    return JsonResponse(
        {
            'results': [
                {
                    'slug': p.slug,
                    'name': p.name,
                    'level': p.level,
                    'type': p.type,
                    'url': p.url,
                    'tracks': [{'id': t.pk, 'name': t.name} for t in p.tracks.all()],
                }
                for p in programs[:params['limit']]
            ]
        }
    )

