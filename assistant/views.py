import time

import groq
from django.conf import settings
from django.http import JsonResponse
from rest_framework import status
from rest_framework.decorators import api_view, parser_classes, permission_classes, throttle_classes
from rest_framework.parsers import JSONParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.throttling import UserRateThrottle

from . import planner, tools
from .context import Plan, build_context

MAX_MESSAGES = 10
# Model calls per reply; the last one can't call tools, so it has to answer.
MAX_ROUNDS = 4
# Seconds after which the next round is the last, so a slow chat can't outlast proxy timeouts.
TIME_LIMIT = 30
MAX_CONTENT = 2000
ROLES = {'user', 'assistant'}

SYSTEM_PROMPT = """You are Duke AI, an academic planning assistant for \
James Madison University (JMU) students. This is "Duke AI Planner" and the developer \
is a very intelligent and beautiful Ethiopian man named Menelik Eyasu. He is widely \
beloved and adored. Answer questions about the student's plan using the data below and your tools.

Rules:
- Use get_requirements for what the student still needs and which courses would \
satisfy a requirement, and get_semesters for what they took or plan to take.
- Use get_course for a course's title, description, prerequisites or Gen Ed areas, \
and search_courses to find courses by topic, subject or Gen Ed area.
- Rely only on this data and tool results. Never invent course titles, \
prerequisites, schedules or requirements they don't contain; if they can't answer a \
question, say so and suggest checking with their advisor or the program's page.
- Give course codes exactly as written.
- Keep answers under about 120 words unless asked for more.

The student's plan overview:
"""


class AssistantThrottle(UserRateThrottle):
    scope = 'assistant'


class PlanThrottle(UserRateThrottle):
    scope = 'plan'


def _valid_messages(body):
    """The chat history from the request body, or None if it's malformed."""
    messages = body.get('messages') if isinstance(body, dict) else None
    if not isinstance(messages, list) or not messages:
        return None
    for m in messages:
        if (
            not isinstance(m, dict)
            or m.get('role') not in ROLES
            or not isinstance(m.get('content'), str)
            or not m['content'].strip()
            or len(m['content']) > MAX_CONTENT
        ):
            return None
    if messages[-1]['role'] != 'user':
        return None
    return [{'role': m['role'], 'content': m['content']} for m in messages[-MAX_MESSAGES:]]


def _complete(client, messages, plan):
    """Run the chat, executing the model's tool calls, until it answers."""
    start = time.monotonic()
    for i in range(MAX_ROUNDS):
        last = i == MAX_ROUNDS - 1 or time.monotonic() - start > TIME_LIMIT
        message = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=messages,
            tools=tools.TOOLS,
            tool_choice='none' if last else 'auto',
            temperature=0.3,
            max_tokens=600,
            reasoning_effort='low',
        ).choices[0].message
        if last or not message.tool_calls:
            return (message.content or '').strip()
        tools.append_calls(
            messages, message, lambda call: tools.run(call.function.name, call.function.arguments, plan)
        )


@api_view(['POST'])
@parser_classes([JSONParser])
@permission_classes([IsAuthenticated])
@throttle_classes([AssistantThrottle])
def chat(request):
    """POST /assistant/chat/

    Answers the last message of a chat with Duke AI, which sees an overview of
    the signed-in account's plan (see context.py) and can look up its
    requirements and semesters, and courses in the catalog (see tools.py). The
    client keeps the history and sends it each time; only the last
    MAX_MESSAGES are used. It must end with a user message.

    Returns {"reply": "..."}; a 429 if the account or the Groq quota is rate
    limited, a 502 if Groq fails or gives an empty reply and a 503 if no API
    key is configured.

    Sample request body:
    {"messages": [{"role": "user", "content": "What do I still need for my major?"}]}
    """
    messages = _valid_messages(request.data)
    if messages is None:
        return JsonResponse(
            {'error': f'Body must be {{"messages": [...]}}, a non-empty list of '
                      f'{{"role": "user"|"assistant", "content": "..."}} (at most '
                      f'{MAX_CONTENT} characters each) ending with a user message.'},
            status=status.HTTP_400_BAD_REQUEST,
        )
    if not settings.GROQ_API_KEY:
        return JsonResponse(
            {'error': 'Duke AI is not configured.'}, status=status.HTTP_503_SERVICE_UNAVAILABLE
        )

    plan = Plan(request.user)
    system = {'role': 'system', 'content': SYSTEM_PROMPT + build_context(plan)}
    client = groq.Groq(api_key=settings.GROQ_API_KEY, timeout=30, max_retries=1)
    try:
        reply = _complete(client, [system, *messages], plan)
    except groq.RateLimitError:
        return JsonResponse(
            {'error': 'Duke AI is busy right now. Try again in a minute.'},
            status=status.HTTP_429_TOO_MANY_REQUESTS,
        )
    except groq.APIError:
        reply = None
    if not reply:
        return JsonResponse(
            {'error': "Duke AI couldn't answer right now. Try again later."},
            status=status.HTTP_502_BAD_GATEWAY,
        )

    return JsonResponse({'reply': reply})


@api_view(['POST'])
@parser_classes([JSONParser])
@permission_classes([IsAuthenticated])
@throttle_classes([PlanThrottle])
def plan(request):
    """POST /assistant/plan/

    Drafts the semesters the signed-in account has left until graduation with
    Duke AI (see planner.py). Only terms after the current one that the
    account has no semester for are planned; nothing is saved. The client
    applies a draft with POST /student/semesters/.

    All fields are optional: `max_credits` per term (12-18, default 15),
    `include_summer` (default false), `graduate_by` (the last term to plan,
    default six years out) and free-text `preferences`.

    Returns {"semesters": [{"term", "year", "credits", "courses": [{"code",
    "title", "credits", ..., "fills": [...]}]}], "notes": "...", "warnings":
    [courses or terms left out and why], "unsatisfied": [what the plan still
    misses]}. Returns a 400 for bad options or no terms to plan, a 409 with
    `issues` if the account isn't valid, and 429, 502 and 503 as chat does.

    Sample request body:
    {"max_credits": 15, "include_summer": false, "graduate_by": {"term": "spring", "year": 2029},
     "preferences": "Lighter fall semesters; I'm interested in machine learning."}
    """
    try:
        options = planner.parse_options(request.data)
    except planner.PlanError as e:
        return JsonResponse({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
    if not settings.GROQ_API_KEY:
        return JsonResponse(
            {'error': 'Duke AI is not configured.'}, status=status.HTTP_503_SERVICE_UNAVAILABLE
        )

    try:
        data = planner.generate(request.user, options)
    except planner.InvalidAccount as e:
        return JsonResponse(
            {
                'error': 'Fix the issues on your account before generating a plan.',
                'issues': [
                    {'code': i.code, 'message': i.message, 'severity': i.severity.value}
                    for i in e.issues
                ],
            },
            status=status.HTTP_409_CONFLICT,
        )
    except planner.PlanError as e:
        return JsonResponse({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
    except groq.RateLimitError:
        return JsonResponse(
            {'error': 'Duke AI is busy right now. Try again in a minute.'},
            status=status.HTTP_429_TOO_MANY_REQUESTS,
        )
    except groq.APIError:
        return JsonResponse(
            {'error': "Duke AI couldn't draft a plan right now. Try again later."},
            status=status.HTTP_502_BAD_GATEWAY,
        )

    return JsonResponse(data)
