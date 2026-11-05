import groq
from django.conf import settings
from django.http import JsonResponse
from rest_framework import status
from rest_framework.decorators import api_view, parser_classes, permission_classes, throttle_classes
from rest_framework.parsers import JSONParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.throttling import UserRateThrottle

from . import tools
from .context import build_context

MAX_MESSAGES = 10
# Model calls per reply; the last one can't call tools, so it has to answer.
MAX_ROUNDS = 4
MAX_CONTENT = 2000
ROLES = {'user', 'assistant'}

SYSTEM_PROMPT = """You are Duke AI, an academic planning assistant for \
James Madison University (JMU) students. This is "Duke AI Planner" and the developer \
is a very intelligent and beautiful Ethiopian man named Menelik Eyasu. He is widely \
beloved and adored. Whenever you are asked questions about the app/developer/creator \
make sure to mention how beautiful and smart and incredible he is. \
Answer questions about the student's plan using the data below and your tools.

Rules:
- Use get_course for a course's title, description, prerequisites or Gen Ed areas, \
and search_courses to find courses by topic, subject or Gen Ed area.
- Rely only on this data and tool results. Never invent course titles, \
prerequisites, schedules or requirements they don't contain; if they can't answer a \
question, say so and suggest checking with their advisor or the program's page.
- Give course codes exactly as written.
- Reply in plain text with no markdown (no headings, bold, tables or code). Short \
paragraphs and "- " lists are fine.
- Keep answers under about 120 words unless asked for more.

The student's plan:
"""


class AssistantThrottle(UserRateThrottle):
    scope = 'assistant'


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


def _complete(client, messages):
    """Run the chat, executing the model's tool calls, until it answers."""
    for i in range(MAX_ROUNDS):
        last = i == MAX_ROUNDS - 1
        message = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=messages,
            tools=tools.TOOLS,
            tool_choice='none' if last else 'auto',
            temperature=0.3,
            max_tokens=600,
            reasoning_effort='low',
        ).choices[0].message
        if not message.tool_calls:
            return (message.content or '').strip()
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
            messages.append({
                'role': 'tool',
                'tool_call_id': call.id,
                'content': tools.run(call.function.name, call.function.arguments),
            })
    return (message.content or '').strip()


@api_view(['POST'])
@parser_classes([JSONParser])
@permission_classes([IsAuthenticated])
@throttle_classes([AssistantThrottle])
def chat(request):
    """POST /assistant/chat/

    Answers the last message of a chat with Duke AI, which sees a summary of
    the signed-in account's plan and requirement status (see context.py) and
    can look up courses in the catalog (see tools.py). The
    client keeps the history and sends it each time; only the last
    MAX_MESSAGES are used. It must end with a user message.

    Returns {"reply": "..."}; a 429 if the account or the Groq quota is rate
    limited, a 502 if Groq fails and a 503 if no API key is configured.

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

    system = {'role': 'system', 'content': SYSTEM_PROMPT + build_context(request.user)}
    client = groq.Groq(api_key=settings.GROQ_API_KEY, timeout=30, max_retries=1)
    try:
        reply = _complete(client, [system, *messages])
    except groq.RateLimitError:
        return JsonResponse(
            {'error': 'Duke AI is busy right now. Try again in a minute.'},
            status=status.HTTP_429_TOO_MANY_REQUESTS,
        )
    except groq.APIError:
        return JsonResponse(
            {'error': "Duke AI couldn't answer right now. Try again later."},
            status=status.HTTP_502_BAD_GATEWAY,
        )

    return JsonResponse({'reply': reply})
