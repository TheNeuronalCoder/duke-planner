<p align="center">
  <img src="https://menelik.xyz/images/uploads/duke-ai-icon.png" alt="Duke AI Planner" width="88">
</p>

<h1 align="center">Duke AI Planner</h1>

<p align="center">
  <b>Plan your JMU semester with Duke AI</b>
</p>

<p align="center">
  <a href="https://dukeaiplanner.web.app/"><img src="https://img.shields.io/badge/Launch_App-450084?style=for-the-badge&logo=firebase&logoColor=white" alt="Launch App"></a>
  <a href="https://api.menelik.xyz/docs/dukeplanner/"><img src="https://img.shields.io/badge/API_Docs-CBB677?style=for-the-badge&logo=readthedocs&logoColor=black" alt="API Docs"></a>
</p>

<p align="center">
  <img src="https://menelik.xyz/images/uploads/Duke-Planner-11-01-2026_02_03_PM.png" alt="Duke AI Planner screenshot" width="90%">
</p>

Django backend for AI academic planner made for JMU students that drafts graduation plans and answers degree questions through an LLM chat assistant

## Features

- **Course catalog**: about 4,900 JMU courses and 280 programs (catalog year 2026–2027). Courses can be looked up by code or found by meaning with semantic search.
- **Requirement tracking**: each requirement of the student's majors and minors, General Education and the B.A./B.S. degree is marked satisfied, in progress, planned or unsatisfied, based on the courses in their semesters.
- **Account validation**: accounts are checked against JMU policy, such as program combinations and credit limits. The rules are in [account/data/valid_account.yaml](account/data/valid_account.yaml).
- **Duke AI chat**: an LLM assistant served through Groq. It calls tools to search the catalog and read the student's requirements and semesters, so the whole plan isn't packed into its prompt.
- **AI graduation plans**: Duke AI drafts the remaining semesters. The server checks the draft (unknown or repeated courses, credit caps, terms that aren't open, unmet requirements) and sends the problems back to the model for one revision. Nothing is saved until the client applies the draft.
- **Passwordless sign-in**: a one-time code is emailed to `<eid>@dukes.jmu.edu` and exchanged for JWTs.

## Tech stack

| | |
|---|---|
| Framework | Django 4.2, Django REST Framework, Simple JWT |
| Database | PostgreSQL with [pgvector](https://github.com/pgvector/pgvector) (HNSW index on course embeddings) |
| Embeddings | `BAAI/bge-small-en-v1.5` via [fastembed](https://github.com/qdrant/fastembed) (ONNX Runtime) |
| LLM | [Groq](https://groq.com/), `openai/gpt-oss-120b` by default |
| Serving | Gunicorn + WhiteNoise in Docker, deployed with [Dokploy](https://dokploy.com/) on AWS EC2 behind [Traefik](https://github.com/traefik/traefik) |

## Project layout

```
duke_planner/   Settings, root URLs, and the optional SSH tunnel to the production DB
account/        Sign-in codes, JWTs, onboarding, account deletion, and account validation
catalog/        Courses, programs, and requirements; search and embeddings
  data/         Gen Ed and B.A./B.S. degree requirement definitions (YAML)
  fixtures/     Catalog data loaded on every start
  management/   load_catalog and embed_courses commands
student/        The signed-in student's programs, semesters, classes, and requirement status
assistant/      Duke AI: chat, tool definitions, plan context, and the graduation planner
openapi.yaml    OpenAPI 3.1 spec for the API
```

## API overview

The base URL in production is `https://api.menelik.xyz/dukeplanner`. The full reference, with request and response schemas, is at **[api.menelik.xyz/docs/dukeplanner](https://api.menelik.xyz/docs/dukeplanner/)** and in [openapi.yaml](openapi.yaml).

| Area | Endpoints | Auth |
|---|---|---|
| Catalog | `GET /catalog/courses`, `GET /catalog/search/course`, `GET /catalog/search/program` | Public |
| Auth | `POST /auth/request-code/`, `/auth/verify-code/`, `/auth/refresh/`, `/auth/logout/` | Public |
| Account | `POST /account/setup/`, `DELETE /account/delete/` | Bearer |
| Student | `/student/courses/`, `/student/programs/`, `/student/semesters/`, `GET /student/requirements/` | Bearer |
| Assistant | `POST /assistant/chat/`, `POST /assistant/plan/` | Bearer |

Requests are authenticated with `Authorization: Bearer <access token>`. Access tokens last 60 minutes and refresh tokens last 30 days; refresh tokens rotate each time they are used. To protect the shared Groq quota, the assistant is rate limited to 20 chat requests per minute and 5 plans per hour for each user.
