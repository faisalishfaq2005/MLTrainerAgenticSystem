# LLM Environment And Key Handling Plan

## Current Stage (Now)
You are building agents first and using fallback LLM behavior.

### Where to put .env now
Create one `.env` file in project root:
- [d:/Projects/MLTrainerAgenticSystem](d:/Projects/MLTrainerAgenticSystem)

Example:

```env
GROQ_API_KEY=gsk_your_real_key_here
```

### Important behavior of current router
Current router reads **process environment variables** using `os.getenv(...)`.
It does not parse `.env` by itself.

So you must do one of these:
1. Export env vars in terminal before running app/tests.
2. Load `.env` in app entrypoint (for example with `python-dotenv`) before creating `LLMRouter`.

If `GROQ_API_KEY` is available in process env, fallback to Groq works.

## How to run now (quick)
From project root:

```bat
set GROQ_API_KEY=gsk_your_real_key_here
python -m tests.unit.intake_manager_test
```

Alternative (later): install and use `python-dotenv` in startup path.

## Multi-User Backend Design (Later)
Do **not** create one `.env` per user.

### Recommended design
1. Keep only system fallback keys in global env (example: `GROQ_API_KEY`).
2. Store user-provided provider keys in DB/secret manager (encrypted at rest).
3. At request time, fetch and decrypt that user key.
4. Pass key directly to router instance for that user/job.
5. Never persist raw user keys in logs or conversation history.

### Why not per-user .env
1. Hard to manage at scale.
2. Unsafe for concurrency.
3. High chance of leakage/misrouting.
4. Operationally difficult for rotation and revocation.

## Concurrency Note (Important)
Your current router sets environment variables in-process when key is provided.
In a multi-user backend with concurrent jobs, this can leak/override keys between users.

### Refactor planned for backend phase
1. Prefer passing `api_key` per request/router instance only.
2. Avoid mutating `os.environ` with user keys.
3. Keep env mutation only for static server-level fallback keys if needed.

## Implementation Checklist For Backend Phase
1. Add secure key storage table/secret manager integration.
2. Add encryption/decryption service for user API keys.
3. Update orchestrator key loading per job/user.
4. Refactor router to avoid global env mutation for user-scoped keys.
5. Add request/job-scoped audit logs (provider + model only, no secrets).
6. Add key rotation support in secret store (not in source code).

## Bottom Line
- For now: one root `.env` with `GROQ_API_KEY` is fine.
- For backend multi-user: user keys should come from secure per-user storage, not `.env` files.
