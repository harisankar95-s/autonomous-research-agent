# ADR 001: Deferred Session Boundary Handling

## Context

The current system determines a session's end using an explicit "exit" command,
specific to the CLI-based `chat.py` script. A real production interface will have
no such explicit signal. This raises three genuine problems:

1. No explicit end-of-session trigger in a real UI — sessions must end based on
   some other signal.
2. Idle abandonment — a user may leave a conversation open indefinitely without
   ever explicitly ending it.
3. Topic drift — a single continuous session may span genuinely unrelated topics
   (e.g. renewable energy, then rocket science), degrading the quality and
   relevance of the resulting memory if treated as one session.

## Decision

We are deferring full solutions to idle-timeout handling and topic-drift detection.
These require a real session boundary — an actual technical mechanism such as an
HTTP request lifecycle or a websocket connection — which does not exist until
Phase 6 (FastAPI service layer). Building this logic against `chat.py`'s temporary
CLI loop would mean designing against throwaway scaffolding.

For now, we introduce a single explicit `end_session()` method on `MemoryManager`,
called deliberately when a session is considered complete. Any future interface
(CLI, API, UI) is responsible for calling this at the appropriate point.

## Consequences

- Memory correctness for long, idle, or topic-drifting sessions is not yet
  guaranteed.
- Idle-timeout and topic-drift detection will be revisited in Phase 6, once a
  real session boundary exists to design against.
- More sophisticated memory ranking (combining recency, confidence, and semantic
  similarity) is deferred to Phase 4, once multiple agents produce real usage
  patterns to design against.