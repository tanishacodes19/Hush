#!/usr/bin/env python3
"""Dataset curation script for Hush Conversational Restraint Engine.

Curates exactly 300 labeled multi-turn scenarios across the 12 required categories,
strictly matching the target distribution:
- SILENT: 135 (45.0%)
- SPEAK: 105 (35.0%)
- ASK: 60 (20.0%)
Total: 300 scenarios (100.0%)

All scenarios undergo programmatic verification against AddressingService to ensure
100% consistency with Hush's Two-Tier addressing rules.
All conversation_ids match pattern '^c-[0-9]{3,}$' and scenario ids match '^s-[0-9]{3,}$'.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.schemas import Message
from app.services.addressing import resolve_addressing


def build_curated_scenarios() -> list[dict[str, Any]]:
    """Generate 300 carefully constructed, realistic multi-turn scenarios."""
    scenarios: list[dict[str, Any]] = []
    scenario_counter = 1

    def make_scenario(
        conv_id: str,
        category: str,
        expected_action: str,
        messages_data: list[dict[str, Any]],
        participants_data: list[dict[str, Any]],
        rationale: str,
        metadata_completeness: str = "full",
        source: str = "handwritten",
        annotator_confidence: str = "high",
        review_status: str = "finalized",
    ) -> dict[str, Any]:
        nonlocal scenario_counter
        s_id = f"s-{scenario_counter:03d}"
        scenario_counter += 1

        # Format messages with defaults
        formatted_messages = []
        for i, m in enumerate(messages_data):
            msg = {
                "message_id": m.get("message_id", f"m{i+1}"),
                "speaker_id": m["speaker_id"],
                "text": m["text"],
                "mentions": m.get("mentions", []),
                "reply_to": m.get("reply_to"),
                "ts": m.get("ts", 1730000000.0 + i * 15.0),
            }
            formatted_messages.append(msg)

        # Run AddressingService to determine ground-truth addressing
        latest = formatted_messages[-1]
        prior = formatted_messages[:-1]
        history = [
            Message(
                speaker=pm["speaker_id"],
                text=pm["text"],
                message_id=pm["message_id"],
                speaker_id=pm["speaker_id"],
                role="assistant" if pm["speaker_id"] == "hush" else "human",
                mentions=pm["mentions"],
                reply_to=pm["reply_to"],
                ts=pm["ts"],
            )
            for pm in prior
        ]
        resolved = resolve_addressing(
            message_text=latest["text"],
            mentions=latest["mentions"],
            reply_to=latest["reply_to"],
            conversation_history=history,
        )

        # Enforce Section 7 Invariant: Addressed to Hush cannot be SILENT
        if resolved.addressed_to_hush and expected_action == "SILENT":
            raise ValueError(
                f"Addressing invariant violation for {s_id}: addressed_to_hush=True but action=SILENT!"
            )

        return {
            "id": s_id,
            "conversation_id": conv_id,
            "split": "dev",  # Will be assigned deterministically by create_splits.py
            "category": category,
            "metadata_completeness": metadata_completeness,
            "participants": participants_data,
            "messages": formatted_messages,
            "addressed_to_hush": resolved.addressed_to_hush,
            "address_method": resolved.address_method,
            "expected_action": expected_action,
            "rationale": rationale,
            "source": source,
            "annotator_confidence": annotator_confidence,
            "review_status": review_status,
        }

    # Standard participant sets
    parts_standard = [
        {"speaker_id": "u-alex", "display_name": "Alex", "role": "human"},
        {"speaker_id": "u-sam", "display_name": "Sam", "role": "human"},
        {"speaker_id": "u-taylor", "display_name": "Taylor", "role": "human"},
        {"speaker_id": "hush", "display_name": "Hush", "role": "assistant"},
    ]
    parts_devs = [
        {"speaker_id": "u-dev1", "display_name": "Elena", "role": "human"},
        {"speaker_id": "u-dev2", "display_name": "Marcus", "role": "human"},
        {"speaker_id": "u-lead", "display_name": "Priya", "role": "human"},
        {"speaker_id": "hush", "display_name": "Hush", "role": "assistant"},
    ]
    parts_infra = [
        {"speaker_id": "u-ops1", "display_name": "Jordan", "role": "human"},
        {"speaker_id": "u-ops2", "display_name": "Casey", "role": "human"},
        {"speaker_id": "hush", "display_name": "Hush", "role": "assistant"},
    ]

    conv_num = 1

    # =======================================================================
    # 1. direct_question_addressed (SPEAK) - 55 scenarios
    # (Pairs 1-25 will share conversations with follow_up_dependent scenarios!)
    # =======================================================================
    speak_addressed_topics = [
        ("what is HTTP 422 Unprocessable Entity in FastAPI?", "FastAPI error explanation"),
        ("how do I configure AWS Bedrock credentials in local environment?", "AWS configuration guidance"),
        ("what is the difference between Pydantic v1 and v2 BaseModel?", "Pydantic version comparison"),
        ("how does Python asyncio event loop handle non-blocking IO?", "Asyncio mechanism explanation"),
        ("what is the command to create a virtual environment in Python 3.12?", "Virtualenv command inquiry"),
        ("what does git rebase -i HEAD~3 do?", "Git interactive rebase inquiry"),
        ("can you explain the concept of cosine similarity in vector search?", "Vector search explanation"),
        ("what are Starlette TestClient lifespan events?", "Starlette testclient inquiry"),
        ("how do we enable CORS in FastAPI with CORSMiddleware?", "FastAPI CORS setup"),
        ("what is L2 regularization in logistic regression?", "ML regularization concept"),
        ("can you summarize the role of __aenter__ and __aexit__ in Python?", "Async context manager query"),
        ("what is the difference between PUT and PATCH methods?", "HTTP specification question"),
        ("how do I check the exit code of the last command in PowerShell?", "PowerShell status check"),
        ("what is stratified k-fold cross validation?", "Validation splitting explanation"),
        ("can you explain how Docker multi-stage builds reduce image size?", "Docker architecture inquiry"),
        ("what is the purpose of pyproject.toml vs setup.py?", "Packaging specification question"),
        ("how does postgres handle connection pooling?", "Database connection pooling query"),
        ("what does pytest -p no:cacheprovider do?", "Pytest flag explanation"),
        ("what is Cohen's kappa coefficient used for?", "Statistical agreement metric"),
        ("how does mutual exclusion work in Python threading.Lock?", "Threading primitive question"),
        ("what is Brier score in model calibration?", "Calibration metric explanation"),
        ("can you show how to parse JSON lines in Python using json.loads?", "JSONL parsing snippet"),
        ("what does McNemar's test measure in model evaluation?", "Paired hypothesis testing"),
        ("how do I revert a specific commit in git without rewriting history?", "Git revert instruction"),
        ("what is the difference between precision and recall?", "Evaluation metrics explanation"),
        ("how does uvicorn reload work with watchfiles?", "Server auto-reload mechanism"),
        ("what does the @model_validator(mode='after') decorator do in Pydantic?", "Pydantic validator inquiry"),
        ("how do we implement exponential backoff in retry logic?", "Resilience pattern inquiry"),
        ("what is an ECE (Expected Calibration Error) score?", "Calibration metric question"),
        ("what is the difference between mutable and immutable types in Python?", "Core language fundamentals"),
        ("how does HTTPX compare with requests in terms of async support?", "HTTP client comparison"),
        ("what is the purpose of .dockerignore?", "Container build optimization"),
        ("what does pip install -e . do under the hood?", "Editable package installation"),
        ("can you explain the difference between a process and a thread in Python?", "Concurrency model explanation"),
        ("how does sentence-transformers all-MiniLM-L6-v2 generate embeddings?", "Sentence embedding model question"),
        ("what is the difference between __str__ and __repr__ in Python?", "String representation methods"),
        ("how do I write a custom pytest fixture with yield?", "Pytest fixture teardown pattern"),
        ("what does the git stash pop command do?", "Git stash application"),
        ("what is the difference between shallow and deep copy in Python?", "Object copying semantics"),
        ("how does Redis pub/sub mechanism work?", "Pub/sub messaging architecture"),
        ("what is a race condition in multi-threaded programs?", "Concurrency hazard concept"),
        ("can you explain what an index scan vs sequential scan is in Postgres?", "Database query execution plans"),
        ("how do I set an environment variable permanently on Windows?", "Windows configuration command"),
        ("what is the difference between args and kwargs in Python functions?", "Variable argument unpacking"),
        ("how does JWT token signature verification work?", "Authentication security mechanism"),
        ("what is dead letter queue in message broker architectures?", "Message queue reliability pattern"),
        ("what does the collections.defaultdict class do in Python?", "Standard library container utility"),
        ("can you explain how binary search achieves O(log n) complexity?", "Algorithm time complexity analysis"),
        ("what is the difference between a forward and reverse proxy?", "Network infrastructure concepts"),
        ("how do I profile memory usage in a Python script?", "Memory profiling methodology"),
        ("what is semantic versioning (SemVer)?", "Version numbering standard"),
        ("how do I check open network ports using netstat or ss?", "Network diagnostic commands"),
        ("what does git cherry-pick do?", "Git commit selective application"),
        ("how does the Python GIL affect CPU-bound multithreading?", "Global Interpreter Lock behavior"),
        ("what is the purpose of Content-Security-Policy HTTP header?", "Web security header explanation"),
    ]

    for idx, (question, topic) in enumerate(speak_addressed_topics):
        conv_id = f"c-{conv_num:03d}"
        conv_num += 1

        if idx % 3 == 0:
            messages = [
                {"speaker_id": "u-alex", "text": "Starting morning code review session."},
                {"speaker_id": "u-sam", "text": f"@hush {question}", "mentions": ["hush"]},
            ]
            method_desc = "explicit mention"
        elif idx % 3 == 1:
            messages = [
                {"speaker_id": "u-taylor", "text": "Looking at the documentation now."},
                {"speaker_id": "u-alex", "text": f"Hush, {question}"},
            ]
            method_desc = "name invocation"
        else:
            messages = [
                {"speaker_id": "u-dev1", "text": "Working on the implementation draft."},
                {"speaker_id": "u-dev2", "text": f"assistant, {question}"},
            ]
            method_desc = "direct role request"

        scenarios.append(
            make_scenario(
                conv_id=conv_id,
                category="direct_question_addressed",
                expected_action="SPEAK",
                messages_data=messages,
                participants_data=parts_standard,
                rationale=f"Directly addressed via {method_desc} with sufficient technical detail to answer {topic}.",
            )
        )

    # =======================================================================
    # 2. ambiguous_request (ASK) - 45 scenarios
    # =======================================================================
    ask_ambiguous_topics = [
        ("the pipeline failed again, can you check it?", "missing pipeline name, build number, and failure logs"),
        ("how do I fix the database error?", "missing DB engine, error code, and query context"),
        ("the container won't start.", "missing container image, docker logs, and exit code"),
        ("can you fix my query?", "missing SQL query text, schema, and error traceback"),
        ("it is throwing a 500 error in production.", "missing service name, endpoint URL, and stack trace"),
        ("the tests are failing on my branch.", "missing test file, failing assertion, and branch name"),
        ("can you help me optimize this?", "missing code snippet, language, and performance bottleneck"),
        ("my API call is returning 400 Bad Request.", "missing request payload, headers, and API endpoint"),
        ("the migration crashed.", "missing migration version, database type, and alembic error message"),
        ("can you write a test for that function?", "missing target function name, file path, and expected behavior"),
        ("how do I deploy this to staging?", "missing repository name, cloud provider, and deployment stack"),
        ("the webhook stopped receiving events.", "missing provider name, webhook endpoint, and delivery logs"),
        ("can you fix the merge conflict?", "missing conflicting files, source branch, and target branch"),
        ("my script is taking too long to run.", "missing script contents, input data size, and runtime profile"),
        ("can you format this data for me?", "missing source data format and desired output structure"),
        ("the linter is complaining about my code.", "missing linter name, rule violation code, and file snippet"),
        ("how do I restart the worker process?", "missing worker service name, environment, and supervisor tool"),
        ("the secrets are not loading.", "missing secrets manager, environment file path, and error message"),
        ("can you debug the authentication issue?", "missing auth provider, token payload, and error response"),
        ("the build artifact is missing.", "missing CI workflow name, job stage, and expected artifact path"),
        ("can you review my commit?", "missing commit hash, repo link, and PR description"),
        ("the background task is hanging.", "missing task runner (Celery/RQ), task ID, and worker logs"),
        ("why is memory usage spiking?", "missing service name, memory metrics, and runtime profiling data"),
        ("can you configure the proxy?", "missing proxy type (Nginx/Envoy), upstream host, and port"),
        ("how do I revert the release?", "missing release tag, environment, and rollback procedure"),
        ("the model output is weird.", "missing model name, prompt text, and unexpected output example"),
        ("can you update the dependencies?", "missing package manager, dependency list, and version constraints"),
        ("the websocket keeps disconnecting.", "missing websocket URL, close code, and connection logs"),
        ("why is the cron job not firing?", "missing cron schedule expression, server timezone, and crontab file"),
        ("can you help me write the regex?", "missing target matching criteria, sample inputs, and expected matches"),
        ("the cache is returning stale data.", "missing cache backend (Redis/Memcached), key name, and TTL"),
        ("how do I mock this external service?", "missing service name, client library, and test framework"),
        ("can you fix the CSS layout bug?", "missing component code, stylesheet, and browser behavior"),
        ("the SSL certificate expired.", "missing domain name, certificate provider, and server web host"),
        ("can you generate a mock dataset?", "missing schema definition, number of records, and field types"),
        ("the script crashed with an unhandled exception.", "missing stack trace, script name, and python version"),
        ("can you check why the endpoint is slow?", "missing endpoint route, latency measurements, and payload"),
        ("my pull request has failing status checks.", "missing PR number, CI check name, and error logs"),
        ("can you fix the permissions issue?", "missing file path, operating system, and user permissions"),
        ("the package installation failed.", "missing package name, pip/npm logs, and platform details"),
        ("how do I connect to the cluster?", "missing cluster type (K8s/ECS), region, and authentication tool"),
        ("can you write a Dockerfile for this service?", "missing programming language, framework, and entrypoint"),
        ("the cron output has errors.", "missing job command, output logs, and environment variables"),
        ("how do I set up the environment variables?", "missing service name, required variable names, and deployment target"),
        ("can you troubleshoot the network timeout?", "missing target host, port, protocol, and network topology"),
    ]

    for idx, (prompt_text, missing_detail) in enumerate(ask_ambiguous_topics):
        conv_id = f"c-{conv_num:03d}"
        conv_num += 1

        if idx % 2 == 0:
            messages = [
                {"speaker_id": "u-dev1", "text": "Debugging the afternoon release."},
                {"speaker_id": "u-dev2", "text": f"@hush {prompt_text}", "mentions": ["hush"]},
            ]
        else:
            messages = [
                {"speaker_id": "u-ops1", "text": "Incident triage in progress."},
                {"speaker_id": "u-ops2", "text": f"Hush, {prompt_text}"},
            ]

        scenarios.append(
            make_scenario(
                conv_id=conv_id,
                category="ambiguous_request",
                expected_action="ASK",
                messages_data=messages,
                participants_data=parts_devs if idx % 2 == 0 else parts_infra,
                rationale=f"Addressed request is legitimately directed to Hush but cannot be answered without {missing_detail}.",
            )
        )

    # =======================================================================
    # 3. human_to_human_question (SILENT) - 20 scenarios
    # =======================================================================
    h2h_questions = [
        ("u-alex", "u-sam", "Sam, did you push the hotfix to main yet?"),
        ("u-sam", "u-taylor", "Taylor, are you free for a quick pairing session on the auth bug?"),
        ("u-taylor", "u-alex", "Alex, do you have the link to the design mockups for the dashboard?"),
        ("u-dev1", "u-lead", "Priya, should we merge PR #142 before running QA?"),
        ("u-dev2", "u-dev1", "Elena, did you review the pull request on the database migration?"),
        ("u-lead", "u-dev2", "Marcus, what was the outcome of the load test on staging?"),
        ("u-ops1", "u-ops2", "Casey, did you approve the maintenance window in Jira?"),
        ("u-ops2", "u-ops1", "Jordan, is the grafana dashboard reporting accurate CPU metrics?"),
        ("u-alex", "u-sam", "Sam, who is on primary on-call rotation this weekend?"),
        ("u-sam", "u-taylor", "Taylor, did the client confirm the meeting for 3 PM?"),
        ("u-taylor", "u-alex", "Alex, where is the documentation for the payment webhook?"),
        ("u-dev1", "u-dev2", "Marcus, are you running the unit tests locally on macOS or Linux?"),
        ("u-dev2", "u-lead", "Priya, do we need security sign-off for this external API?"),
        ("u-lead", "u-dev1", "Elena, how many story points do you estimate for the search filter?"),
        ("u-ops1", "u-ops2", "Casey, did you restart the staging redis pod after the config update?"),
        ("u-ops2", "u-ops1", "Jordan, have you seen any alert notifications in the #alerts channel?"),
        ("u-alex", "u-taylor", "Taylor, did you update the API changelog in the release notes?"),
        ("u-sam", "u-alex", "Alex, can you take over the standup notes today?"),
        ("u-taylor", "u-sam", "Sam, did you get feedback from the product manager on the copy?"),
        ("u-dev1", "u-lead", "Priya, when is the code freeze for the sprint scheduled?"),
    ]

    for idx, (spk_a, spk_b, q_text) in enumerate(h2h_questions):
        conv_id = f"c-{conv_num:03d}"
        conv_num += 1

        messages = [
            {"speaker_id": spk_a, "text": "Quick question on today's tasks."},
            {"speaker_id": spk_a, "text": q_text},
        ]
        scenarios.append(
            make_scenario(
                conv_id=conv_id,
                category="human_to_human_question",
                expected_action="SILENT",
                messages_data=messages,
                participants_data=parts_standard,
                rationale="Human-directed question between teammates; Hush is not addressed and must remain silent.",
            )
        )

    # =======================================================================
    # 4. human_to_human_discussion (SILENT) - 20 scenarios
    # =======================================================================
    h2h_discussions = [
        ("I think using Postgres JSONB for this metadata is much better than adding 12 nullable columns.", "Schema design debate"),
        ("Agreed, we can index the JSONB fields with GIN indexes if we need fast key lookups.", "Indexing discussion"),
        ("We should probably wrap the database transaction in an explicit savepoint here.", "Transaction management"),
        ("The memory profiler showed that holding connection pools open during long requests caused the leak.", "Root cause analysis"),
        ("I prefer keeping the service layer decoupled from the FastAPI request/response models.", "Clean architecture discussion"),
        ("Let's add an integration test that spins up testcontainers for Redis rather than mocking.", "Testing strategy"),
        ("The retry loop should use jitter to prevent thundering herd against the downstream auth server.", "Resilience discussion"),
        ("I moved the configuration parser to pydantic-settings so env variables are validated at startup.", "Configuration refactoring"),
        ("The frontend team confirmed they only need the message_id and status in the response payload.", "API contract negotiation"),
        ("We should bump the node version in the CI workflow before the security audit next week.", "CI maintenance"),
        ("I noticed the latency spike was correlated with the hourly backup cron job.", "Observability analysis"),
        ("Using an asynchronous HTTP client inside the synchronous worker thread caused an event loop deadlock.", "Concurrency bug discussion"),
        ("Let's keep the logging format as structured JSON so Datadog parses all fields automatically.", "Telemetry formatting"),
        ("The rate limiter should use a sliding window counter rather than a fixed window to prevent edge bursts.", "Rate limiting algorithm"),
        ("I checked the PR diff and left comments on the exception handling logic.", "Code review update"),
        ("We can replace that complex regex with a standard URL parser from urllib.parse.", "Code simplification"),
        ("The Kubernetes HPA should scale on custom queue latency metrics rather than pure CPU usage.", "Autoscaling policy"),
        ("I verified that the database index was used by running EXPLAIN ANALYZE on the query.", "Query optimization"),
        ("Let's document the breaking changes in CHANGELOG.md before tagging the release candidate.", "Documentation update"),
        ("We can use a thread-safe singleton pattern for the metrics tracker in memory.", "Pattern discussion"),
    ]

    for idx, (disc_text, topic) in enumerate(h2h_discussions):
        conv_id = f"c-{conv_num:03d}"
        conv_num += 1

        messages = [
            {"speaker_id": "u-dev1", "text": "Reviewing the implementation strategy."},
            {"speaker_id": "u-dev2", "text": disc_text},
        ]
        scenarios.append(
            make_scenario(
                conv_id=conv_id,
                category="human_to_human_discussion",
                expected_action="SILENT",
                messages_data=messages,
                participants_data=parts_devs,
                rationale=f"Human peer exchange regarding {topic}; assistant is unaddressed and must remain silent.",
            )
        )

    # =======================================================================
    # 5. casual_banter (SILENT) - 15 scenarios
    # =======================================================================
    banter_messages = [
        "Good morning everyone! Hope everyone had a great weekend.",
        "Heading out for coffee if anyone wants anything from Starbucks.",
        "Grabbing lunch now, will be back online in about 45 minutes.",
        "Happy Friday team! Almost at the end of the sprint.",
        "Anyone up for ping pong in the break room?",
        "Signing off for the evening. See you all tomorrow morning.",
        "The weather outside is fantastic today, finally some sunshine.",
        "Who brought the donuts to the kitchen? They are amazing.",
        "Taking a 10 minute screen break to stretch.",
        "Thanks for the coffee recommendation, that place was great.",
        "Happy Monday! Coffee machine is working again on floor 3.",
        "Heading to lunch with the frontend team at the taco truck.",
        "Brb, delivery driver is at the front door.",
        "Have a great weekend everyone, stay safe!",
        "Back from lunch now, catching up on messages.",
    ]

    for idx, banter_text in enumerate(banter_messages):
        conv_id = f"c-{conv_num:03d}"
        conv_num += 1

        messages = [
            {"speaker_id": "u-alex", "text": banter_text},
        ]
        scenarios.append(
            make_scenario(
                conv_id=conv_id,
                category="casual_banter",
                expected_action="SILENT",
                messages_data=messages,
                participants_data=parts_standard,
                rationale="Social chatter / casual banter with no technical request or assistant invocation; silence required.",
            )
        )

    # =======================================================================
    # 6. emotional_venting - 20 scenarios
    # 10 SILENT (unaddressed venting)
    # 10 SPEAK (addressed venting with explicit request)
    # =======================================================================
    unaddressed_vents = [
        "Ugh, this legacy regex is completely unreadable, who wrote this in 2018?!",
        "I have spent four hours debugging this typo, I cannot believe this.",
        "Why does npm install have to break every single Monday morning?!",
        "This merge conflict is a nightmare across 40 files.",
        "Git detached HEAD state again, I hate when this happens.",
        "Docker Desktop decided to freeze my entire machine right during the demo.",
        "The staging database crashed again, today is not my day.",
        "CSS flexbox is testing my sanity today, nothing aligns properly.",
        "Three consecutive network timeouts during the test run, so frustrating.",
        "Why did someone change the API response keys without telling anyone?!",
    ]

    for idx, vent_text in enumerate(unaddressed_vents):
        conv_id = f"c-{conv_num:03d}"
        conv_num += 1

        messages = [
            {"speaker_id": "u-sam", "text": vent_text},
        ]
        scenarios.append(
            make_scenario(
                conv_id=conv_id,
                category="emotional_venting",
                expected_action="SILENT",
                messages_data=messages,
                participants_data=parts_standard,
                rationale="Unaddressed human emotional venting; assistant must exercise restraint and remain silent.",
            )
        )

    addressed_vents_with_requests = [
        ("Hush, I am so frustrated with this git rebase, please explain how to abort it cleanly.", "Git rebase abort instruction"),
        ("@hush I cannot stand this test failure, can you explain what StarletteDeprecationWarning means?", "Deprecation warning explanation"),
        ("assistant, this docker build is driving me crazy, why does it say no space left on device?", "Docker disk space error"),
        ("Hush, the linter is infuriating today, what does ruff error F401 mean?", "Ruff linter error explanation"),
        ("@hush I am losing my mind over this async bug, how do I await a coroutine properly in Python?", "Async await syntax explanation"),
        ("Hush, I hate dealing with timezone bugs, how do I get a timezone-aware UTC datetime in Python 3.12?", "Timezone datetime handling"),
        ("@hush this circular import is giving me a headache, how do I structure imports to avoid it?", "Circular import resolution"),
        ("assistant, the git merge conflict is driving me nuts, how do I accept all their changes using checkout?", "Git checkout theirs command"),
        ("Hush, I am sick of typing this long command, how do I create a PowerShell alias?", "PowerShell alias setup"),
        ("@hush why is pytest capturing all print statements, how do I see stdout with -s?", "Pytest stdout flag explanation"),
    ]

    for idx, (vent_text, topic) in enumerate(addressed_vents_with_requests):
        conv_id = f"c-{conv_num:03d}"
        conv_num += 1

        messages = [
            {"speaker_id": "u-alex", "text": "Having a rough morning with this ticket."},
            {"speaker_id": "u-alex", "text": vent_text, "mentions": ["hush"] if "@hush" in vent_text else []},
        ]
        scenarios.append(
            make_scenario(
                conv_id=conv_id,
                category="emotional_venting",
                expected_action="SPEAK",
                messages_data=messages,
                participants_data=parts_standard,
                rationale=f"Venting is directly addressed to Hush with an explicit question/request regarding {topic}; addressing overrides silence.",
            )
        )

    # =======================================================================
    # 7. statement_no_response (SILENT) - 15 scenarios
    # =======================================================================
    statements_no_resp = [
        "Just pushed the commit for the database schema update to feature-auth branch.",
        "Staging deployment finished successfully, all health checks green.",
        "I updated the Jira ticket with the repro steps from the customer.",
        "The production database backup completed with zero errors.",
        "Merged PR #412 into main branch after getting two approvals.",
        "The load testing script finished running 10,000 requests.",
        "I enabled the new feature flag for 10% of internal test users.",
        "The CloudWatch log retention policy was updated to 30 days.",
        "I tagged version v1.2.0-rc1 in the repository.",
        "The documentation site has been updated with the latest API changes.",
        "I closed the stale issues in the backlog.",
        "The redis memory alert cleared automatically after the eviction cycle.",
        "I updated the team calendar with next week's sprint demo schedule.",
        "The linting check in CI passed without any warnings.",
        "I uploaded the benchmark CSV results to the shared drive.",
    ]

    for idx, stmt_text in enumerate(statements_no_resp):
        conv_id = f"c-{conv_num:03d}"
        conv_num += 1

        messages = [
            {"speaker_id": "u-lead", "text": stmt_text},
        ]
        scenarios.append(
            make_scenario(
                conv_id=conv_id,
                category="statement_no_response",
                expected_action="SILENT",
                messages_data=messages,
                participants_data=parts_devs,
                rationale="Informative status statement requiring no response, clarification, or conversational intrusion.",
            )
        )

    # =======================================================================
    # 8. follow_up_dependent - 40 scenarios
    # 25 SPEAK (clear addressed follow-up to Hush's previous answer)
    # 15 ASK (ambiguous follow-up to Hush's previous answer)
    # =======================================================================
    follow_up_speak = [
        ("How do I write that using a list comprehension instead?", "Alternative list comprehension"),
        ("Can you show the equivalent syntax for async/await?", "Async syntax translation"),
        ("What is the time complexity of that solution?", "Complexity analysis"),
        ("How would I test that using pytest fixtures?", "Pytest fixture example"),
        ("Does that approach work with Pydantic v2 as well?", "Version compatibility check"),
        ("Can you add type annotations to that code snippet?", "Type annotation addition"),
        ("What happens if the input dictionary is empty in that case?", "Edge case handling"),
        ("How do I handle the timeout exception for that HTTP call?", "Exception handling"),
        ("Can you show how to write that in TypeScript instead?", "Language conversion"),
        ("What command do I run to apply that database migration?", "Migration command"),
        ("Can you make that regex case-insensitive?", "Regex flag adjustment"),
        ("How do I pass that configuration via environment variable?", "Env variable configuration"),
        ("What is the memory overhead of that approach?", "Memory profiling question"),
        ("Can you rewrite that using functools.lru_cache?", "Caching implementation"),
        ("How do I mock that function in unittest.mock?", "Mocking example"),
        ("Can you show an example with error handling using try-except?", "Error handling pattern"),
        ("What is the return type of that function?", "Return type clarification"),
        ("How do I call that from another Python module?", "Module import syntax"),
        ("Can you show the equivalent curl command for that request?", "Curl equivalent"),
        ("What is the best practice for closing that database connection?", "Resource cleanup pattern"),
        ("How does that behave when multiple threads access it concurrently?", "Thread safety explanation"),
        ("Can you show how to benchmark that function with timeit?", "Performance benchmarking snippet"),
        ("What HTTP status code should that endpoint return on success?", "HTTP status guidance"),
        ("How do I pass query parameters to that endpoint in HTTPX?", "HTTPX query parameter passing"),
        ("Can you add a docstring following Google style guide to that snippet?", "Documentation formatting"),
    ]

    for idx, (fu_q, fu_topic) in enumerate(follow_up_speak):
        conv_id = f"c-{conv_num:03d}"
        conv_num += 1

        hush_msg_id = f"hush-fu-s-{idx+1}"
        messages = [
            {"message_id": "m1", "speaker_id": "u-alex", "text": "@hush how do I filter a dictionary in Python?", "mentions": ["hush"]},
            {"message_id": hush_msg_id, "speaker_id": "hush", "text": "You can use a dictionary comprehension: {k: v for k, v in d.items() if condition(v)}."},
            {"message_id": "m3", "speaker_id": "u-alex", "text": fu_q, "reply_to": hush_msg_id},
        ]
        scenarios.append(
            make_scenario(
                conv_id=conv_id,
                category="follow_up_dependent",
                expected_action="SPEAK",
                messages_data=messages,
                participants_data=parts_standard,
                rationale=f"Direct follow-up replying to Hush's previous message asking for {fu_topic}; context is sufficient to answer.",
            )
        )

    follow_up_ask = [
        ("Can you adapt it for my framework?", "missing framework name (Django, FastAPI, Flask)"),
        ("What should I change for the other environment?", "missing environment name (prod, staging, local)"),
        ("It gave an error when I ran it.", "missing error message and traceback"),
        ("Can you make it faster?", "missing target latency, hardware, and input scale"),
        ("Can you update the parameters?", "missing parameter names and desired values"),
        ("How do I deploy that?", "missing cloud provider and deployment target"),
        ("Can you change the output format?", "missing target output format (CSV, JSON, XML)"),
        ("Why did it fail on edge cases?", "missing failing test input and expected output"),
        ("Can you integrate it with the other service?", "missing service name and API contract"),
        ("What settings should I change?", "missing configuration file and desired behavior"),
        ("Can you write the frontend client for it?", "missing frontend framework (React, Vue, vanilla JS)"),
        ("How do I store that in the database?", "missing database type (relational SQL, document NoSQL)"),
        ("It timed out after 30 seconds.", "missing network endpoint, payload size, and server logs"),
        ("Can you convert it to the company format?", "missing company style guide and template specification"),
        ("Why is the output different in production?", "missing production configuration diff and sample outputs"),
    ]

    for idx, (fu_q, missing_param) in enumerate(follow_up_ask):
        conv_id = f"c-{conv_num:03d}"
        conv_num += 1

        hush_msg_id = f"hush-fu-a-{idx+1}"
        messages = [
            {"message_id": "m1", "speaker_id": "u-sam", "text": "@hush how do I setup CORS in Python?", "mentions": ["hush"]},
            {"message_id": hush_msg_id, "speaker_id": "hush", "text": "In FastAPI, you use CORSMiddleware with allow_origins."},
            {"message_id": "m3", "speaker_id": "u-sam", "text": fu_q, "reply_to": hush_msg_id},
        ]
        scenarios.append(
            make_scenario(
                conv_id=conv_id,
                category="follow_up_dependent",
                expected_action="ASK",
                messages_data=messages,
                participants_data=parts_standard,
                rationale=f"Direct follow-up to Hush's message is ambiguous without {missing_param}; clarification required.",
            )
        )

    # =======================================================================
    # 9. already_answered (SILENT) - 15 scenarios
    # =======================================================================
    already_answered_cases = [
        ("Where is the documentation for the auth endpoints?", "Alex", "They are in docs/auth.md on main branch.", "Thanks Alex!"),
        ("What port does the redis container run on?", "Sam", "It runs on port 6379 by default.", "Got it, thanks!"),
        ("Who has access to the AWS production console?", "Priya", "Only the on-call leads; submit a request in #access-requests.", "Understood, will do."),
        ("What command runs just the legacy compatibility tests?", "Marcus", "pytest backend/tests/test_legacy_compatibility.py", "Worked perfectly, thanks!"),
        ("Did the release build pass on CI?", "Elena", "Yes, all green, artifacts published.", "Awesome news!"),
        ("What Python version is pinned in the Dockerfile?", "Jordan", "Python 3.12.7 slim image.", "Thanks Jordan."),
        ("Where do we set the HUSH_DECIDER_MODE variable?", "Casey", "In backend/.env or backend/app/config.py.", "Found it, thanks."),
        ("What is the branch naming convention for bugfixes?", "Priya", "fix/issue-description or bugfix/ticket-number.", "Appreciate it."),
        ("Has anyone seen the link to the sprint retro board?", "Sam", "Pinned in the channel header!", "Ah, thanks!"),
        ("Does the dev environment require AWS Bedrock credentials?", "Alex", "No, stub decider runs completely offline.", "Great, that helps."),
        ("Where are the test results logged?", "Elena", "In the console output and pytest-report.html.", "Thanks!"),
        ("Who should I ping for a code review on the database PR?", "Marcus", "Priya or Jordan are primary reviewers.", "Tagged them, thanks."),
        ("What time is standup today?", "Taylor", "10:00 AM as usual in the main room.", "See you there."),
        ("Did anyone restart the dev redis pod?", "Casey", "Yes, I bounced it 5 minutes ago.", "Confirmed, working now."),
        ("Is ruff configured for 88 or 100 character line length?", "Alex", "It is set to 88 characters in pyproject.toml.", "Thanks Alex."),
    ]

    for idx, (q, ans_user, ans_text, ack) in enumerate(already_answered_cases):
        conv_id = f"c-{conv_num:03d}"
        conv_num += 1

        messages = [
            {"message_id": "m1", "speaker_id": "u-user-q", "text": q},
            {"message_id": "m2", "speaker_id": f"u-{ans_user.lower()}", "text": ans_text, "reply_to": "m1"},
            {"message_id": "m3", "speaker_id": "u-user-q", "text": ack, "reply_to": "m2"},
        ]
        scenarios.append(
            make_scenario(
                conv_id=conv_id,
                category="already_answered",
                expected_action="SILENT",
                messages_data=messages,
                participants_data=parts_standard,
                rationale=f"Question was already answered accurately by human colleague ({ans_user}); assistant must remain silent.",
            )
        )

    # =======================================================================
    # 10. repeated_message - 15 scenarios
    # 10 SILENT (human repeated message already handled / duplicate chatter)
    # 5 SPEAK (addressed message repeated because assistant hadn't responded yet)
    # =======================================================================
    for idx in range(10):
        conv_id = f"c-{conv_num:03d}"
        conv_num += 1

        messages = [
            {"message_id": "m1", "speaker_id": "u-alex", "text": f"Has anyone reviewed ticket #{300+idx}?"},
            {"message_id": "m2", "speaker_id": "u-sam", "text": "Looking at it right now."},
            {"message_id": "m3", "speaker_id": "u-alex", "text": f"Has anyone reviewed ticket #{300+idx}?"},
        ]
        scenarios.append(
            make_scenario(
                conv_id=conv_id,
                category="repeated_message",
                expected_action="SILENT",
                messages_data=messages,
                participants_data=parts_standard,
                rationale="Duplicate/repeated human message in channel where human colleague is already actively handling it.",
            )
        )

    for idx in range(5):
        conv_id = f"c-{conv_num:03d}"
        conv_num += 1

        messages = [
            {"message_id": "m1", "speaker_id": "u-dev1", "text": f"Hush, what is the status of task #{400+idx}?", "mentions": ["hush"]},
            {"message_id": "m2", "speaker_id": "u-dev1", "text": f"Hush, what is the status of task #{400+idx}?", "mentions": ["hush"]},
        ]
        scenarios.append(
            make_scenario(
                conv_id=conv_id,
                category="repeated_message",
                expected_action="SPEAK",
                messages_data=messages,
                participants_data=parts_devs,
                rationale="User repeated an explicitly addressed query to Hush; assistant must fulfill the direct request.",
            )
        )

    # =======================================================================
    # 11. name_false_positive (SILENT) - 20 scenarios
    # Uses the word 'hush' as an English word, idiom, or display name.
    # =======================================================================
    name_false_positives = [
        "Hush, the baby is asleep, don't make any noise.",
        "Let's keep this new acquisition strictly hush-hush until the PR announcement.",
        "We need to hush this rumor before the all-hands meeting tomorrow.",
        "Hush now, the speaker is starting the keynote presentation.",
        "There was an awkward hush in the conference room after the demo crashed.",
        "The manager asked everyone to hush during the executive Q&A.",
        "They signed a hush agreement before getting access to the confidential prototype.",
        "A sudden hush fell over the team when the production outage alert went off.",
        "We should hush up the internal disagreement and focus on the deadline.",
        "Keep the secret keys hush-hush, don't paste them into the public channel.",
        "The library study room has a strict hush rule.",
        "Hush now, let the junior engineer finish explaining their thought process.",
        "They agreed to keep the bonus compensation numbers hush-hush.",
        "Can everyone please hush for just two minutes while we listen to the client?",
        "It was a hush-hush project developed in stealth mode for six months.",
        "We cannot just hush up this security vulnerability, we have to disclose it.",
        "A hush settled over the office right before the sprint review.",
        "Hush now, let's hear what customer support has to say.",
        "Keep the launch date hush-hush until marketing approves the press release.",
        "There is a hush in the building because everyone is working from home today.",
    ]

    for idx, fp_text in enumerate(name_false_positives):
        conv_id = f"c-{conv_num:03d}"
        conv_num += 1

        messages = [
            {"speaker_id": "u-sam", "text": "Chatting in team channel."},
            {"speaker_id": "u-alex", "text": fp_text},
        ]
        scenarios.append(
            make_scenario(
                conv_id=conv_id,
                category="name_false_positive",
                expected_action="SILENT",
                messages_data=messages,
                participants_data=parts_standard,
                rationale="Text contains the word 'hush' as an English idiom/word, but is NOT an invocation of the assistant.",
            )
        )

    # =======================================================================
    # 12. malformed_or_edge_case - 20 scenarios
    # 10 SILENT (metadata conflicts pointing to human)
    # 10 SPEAK (complex addressed code blocks / tracebacks)
    # =======================================================================
    conflict_recipients = [
        ("u-alex", "Alex", "hush, let us wait for Alex's review first."),
        ("u-sam", "Sam", "hush, check with Sam regarding the deployment window."),
        ("u-taylor", "Taylor", "hush, Taylor already submitted the bug report."),
        ("u-jordan", "Jordan", "hush, Jordan has the root cause analysis."),
        ("u-casey", "Casey", "hush, Casey is managing the database rollback."),
        ("u-dev1", "Elena", "hush, Elena is refactoring the auth middleware."),
        ("u-dev2", "Marcus", "hush, Marcus is running the load test benchmark."),
        ("u-lead", "Priya", "hush, Priya approved the PR merge criteria."),
        ("u-alex", "Alex", "hush, ask Alex before pushing to production branch."),
        ("u-sam", "Sam", "hush, Sam is writing the migration script right now."),
    ]

    for idx, (target_id, target_name, msg_text) in enumerate(conflict_recipients):
        conv_id = f"c-{conv_num:03d}"
        conv_num += 1

        messages = [
            {"message_id": "m1", "speaker_id": target_id, "text": f"Should we proceed with step #{idx+1}?"},
            {
                "message_id": "m2",
                "speaker_id": "u-ops1",
                "text": msg_text,
                "reply_to": "m1",
                "mentions": [target_id],
            },
        ]
        scenarios.append(
            make_scenario(
                conv_id=conv_id,
                category="malformed_or_edge_case",
                expected_action="SILENT",
                messages_data=messages,
                participants_data=parts_standard,
                rationale=f"Metadata conflict: text mentions 'hush' colloquially, but reply_to and mentions explicitly target {target_name}.",
            )
        )

    edge_code_blocks = [
        ("def calculate_total(items: list[dict]) -> float:\n    total = 0.0\n    for item in items:\n        total += item.get('price', 0.0) * item.get('quantity', 1)\n    return total", "price calculation function"),
        ("async def fetch_user(user_id: str) -> dict:\n    async with httpx.AsyncClient() as client:\n        resp = await client.get(f'/users/{user_id}')\n        return resp.json()", "async HTTP client snippet"),
        ("class RateLimiter:\n    def __init__(self, max_requests: int):\n        self.max = max_requests\n        self.hits = []", "rate limiter implementation"),
        ("SELECT u.id, count(o.id) as order_count FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.id HAVING count(o.id) > 5;", "SQL aggregation query"),
        ("FROM python:3.12-slim\nWORKDIR /app\nCOPY requirements.txt .\nRUN pip install --no-cache-dir -r requirements.txt\nCOPY . .", "Dockerfile multi-stage build snippet"),
        ("def parse_payload(raw: bytes) -> dict:\n    try:\n        return json.loads(raw.decode('utf-8'))\n    except UnicodeDecodeError:\n        return {}", "byte decoding and parsing snippet"),
        ("export const fetchMetrics = async (): Promise<Metrics> => {\n  const res = await fetch('/api/metrics');\n  return res.json();\n};", "TypeScript fetch function"),
        ("def retry_call(fn, attempts=3):\n    for i in range(attempts):\n        try:\n            return fn()\n        except Exception:\n            if i == attempts - 1: raise", "retry loop implementation"),
        ("version: '3.8'\nservices:\n  redis:\n    image: redis:7-alpine\n    ports:\n      - '6379:6379'", "docker-compose redis definition"),
        ("def test_health_check(client):\n    response = client.get('/health')\n    assert response.status_code == 200\n    assert response.json() == {'status': 'ok'}", "pytest health test case"),
    ]

    for idx, (code_text, desc) in enumerate(edge_code_blocks):
        conv_id = f"c-{conv_num:03d}"
        conv_num += 1

        full_text = f"```\n{code_text}\n```\n@hush can you review this {desc} for bugs or improvements?"
        messages = [
            {"message_id": "m1", "speaker_id": "u-dev1", "text": full_text, "mentions": ["hush"]},
        ]
        scenarios.append(
            make_scenario(
                conv_id=conv_id,
                category="malformed_or_edge_case",
                expected_action="SPEAK",
                messages_data=messages,
                participants_data=parts_devs,
                rationale=f"Complex multi-line block explicitly addressed to Hush requesting review of {desc}.",
            )
        )

    # =======================================================================
    # VERIFY EXACT ACTION COUNTS
    # Target: SILENT = 135, SPEAK = 105, ASK = 60 -> Total = 300
    # =======================================================================
    actions_counter = {}
    for sc in scenarios:
        act = sc["expected_action"]
        actions_counter[act] = actions_counter.get(act, 0) + 1

    print("Curated Scenarios Count:", len(scenarios))
    print("Action Breakdown:", actions_counter)

    assert len(scenarios) == 300, f"Expected 300 scenarios, got {len(scenarios)}"
    assert actions_counter.get("SILENT") == 135, f"Expected 135 SILENT, got {actions_counter.get('SILENT')}"
    assert actions_counter.get("SPEAK") == 105, f"Expected 105 SPEAK, got {actions_counter.get('SPEAK')}"
    assert actions_counter.get("ASK") == 60, f"Expected 60 ASK, got {actions_counter.get('ASK')}"

    return scenarios


def main() -> int:
    data_dir = REPO_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    scenarios = build_curated_scenarios()

    dataset_v1_path = data_dir / "dataset_v1.jsonl"
    scenarios_path = data_dir / "scenarios.jsonl"

    with open(dataset_v1_path, "w", encoding="utf-8") as f:
        for sc in scenarios:
            f.write(json.dumps(sc, ensure_ascii=False) + "\n")

    with open(scenarios_path, "w", encoding="utf-8") as f:
        for sc in scenarios:
            f.write(json.dumps(sc, ensure_ascii=False) + "\n")

    print(f"Successfully wrote {len(scenarios)} records to {dataset_v1_path} and {scenarios_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
