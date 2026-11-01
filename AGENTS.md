# AGENTS.md — Multimodal Translator

## 1. Project Overview

This project is a Windows desktop application that provides context-aware translation for games and video content.

Core features:
- Minimal always-on-top overlay UI
- Region-based and full-screen translation
- OCR-based text extraction
- Context-aware translation using Qwen
- Content identification and per-content memory
- Multimodal scene understanding
- Local context storage

The long-term goal is to support interchangeable translation engines, including custom-trained NLP models and optional cloud APIs.

## 2. Development Principles

- Use a Modular Monolith architecture.
- Follow SOLID principles where they improve maintainability.
- Keep components loosely coupled and independently testable.
- Prefer simple solutions over unnecessary abstractions.
- Avoid premature optimization.
- Do not introduce new dependencies without a clear reason.
- Preserve existing functionality when modifying code.
- Do not implement features outside the requested scope.

## 3. Technology Stack

- Python 3.11+
- PySide6 for desktop UI
- Windows APIs and MSS for screen capture
- PaddleOCR for text recognition
- Qwen for translation and visual understanding
- SQLite for local persistence
- pytest for testing

These are preferred technologies, not mandatory dependencies for every development phase.

## 4. Architecture Rules

Separate the application into logical responsibilities:

- UI: overlays, user interactions, settings
- Application: workflows, task scheduling, state management
- Core: shared data models, interfaces, domain rules
- Adapters: OCR, AI models, screen capture, storage, platform integration

Rules:
- UI components must not directly depend on specific AI implementations.
- Translation engines must use a shared interface.
- OCR and scene analysis must remain separate responsibilities.
- Context storage must be independent of translation engines.
- Prefer dependency injection over hard-coded dependencies.
- Use mock implementations when testing application workflows.

## 5. UI and Concurrency Rules

- Never run expensive inference or blocking operations on the Qt UI thread.
- Use appropriate background workers for long-running tasks.
- Keep UI responsive during translation and screen analysis.
- Support multiple monitors and Windows DPI scaling.
- Handle unexpected window closures and display changes.
- Prevent stale translation results from appearing after session changes.
- Ensure overlay windows do not interfere with normal application input.

## 6. AI Integration Rules

- Use Qwen as the initial translation engine.
- Keep model-specific implementation details inside adapters.
- Support future replacement with custom NLP models or cloud APIs.
- Do not assume AI outputs are always correct.
- Validate structured model outputs before using them.
- Store evidence and confidence information separately.
- Do not treat heuristic confidence scores as calibrated probabilities.

## 7. Context Memory Rules

- Separate memory by content identity.
- Distinguish recent dialogue, scene summaries, long-term facts, and archived records.
- Do not send the entire memory history to translation models.
- Retrieve only relevant context within a configurable token budget.
- Avoid repeatedly storing identical subtitle frames.
- Preserve meaningful contextual differences between identical sentences.
- Prevent content from unrelated sessions from being mixed.
- Keep observed facts distinguishable from AI-generated inferences.

## 8. Error Handling and Logging

- Handle expected failures explicitly.
- Do not silently suppress exceptions.
- Log useful diagnostic information without exposing sensitive content.
- Keep the UI operational when AI inference fails.
- Handle GPU out-of-memory errors gracefully.
- Support cancellation and invalidation of outdated tasks.
- Avoid infinite retry loops.
- Provide actionable error messages when recovery is impossible.

## 9. Privacy and Data Rules

- Prefer local processing and local storage.
- Do not retain raw screenshots or audio by default.
- Do not upload user content without explicit authorization.
- Do not automatically collect user translation data for model training.
- Respect copyright, platform restrictions, and third-party licenses.
- Do not implement DRM bypass mechanisms.
- Separate operational context data from research datasets.

## 10. Testing Rules

Use pytest for automated tests.

Prioritize tests for:
- Translation workflow
- Content identification
- Context isolation
- Duplicate detection
- Error handling
- Task cancellation
- Stale response prevention
- Database persistence

Use mocks to test workflows without requiring GPU models.

For bug fixes, add regression tests when practical.

## 11. Code Quality and Documentation

- Use descriptive names for all variables, functions, classes, and modules.
- Use type hints for function parameters and return values.
- Keep functions focused on a single responsibility.
- Avoid unnecessary global mutable state.
- Document non-obvious design decisions.

### Korean Documentation Rules

- All code comments and docstrings MUST be written in Korean.
- Every class, function, and method MUST include a Korean docstring.
- Every variable MUST have a Korean comment explaining its purpose.
- Function docstrings MUST describe the purpose, parameters, return values, and possible exceptions in Korean.
- Use `""" ... """` docstrings for Python classes, functions, and methods.
- Use `#` comments for Python variables and implementation details.
- Use `/** ... */` comments only in languages that support this syntax.
- Keep variable names, function names, class names, and technical identifiers in English.
- Update Korean comments whenever the related implementation changes.
- Do not leave unnecessary English explanations in code comments.

## 12. Codex Working Instructions

Before implementing:
1. Inspect the relevant project files.
2. Understand the current architecture.
3. Identify the smallest reasonable change.
4. Review existing tests and dependencies.

During implementation:
1. Preserve established interfaces.
2. Avoid unrelated refactoring.
3. Add or update tests for changed behavior.
4. Follow existing naming and formatting conventions.
5. Request clarification only when an essential requirement cannot be determined.

After implementation:
1. Run relevant tests when possible.
2. Check for syntax and import errors.
3. Summarize modified files.
4. Explain important design decisions.
5. Report remaining limitations or unverified behavior.
6. Never claim tests passed unless they were actually executed.

## 13. Current Development Scope

Current milestone: Windows Overlay UI Prototype.

Implement only:
- Minimal always-on-top overlay
- Region selection
- Full-screen capture
- Movable translucent translation result window
- Mock translation output
- Basic tests and logging

Do not install or integrate Qwen, PaddleOCR, or other inference models during this milestone unless explicitly requested.

## 14. Project Documentation

Notion is the project documentation reference:

https://app.notion.com/p/3f44130333b8814f9cf3e8bb7a01228c?pvs=204

Use the accessible documentation to understand requirements.

If Notion cannot be accessed, continue using local project requirements and report any missing information. Do not invent undocumented decisions.

## 15. Code Review Policy

### Review Timing
- Perform a code review after completing each feature.
- Perform an additional review after significant architectural changes, AI model integration, database schema changes, or major refactoring.
- Perform a full project review before a production release.
- Small changes may use a lightweight review.

### Review Checklist
1. Check for functional bugs and incorrect behavior.
2. Verify compliance with the project's architecture and coding conventions.
3. Verify that all classes, functions, methods, and variables have the required Korean documentation comments.
4. Check exception handling and resource cleanup.
5. Identify concurrency issues, race conditions, and potential memory leaks.
6. Check for security and privacy vulnerabilities.
7. Check for unnecessary dependencies and duplicated logic.
8. Verify relevant tests and identify missing test cases.
9. Identify possible performance bottlenecks.
10. Ensure no unrelated functionality was modified.

### Review Reporting
- Report issues ordered by severity: Critical, High, Medium, Low.
- Include file paths, line numbers, problem descriptions, and recommended fixes.
- Prioritize actual bugs and risks over subjective style preferences.
- Do not claim that code is correct solely because automated tests pass.
- Distinguish confirmed issues from potential risks.
- Re-run relevant tests after applying fixes.

### Review Completion
- Resolve Critical and High issues before completing a feature.
- Document remaining Medium and Low issues.
- Summarize review results before committing.
- Do not automatically commit changes unless explicitly requested.

## 16. Tuning Parameters and Magic Numbers

- Do not hardcode arbitrary numeric values (magic numbers) in application logic.
- Store adjustable numeric parameters in `config/tuning.py`.
- Examples include:
  - Screen capture and detection intervals
  - OCR confidence thresholds
  - VLM analysis cooldowns
  - Translation timeout and retry limits
  - Context memory token budgets
  - Context confidence weights and thresholds
  - Cache sizes and retention limits
  - Similarity thresholds for duplicate detection
- Use descriptive `UPPER_SNAKE_CASE` constant names.
- Every tuning parameter MUST include a Korean comment explaining its purpose, unit, and expected effect.
- Clearly distinguish experimentally validated values from initial estimates.
- Do not silently modify tuning values without explaining the reason.
- Avoid duplicating configuration values across modules.
- Mathematical constants, protocol-defined values, and values that are inherently part of an algorithm do not need to be moved to `tuning.py`.
- Sensitive values such as API keys and passwords must never be stored in `tuning.py`. Use environment variables or secure storage instead.