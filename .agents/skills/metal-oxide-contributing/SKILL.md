---
name: metal-oxide-contributing
description: Use when preparing metal-oxide Rust changes, documentation, verification, commits, or pull requests for review and publication.
---

# Preparing changes

Read `AGENTS.md` and `CONTRIBUTING.md`. Describe concrete behavior and the checks
actually performed. Keep unrelated work out of a change.

- Define dependencies once in root workspace.dependencies; every consumer uses
  workspace = true. Keep members and dependency keys alphabetically sorted.
- Inherit package settings and lints. Keep handwritten Rust below 600 lines;
  split by responsibility instead of compressing formatting or removing docs.
- Prefer turbofish at the expression when it carries the needed type.
- Review each public type, helper, option, and alternative path against a concrete
  current use case. Keep one clear path per operation; defer speculative wrappers
  and general abstractions. Implementation helpers stay private.
- Write English rustdoc for behavior and caller contracts. Plain comments carry
  non-obvious reasons and safety invariants. Exclude banners, separator lines,
  edit narration, commented-out code, and statement-by-statement prose.
- Use names and small functions to express intent. Omit trivial explanations
  and implementation narration from comments and rustdoc.
- Put release history outside source doc comments.

Run formatting, Clippy with warnings denied, and workspace tests. Execute GPU
checks on actual hardware when relevant. Report skipped checks and environment
limitations; compiling or skipping a GPU test does not establish execution.

Use a concise Conventional Commit subject with `feat`, `fix`, `chore`, `docs`,
`test`, `refactor`, `perf`, or `ci`. For example:

```text
feat: add synchronous Metal dispatch
```

Commit messages, PR text, code, and project docs contain no co-author trailers,
AI attribution, session IDs, tool transcripts, or generated-by footers. A commit
body is optional and explains only the change, rationale, or material validation.
Keep the repository private until the user authorizes changing its visibility.

Keep editor settings local and exclude editor configuration through
`.git/info/exclude`. Commit the compiler's `rustc_private` rust-analyzer metadata
in its `Cargo.toml`.
