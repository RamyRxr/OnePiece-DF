# Claude Code Toolset — Ramy's Global Setup

**What this file is:** a complete inventory of the agents, skills, hooks, and rules installed at `C:\Users\Ramy\.claude\` on this machine, written so a Claude Code session can decide what to reach for without scanning the folders first.

**How to use it:** read the Quick Routing table, then jump to the matching section. Everything listed as active is available immediately — no install step.

> Verified against the live folder. Counts: **48 agents**, **126 skills**, **29 hook blocks** across 7 events, **22 rule folders**.

---

## 1. Quick routing — task to tool

| The task | Reach for |
|---|---|
| Plan a complex feature | `planner` agent (opus) |
| Architectural decision | `architect` agent (opus) |
| Just wrote or changed code | `code-reviewer` agent — **required by the user's rules** |
| New feature or bug fix | `tdd-guide` agent + `tdd-workflow` skill |
| Build or type errors | the matching `*-build-resolver` agent |
| Auth, user input, secrets, payments | `security-reviewer` agent + `security-review` skill |
| Codebase feels messy / tech debt | `anti-koshary` skill |
| Unfamiliar codebase | `codebase-onboarding` skill or `code-explorer` agent |
| UI/UX design work | `design` (built-in) → see §4 |
| Colors / dark mode palette | `dataviz` skill — the only validated palette + validator |
| Dead code cleanup | `refactor-cleaner` agent |
| Library or API docs | `documentation-lookup` skill (Context7) |
| Before writing anything new | `search-first` skill — user's rules mandate research first |

---

## 2. Operating facts that change behavior

Specific to this machine. Read before acting.

1. **Agents are pre-authorized.** `~/.claude/rules/ecc/common/agents.md` states *"Immediate Agent Usage — No user prompt needed"* for `planner`, `code-reviewer`, `tdd-guide`, and `architect`. This overrides the default "don't spawn agents unless asked." Use them proactively.

2. **Per-language rules are NOT auto-loaded.** Only `rules/ecc/common/` (10 files) loads at session start. The 21 language folders — `python/`, `react/`, `typescript/`, `rust/`, `vue/`, etc. — are inert. Read them explicitly when working in that language:
   `Read ~/.claude/rules/ecc/react/ and follow it for this session.`

3. **Some hooks block actions.** `git commit --no-verify` is blocked outright. Linter and formatter config files cannot be edited. Do not plan around these.

4. **`extra/` is storage, not config.** `extra/agents` (19) and `extra/skills` (154) are inert until moved into `agents/` or `skills/`.

5. **`taste` is misleadingly named.** It is creative direction for *music videos* in the angelcore/hyperpop family — not general design taste. Never use it for UI work.

6. **User standards (from `rules/ecc/common/`, always active):** immutability — never mutate, always return new objects; files under 800 lines, functions under 50; 80% test coverage minimum; TDD red → green → refactor; conventional commits; no hardcoded secrets.

---

## 3. Agents (48)

Separate Claude instances with their own context window and restricted tools. They return one report; their intermediate work never enters the main context. Use when a task would read many files, or needs an isolated specialist.

### Planning and architecture

| Agent | Model | Use when |
|---|---|---|
| `planner` | opus | Complex features, refactors — produces a phased plan |
| `architect` | opus | System design, scalability, technical decisions |
| `code-architect` | sonnet | Feature blueprint from existing repo patterns — files, interfaces, build order |
| `code-explorer` | sonnet | Trace execution paths, map layers, document dependencies |

### Review and quality (read-only — cannot edit code)

| Agent | Use when |
|---|---|
| `code-reviewer` | After any code change. Quality, security, maintainability |
| `code-simplifier` | Simplify recently modified code while preserving behavior |
| `refactor-cleaner` | Dead code removal — runs knip, depcheck, ts-prune |
| `performance-optimizer` | Bottlenecks, bundle size, memory leaks, render performance |
| `silent-failure-hunter` | Swallowed errors, bad fallbacks, missing propagation |
| `type-design-analyzer` | Type encapsulation, invariants, enforcement |
| `comment-analyzer` | Comment accuracy and rot risk |
| `pr-test-analyzer` | PR test coverage quality — behavioral, not line count |

### Security

| Agent | Use when |
|---|---|
| `security-reviewer` | **Mandatory** before commits touching auth, user input, APIs, or secrets. Flags OWASP Top 10, SSRF, injection, unsafe crypto |

### Language reviewers (read-only)

`typescript-reviewer` · `python-reviewer` · `go-reviewer` · `rust-reviewer` · `java-reviewer` (auto-detects Spring Boot vs Quarkus) · `kotlin-reviewer` · `swift-reviewer` · `cpp-reviewer` · `csharp-reviewer` · `fsharp-reviewer` · `php-reviewer` · `react-reviewer` · `vue-reviewer` · `django-reviewer` · `fastapi-reviewer` · `flutter-reviewer` · `database-reviewer` (PostgreSQL/Supabase)

### Build error resolvers (can edit — minimal diffs only, no architecture changes)

`build-error-resolver` (TS/general) · `react-build-resolver` (Vite, webpack, Next, CRA, Parcel, esbuild, Bun) · `go-build-resolver` · `rust-build-resolver` · `java-build-resolver` · `kotlin-build-resolver` · `swift-build-resolver` · `cpp-build-resolver` · `dart-build-resolver` · `django-build-resolver` · `pytorch-build-resolver` (CUDA, tensor shapes, DataLoader)

### Specialists

| Agent | Use when |
|---|---|
| `tdd-guide` | New feature or bug fix — enforces tests-first, 80% coverage |
| `e2e-runner` | E2E tests; Vercel Agent Browser preferred, Playwright fallback |
| `a11y-architect` | WCAG 2.2 for web and native; design systems, component audits |
| `seo-specialist` | Technical SEO, structured data, Core Web Vitals |
| `doc-updater` | (haiku) Codemaps, READMEs, guides |
| `docs-lookup` | Library and API docs via Context7 |
| `harmonyos-app-resolver` | HarmonyOS / ArkTS / ArkUI |

---

## 4. Skills (126)

Folders of instructions. Only the description sits in context; the body loads on invoke. Triggered by matching the user's wording, or forced with `/skill-name`.

### Code quality and architecture

**`anti-koshary`** — two-pass audit of a congealed codebase. Pass 1 reports and changes nothing; Pass 2 fixes only after approval. Triggers on "spaghetti code", "tech debt", "this is a mess", "unmaintainable", or a security review request.

`repo-scan` · `codebase-onboarding` · `code-tour` · `codehealth-mcp` · `coding-standards` · `hexagonal-architecture` · `architecture-decision-records` · `click-path-audit` · `inherit-legacy-style` · `intent-driven-development` · `content-hash-cache-pattern` · `regex-vs-llm-structured-text` · `plankton-code-quality`

### Security

`security-review` (checklist for auth, input, secrets, payments) · `security-scan` (audits the `.claude/` config itself) · `security-bounty-hunter`

### Testing

`tdd-workflow` · `e2e-testing` (Playwright) · `ai-regression-testing` · `browser-qa` · `windows-desktop-e2e` (WPF, WinForms, Qt) · `canary-watch` · `python-testing` · `golang-testing` · `rust-testing` · `kotlin-testing` · `react-testing` · `cpp-testing` · `csharp-testing` · `fsharp-testing`

### Design and UI/UX

| Skill | Use for |
|---|---|
| `design` *(built-in)* | **The design canvas.** Multi-artboard mockups, wireframes, screen flows, landing pages, posters. Published as an editable artifact |
| `dataviz` *(built-in)* | Charts plus **the validated color palette and a runnable validator**. Dark-mode values included |
| `frontend-design-direction` | Set design direction before building production UI |
| `design-system` | Generate or audit a design system; visual consistency |
| `make-interfaces-feel-better` | Polish pass: spacing, type, shadows, hit areas, states |
| `artifact-design` *(built-in)* | Page contract, dark-mode token pattern |
| `artifact-diagramming` *(built-in)* | Diagrams that show real mechanism |
| `accessibility` | WCAG 2.2 AA, web plus native traits |
| `frontend-a11y` | React/Next: ARIA, focus management, keyboard nav |
| `liquid-glass-design` | iOS 26 Liquid Glass material |
| `ui-to-vue` | Screenshots or design exports into Vue 3 components |
| `ui-demo` | Record UI demo videos with Playwright |
| `frontend-slides` | Animation-rich HTML presentations |
| `ios-icon-gen` | iOS icons from SF Symbols or Iconify |
| `dashboard-builder` | Grafana/SigNoz operator dashboards |

**Motion — layered, load in order:** `motion-foundations` (base: tokens, springs, accessibility, SSR) → `motion-patterns` (modals, toasts, transitions) → `motion-advanced` (drag, gestures, SVG paths). `motion-ui` overlaps these; prefer the trio.

**Dark mode note:** dark is not a flip of light. `dataviz` ships the same hues *re-stepped* for the dark surface (`#1a1a19`), validated at OKLCH lightness 0.48–0.67.

### Brand and content

`brand-discovery` (8-module identity interview producing a brandbook) · `brand-voice` · `seo` · `taste` (**music video aesthetics only**) · `competitive-platform-analysis`

### Frontend frameworks

`react-patterns` · `react-performance` · `vue-patterns` · `nuxt4-patterns` · `angular-developer` · `frontend-patterns` · `vite-patterns` · `nextjs-turbopack` · `bun-runtime`

### Backend and APIs

`backend-patterns` · `api-design` · `api-connector-builder` · `nestjs-patterns` · `fastapi-patterns` · `django-patterns` · `django-celery` · `django-security` · `django-tdd` · `django-verification` · `springboot-patterns` · `springboot-security` · `springboot-tdd` · `springboot-verification` · `quarkus-patterns` · `quarkus-security` · `quarkus-tdd` · `quarkus-verification` · `laravel-patterns` · `laravel-security` · `laravel-tdd` · `laravel-verification` · `jpa-patterns` · `mcp-server-patterns` · `error-handling` · `latency-critical-systems` · `data-throughput-accelerator`

### Languages

`python-patterns` · `golang-patterns` · `rust-patterns` · `kotlin-patterns` · `kotlin-coroutines-flows` · `kotlin-ktor-patterns` · `kotlin-exposed-patterns` · `java-coding-standards` · `cpp-coding-standards` · `dotnet-patterns` · `dart-flutter-patterns` · `flutter-dart-code-review` · `swiftui-patterns` · `swift-concurrency-6-2` · `swift-actor-persistence` · `swift-protocol-di-testing`

### Mobile

`react-native-patterns` · `android-clean-architecture` · `compose-multiplatform-patterns`

### Data and infrastructure

`postgres-patterns` · `mysql-patterns` · `redis-patterns` · `prisma-patterns` · `database-migrations` · `docker-patterns` · `kubernetes-patterns` · `deployment-patterns` · `flox-environments`

### Workflow and research

`search-first` (**research before coding — required by the user's rules**) · `documentation-lookup` · `deep-research` · `exa-search` · `git-workflow` · `github-ops` · `jira-integration` · `mailtrap-email-integration` · `data-scraper-agent`

### Built-in Claude Code skills (not in the folder, always available)

`code-review` · `simplify` · `run` · `loop` · `schedule` · `update-config` · `keybindings-help` · `fewer-permission-prompts` · `claude-api` · `init`

---

## 5. Hooks — automatic, cannot be skipped

Run by the Claude Code harness, not by the model. 29 blocks across 7 events. A rule asks for compliance; a hook removes the option.

### PreToolUse (8) — can block an action

| Matcher | Script | Effect |
|---|---|---|
| `Bash` | `pre-bash-dispatcher.js` | Routes Bash sub-hooks: **blocks `git commit --no-verify`** and `core.hooksPath=`; pre-commit quality check; dev-server block; git-push reminder |
| `Write\|Edit\|MultiEdit` | `config-protection.js` | **Blocks edits to linter and formatter config files** |
| `Edit\|Write\|MultiEdit` | `gateguard-fact-force.js` | Forces investigation before editing — read first, then edit |
| `Write` | `doc-file-warning.js` | Warns on ad-hoc doc file creation |
| `Edit\|Write` | `suggest-compact.js` | Suggests compaction at strategic points |
| `*` | `observe-runner.js`, `mcp-health-check.js` | Observability, MCP health |
| `Bash\|Write\|Edit\|MultiEdit` | `governance-capture.js` | Records governance events |

### PostToolUse (10)

`quality-gate.js` (quality checks after edits) · `design-quality-check.js` (frontend design reminder) · `post-edit-accumulator.js` (queues edited JS/TS for batch processing) · `post-edit-console-warn.js` (flags `console.log`) · `post-bash-dispatcher.js` (build-complete, PR-created, command log) · `session-activity-tracker.js` · `ecc-metrics-bridge.js` · `ecc-context-monitor.js` · `governance-capture.js` · `observe-runner.js`

### Stop (6) — end of each response

`stop-format-typecheck.js` — **batch formats and typechecks every JS/TS file edited this turn** (formatting happens here, not at PostToolUse) · `check-console-log.js` · `cost-tracker.js` (the `COST NOTICE` lines) · `session-end.js` (persists learnings) · `evaluate-session.js` · `desktop-notify.js`

### SessionStart (2)

`session-start-bootstrap.js` (loads ECC context and the `common/` rules) · `herdr-agent-state.ps1`

### PreCompact (1), PostToolUseFailure (1), SessionEnd (1)

`pre-compact.js` (saves a summary before compaction) · `mcp-health-check.js` · `session-end-marker.js`

**Practical consequences:** do not try to bypass git hooks; do not edit ESLint or Prettier configs; expect JS/TS to be reformatted after the turn ends; investigate files before editing them, because a gate enforces it.

---

## 6. Rules (`~/.claude/rules/ecc/`)

**`common/` — auto-loaded every session (10 files):**
`agents.md` · `code-review.md` · `coding-style.md` · `development-workflow.md` · `git-workflow.md` · `hooks.md` · `patterns.md` · `performance.md` · `security.md` · `testing.md`

**21 language folders — NOT auto-loaded.** Must be read explicitly:
`angular` `arkts` `cpp` `csharp` `dart` `fsharp` `golang` `java` `kotlin` `nuxt` `perl` `php` `python` `react` `react-native` `ruby` `rust` `swift` `typescript` `vue` `web`

To make them automatic for one repo, add this to that project's `CLAUDE.md`:

```
Follow ~/.claude/rules/ecc/<language>/ and ~/.claude/rules/ecc/common/ in this repo.
```

---

## 7. Inactive inventory

Present but not loaded. Move a folder into `agents/` or `skills/` to activate.

- `~/.claude/extra/agents/` — 19 agents (healthcare, homelab, crypto pipeline, agent-orchestration meta-tooling)
- `~/.claude/extra/skills/` — 154 skills, including `remotion-video-creation`, `manim-video`, `blender-motion-state-inspection`, `gan-style-harness`, `loop-design-check`, `dmux-workflows`
- `~/.claude/extra/other-ai-tools/` — the same kit adapted for Cursor, Codex, Gemini, Kiro, Trae

---

## 8. MCP servers

| Server | Status |
|---|---|
| Claude Docs | Connected |
| Notion | Connected |
| **Figma** | **Needs authorization.** Design-to-code, code-to-design, Code Connect, FigJam. Authorize via claude.ai connector settings or `/mcp` |
| **obsidian** | **Connection refused.** Local server not running |
| Context7 | Used by `docs-lookup` and `documentation-lookup` |

---

## 9. Standard workflow (from the user's rules)

```
0. search-first        — GitHub search, then library docs, then Exa.
                         Prefer proven code over net-new.
1. planner agent       — phased plan before coding
2. tdd-guide agent     — red → green → refactor, 80% coverage
3. code-reviewer agent — immediately after writing code
   security-reviewer   — if auth, input, secrets, or payments are touched
4. commit              — conventional commits; --no-verify is blocked
```

**Severity handling:** CRITICAL blocks the merge. HIGH should be fixed first. MEDIUM is informational. LOW is optional.

---

*Generated 2026-09-20 from the live contents of `C:\Users\Ramy\.claude\`. Regenerate if agents, skills, or hooks change.*
