# Phase 15: Lint Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix all 111 ruff lint errors across 23 files to achieve a clean `ruff check` pass.

**Architecture:** Use `ruff check --fix` for auto-fixable errors first, then manually fix the remaining issues. Work in batches by file category: core modules, support modules, tests, examples. Run the full test suite after each batch to catch regressions.

**Tech Stack:** ruff (linter), pytest

---

### Task 1: Auto-fix all safe ruff errors

**Files:**
- All 23 files with lint errors

**Step 1: Run ruff auto-fix**

```bash
ruff check --fix . --exclude=venv,test_data,.claude
```

This handles ~78 errors: F401 (unused imports), F541 (f-string placeholders), F841 (unused variables), F811 (redefined unused), E712 (`== True`/`== False` comparisons).

**Step 2: Run tests to verify no regressions**

Run: `python -m pytest tests/ -q --tb=short`
Expected: 560 passed, 2 skipped

**Step 3: Check remaining errors**

Run: `ruff check . --exclude=venv,test_data,.claude --statistics`
Expected: Only E402 (module-import-not-at-top) and E722 (bare-except) remain (~33 errors)

**Step 4: Commit**

```bash
git add -u
git commit -m "style: Auto-fix ruff lint errors (unused imports, f-strings, bool comparisons)"
```

---

### Task 2: Fix E722 bare excepts

**Files:**
- Check which files have E722 with: `ruff check . --exclude=venv,test_data,.claude --select E722`

E722 means `except:` without specifying an exception type. Replace each bare `except:` with `except Exception:`.

**Step 1: Find and fix bare excepts**

Run: `ruff check . --exclude=venv,test_data,.claude --select E722`

For each match, change `except:` to `except Exception:`. The bare excepts are likely in:
- `benchmarks.py` (inside try blocks for model scoring)
- `cli.py` (inside try blocks for optional imports)

**Step 2: Run tests**

Run: `python -m pytest tests/ -q --tb=short`
Expected: 560 passed, 2 skipped

**Step 3: Commit**

```bash
git add -u
git commit -m "style: Replace bare except with except Exception"
```

---

### Task 3: Fix E402 module-import-not-at-top-of-file

**Files:**
- Check which files have E402 with: `ruff check . --exclude=venv,test_data,.claude --select E402`

E402 fires when imports appear after non-import code. Common patterns causing this:
- Imports after `os.environ` setup (legitimate -- add `# noqa: E402` comments)
- Imports after conditional blocks like `try/except` for optional deps (legitimate -- add `# noqa: E402`)
- Imports inside functions (generally fine, ruff shouldn't flag these)

**Step 1: Find all E402 violations**

Run: `ruff check . --exclude=venv,test_data,.claude --select E402`

For each match, decide:
- If the import MUST come after setup code (e.g., `os.environ` set before import), add `# noqa: E402` on the import line
- If the import can be moved to the top of the file, move it

**Step 2: Run tests**

Run: `python -m pytest tests/ -q --tb=short`
Expected: 560 passed, 2 skipped

**Step 3: Commit**

```bash
git add -u
git commit -m "style: Fix E402 import ordering (noqa where necessary)"
```

---

### Task 4: Final verification

**Step 1: Run ruff check -- should be clean**

Run: `ruff check . --exclude=venv,test_data,.claude`
Expected: 0 errors (or "All checks passed!")

**Step 2: Run ruff format check**

Run: `ruff format --check . --exclude=venv,test_data,.claude`
Note: Don't auto-format if it would create a large diff. Just check.

**Step 3: Run full test suite**

Run: `python -m pytest tests/ -q --tb=short`
Expected: 560 passed, 2 skipped

**Step 4: Commit any remaining fixes**

```bash
git add -u
git commit -m "chore: Phase 15 lint cleanup complete, 0 ruff errors"
```
