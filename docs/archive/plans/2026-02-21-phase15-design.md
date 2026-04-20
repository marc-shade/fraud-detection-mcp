# Phase 15: Lint Cleanup

## Goal
Fix all 111 ruff lint errors across 23 files to achieve a clean lint pass.

## Approach
1. Auto-fix 78 errors with `ruff check --fix`
2. Manually fix remaining 33 (E402 import ordering, E722 bare excepts)
3. Run tests after each batch
4. Commit by file category
