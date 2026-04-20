# Phase 14: Test Coverage to 95%+ & Warning Cleanup

## Goal

Push server.py test coverage from 89% to 95%+ by covering exception handlers, import fallback branches, and edge cases. Fix remaining deprecation warnings.

## Current State

- 533 tests, 89% server.py coverage, 112 uncovered lines
- 22 warnings (mostly `datetime.utcnow()` deprecation from monitoring.py)

## Uncovered Line Categories

### 1. Import Fallback Branches (lines 44-89)
The `except ImportError` paths for 6 optional modules. Require mocking imports to simulate unavailability.

### 2. Exception Handler Branches (~20 locations)
Error catch blocks in analyzer classes and `_impl` functions. Require triggering errors via mocked failures.

### 3. Module-Level Initialization Fallbacks (lines 869-919)
The `else` branches where optional singletons are None. Same mock approach as import fallbacks.

### 4. MCP Tool Wrapper Passthroughs (lines 2149-2383)
The `@mcp.tool()` decorated functions are thin delegates. Tests call `_impl` directly so these are uncovered. Not worth testing separately (they're single-line delegates).

### 5. Warning Cleanup
Fix `datetime.utcnow()` -> `datetime.now(datetime.UTC)` in monitoring.py (2 occurrences).

## Approach

- New test file: `tests/test_coverage_gaps.py`
- Use `unittest.mock.patch` for simulating failures
- Fix monitoring.py warnings directly
- Skip covering `__main__` block and MCP wrapper lines (diminishing returns)

## Success Criteria

- server.py coverage >= 95%
- 0 deprecation warnings from our code
- All existing 533 tests still pass
