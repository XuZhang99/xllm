---
name: add-unit-test
description: Add or update xLLM unit tests in the repository. Use when Codex needs to create a new C++/CUDA/NPU/MLU unit test, place a test under tests/, wire it into CMake with cc_test, update an existing test target, choose platform gates, or validate test naming and dependencies against current xLLM test conventions.
---

# Add Unit Test

## Workflow

1. Inspect the production code, nearest tests, and nearest test `CMakeLists.txt` before writing a new test.
   - Match the production path under `xllm/` to `tests/` where possible.
   - Prefer extending an existing nearby `*_test.cpp` and `cc_test` target when the behavior belongs to the same domain.
   - Create a new test source only when it improves isolation, keeps platform setup separate, or follows an existing directory pattern.

2. Read and apply the project style guide for all new or modified C++/CUDA test source:
   `.agents/skills/code-review/references/custom-code-style.md`.

3. Use current test tree conventions from [xllm-test-patterns.md](references/xllm-test-patterns.md) when adding a new test file, new `cc_test`, platform-specific test, or test directory.
   - Keep unit tests directly in the relevant leaf directory under `tests/`.
   - Use `*_test.cpp` for C++ tests and `*_test.cu` for CUDA source tests.
   - Include production headers with direct project paths such as `core/...`, `api_service/...`, or `function_call/...`; include test helpers with `tests/...`.

4. Wire tests through CMake with `include(cc_test)` and `cc_test(...)`.
   - Keep source names relative to the current test directory unless an existing target already uses an absolute source path for a production `.cpp`.
   - Use target names ending in `_test`.
   - Put platform-directory gates in the parent `CMakeLists.txt` when the whole child directory is platform-specific.
   - Use target-level `if(USE_NPU)`, `if(USE_MLU)`, `if(USE_CUDA)`, or generator expressions only when a mixed directory contains both generic and platform-specific tests.

5. Write tests for observable behavior, not implementation trivia.
   - Cover success, edge, and error paths touched by the change.
   - Prefer deterministic inputs, fixed seeds, and small tensors/data structures.
   - Keep helpers file-local in an anonymous namespace unless shared by multiple test files.
   - Use `TEST`/`TEST_F` names that describe behavior clearly.

6. Validate narrowly before finishing.
   - Always run `git diff --check` for the changed test paths.
   - Search for stale filenames, target names, and old short include paths after moving or renaming tests.
   - Run the narrowest build/test command available locally; if not feasible, state the exact reason and what was checked instead.

## Common Commands

```bash
rg --files tests/<area>
rg "old_test_name|old_file_name|old_header_name" tests xllm CMakeLists.txt
git diff --check -- tests/<area>
```

For full remote validation on the development machine, use the repository AGENTS instructions for SSH, container, build, and test commands.
