# xLLM Unit Test Patterns

Use this reference when creating or changing unit tests under `tests/`.

## Layout

- Mirror the production structure where practical:
  - `xllm/core/framework/tokenizer` -> `tests/core/framework/tokenizer`
  - `xllm/core/layers/mlu` -> `tests/core/layers/mlu`
  - `xllm/function_call/...` -> `tests/function_call/...`
- Keep tests directly in the relevant leaf directory.
- Avoid new nested `test/` or `tests/` directories. Existing unit tests now live directly in their parent test directories.
- Shared helpers may live beside tests, such as `tests/core/layers/mlu/tests_utils.cpp`.

## Naming

- Test source files use singular suffixes:
  - C++: `thing_test.cpp`
  - CUDA source: `thing_test.cu`
- Test CMake target names also end in `_test`.
- If a target aggregates several source files, keep the target name at the domain level, for example `layer_test`, `moe_layer_test`, or `sampler_test`.
- Keep helper files out of the `*_test.cpp` suffix unless they define tests.

## CMake Basics

Use `cc_test` for C++/CUDA unit test binaries. `tests/CMakeLists.txt` should stay as a simple test-tree entry point; do not add broad include-path shims there to compensate for moved tests.

```cmake
include(cc_test)

cc_test(
  NAME
    feature_test
  SRCS
    feature_test.cpp
  DEPS
    :feature
    GTest::gtest_main
    glog::glog
)
```

Rules:

- Add `include(cc_test)` in each CMake file that declares `cc_test`.
- Use dependencies that match nearby tests first.
- Prefer `GTest::gtest_main`; add `GTest::gtest` only when nearby tests need it or the target explicitly uses it.
- Add `target_link_libraries(...)` and `add_dependencies(...)` after `cc_test` when needed for `brpc`, `OpenSSL`, `protobuf`, platform runtime libraries, or link-group handling.
- Use `:target_name` for local production CMake targets where existing tests do so.

## Include Paths

Use direct project paths in test source files instead of relying on extra test-only include directories.

Examples:

```cpp
#include "core/framework/batch/batch.h"
#include "core/kernels/cuda/cuda_ops_api.h"
#include "api_service/sample_service_impl.h"
#include "function_call/qwen25_detector.h"
#include "function_call/partial_json_parser/include/partial_json_parser/parser.h"
#include "tests/core/layers/mlu/tests_utils.h"
```

Rules:

- Production headers under `xllm/core` should be included as `core/...`.
- Production headers under `xllm/api_service` should be included as `api_service/...`.
- Production headers under `xllm/function_call` should be included as `function_call/...`.
- Test helper headers under `tests` should be included as `tests/...`.
- Leave external SDK headers and generated headers in their natural form, such as `acl/acl.h` or `anthropic.pb.h`.
- Do not add include-path logic to `cmake/cc_test.cmake` for test relocation. Keep `cc_test` generic.

## Platform Gates

Gate entire platform-only directories in the parent `CMakeLists.txt`.

Current examples:

```cmake
if(USE_CUDA)
  add_subdirectory(cuda)
endif()

if(USE_NPU)
  add_subdirectory(npu)
endif()
```

```cmake
if(USE_CUDA)
  add_subdirectory(cuda)
endif()

if(USE_MLU)
  add_subdirectory(mlu)
endif()
```

Do not repeat the same platform `if(...)` inside every child CMake file when the parent already gates the directory.

Use target-level platform gates only for mixed directories where generic and platform-specific tests coexist, such as `tests/core/runtime` or framework directories with both generic and NPU-only targets.

Use generator expressions for platform-specific optional link libraries when the target exists across platforms:

```cmake
target_link_libraries(example_test
                      PUBLIC
                      Python::Python
                      $<$<BOOL:${USE_NPU}>:ascendcl>
                      $<$<BOOL:${USE_NPU}>:hccl>
                      $<$<BOOL:${USE_NPU}>:c_sec>)
```

## Test Design

- Test behavior through public or stable internal interfaces used by nearby tests.
- Cover the regression or edge case that motivated the test.
- For parser and pure logic tests, keep inputs small and assert exact outputs/errors.
- For tensor/device tests, keep tensor shapes small, check dtype/device expectations, and compare against a simple reference implementation.
- For forked-process or device-init-sensitive tests, follow nearby standalone target patterns and leave a short CMake comment explaining why the target is isolated.

## Validation Checklist

Before finishing:

```bash
rg --files tests/<area>
rg "old_file_name|old_target_name" tests xllm CMakeLists.txt
rg '^#include "' tests/<area>
git diff --check -- tests/<area>
```

Run the narrowest feasible validation:

- Local CMake/build target if available.
- `python setup.py test` in the project container when full validation is requested or risk is high.
- For development-machine validation, follow the repo AGENTS instructions for `ssh gpu-h800-195`, `/export/home/zhangxu709/xllm`, container `zx-xllm-cuda`, and the build/test commands.
