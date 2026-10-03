# Independent model forward inputs

## Objective

Develop on `features/refactor_forwardinput`, starting at
`9fbc85c9e709899c09da76b4402842dc0ec48969`.

LLM, VLM, Rec and DiT must own independent input contracts. Repeating token,
position, attention and KV metadata declarations is intentional: a change to one
model domain must not add unused fields or change the contract of another domain.
Shared infrastructure should handle transport, tensor storage, readiness and
lifetime, rather than define a universal model input.

The final domain types are `LlmForwardInput`, `VlmForwardInput`,
`RecForwardInput` and `DiTForwardInput`. They are not aliases or subclasses of a
shared token input. Their owning model parameters must also exclude unrelated
domains; embedding the current universal `ModelInputParams` in four new types
would not achieve the split.

An input variant is allowed at the Worker/WorkerClient/RPC dispatch boundary.
Concrete engines, builders and workers use concrete domain types. Reusing an
executor is allowed through an explicit execution projection with defined
ownership; projections must not deep-copy host metadata on every decode step.

## Commit sequence

The plan itself is a documentation commit. Each implementation step below is a
separate commit, after formatting and its remote validation have passed. Each
intermediate commit must build and preserve current serving behavior.

| Step | Change | Required validation |
| --- | --- | --- |
| 1 | Extract `ForwardRuntimeState` from the existing input. Move host/device buffers, layout/materialization flags, readiness events, retained tensors and KV slot layout without changing the wire format. | Packed input ownership/lazy decoding; batch transport; task input and slot tests. |
| 2 | Route native `DiTForwardInput` from its builder through engine, worker and transport. Remove DiT ownership from token model parameters. Add explicit domain dispatch and protocol validation where needed. | DiT batch and RPC/SHM round trips; generation options, named source ordering/dtypes and SHM overwrite; existing token transport tests. |
| 3 | Introduce independently owned `RecForwardInput` and Rec model parameters. Migrate all Rec builders and pipelines, including ordinary sequence mode. Move decoder sampling and round metadata into Rec. | Rec sequence/multi-round construction; OneRec/xattention/beam metadata; feature embedding inputs; batch regressions. |
| 4 | Name the existing token owner `LlmForwardInput` and migrate ordinary/embedding LLM workers, task slots, graph preparation and speculative draft builders. Its legacy model parameters remain temporary until the VLM path is migrated and step 6 narrows the owning contracts. | Task/slot ownership, empty DP shards, CP metadata, MTP prefill/decode/context and JSON constraints. |
| 5 | Introduce independently owned `VlmForwardInput` and VLM builders/workers. Preserve typed VLM decode and explicitly project VLM targets into LLM draft inputs. | Multimodal prefill/decode, mRoPE/deep stacks, vision embedding and speculative target/draft ownership. |
| 6 | Remove the universal owning input and temporary compatibility adapters. Keep the variant only at dispatch boundaries; finish domain codecs and protocol checks. | All affected batch, transport, worker/task and speculative unit suites; compile the affected native targets. |

Update this document with the actual commit and validation results as each step
lands. A temporary compatibility bridge must be named explicitly and removed by
step 6; it is not the final architecture.

## Behavior and ownership constraints

- VLM decode remains a VLM input even when its multimodal data is empty. The
  configured domain, rather than populated fields, determines dispatch.
- Rec uses feature inputs named `MULTI_MODAL_VALUES` and
  `MULTI_MODAL_INDICES`; preserve them as Rec data. Rec multi-round needs both
  sampling configurations and `StepDecodeMeta`.
- Ordinary single-round LlmRec executes through RPC/SHM. OneRec and multi-round
  Rec use local pipelines. Preserve both paths. The packed Rec codec transports
  ordinary sequence inputs and feature embeddings; reject local strategy
  parameters, decoder sampling and round metadata that the codec cannot carry.
- VLM speculative targets retain their vision state; LLM drafts explicitly
  exclude vision and target-only recurrent state.
- Preserve pinned storage, CPU host views, continuous device buffer ownership,
  stream readiness, no-sync retained inputs and one-shot CP KV remapping.
- Preserve DiT source conversion rules and SHM-owned tensor stabilization.
- RPC and SHM currently share a positional packed codec. Keep the tensor arena
  aligned. Validate domain/version/layout before tensor materialization. A
  reader that accepts an old layout does not make new writers compatible with
  old workers; document or negotiate supported peer versions explicitly.
- Data structs have no methods. Types that retain methods use `class ... final`.
  Run clang-format 20.1.6 with the repository configuration on every touched C++
  file in each step.

## Remote validation isolation

The shared remote checkout is
`cruise@192.168.200.27:/mnt/models1/home/cruise/xllm/xllm`.
It may be used by other chats and must not be switched, reset, cleaned or built
in place for this work.

This task's remote Git worktree is
`/mnt/models1/home/cruise/xllm/codex-forwardinput-a431`, with a fresh build at
`build/cmake.linux-aarch64-cpython-311`. Execute commands inside the existing
`cruise` container using `sudo -n docker exec cruise bash ...`.

Transfer only this branch's commits into the worktree and verify its HEAD before
each run. Initialize clean submodule source checkouts. Read-only dependency
reuse is allowed; source edits, generated artifacts, CMake caches and test logs
belong to this worktree. Prevent builds from installing/replacing shared CANN
operators or changing other checkouts.

Use the narrowest relevant CMake targets for each step, then broaden validation
for the final cleanup. Record the tested commit, build configuration, commands,
exit codes, counts and log locations. Existing binaries from another commit do
not count as validation of this branch.

## Progress

| Step | Commit | Remote validation |
| --- | --- | --- |
| Plan | `7d35c13d` | Documentation only; no executable behavior changed. |
| 1 | `f50c5cae` | All 11 affected targets built; 221 ordinary tests passed across 12 invocations. 17 death-containing cases were excluded from this run. Results: `validation/step1-status.tsv` in the isolated remote worktree. |
| 2 | `f9700d37` | All 269 build tasks succeeded; 95 ordinary tests passed (packed 12, Batch 79, DiT 4). Results: `validation/step2-status.tsv`. The tested patch SHA256 is `5dd4d11dc2c1a8916e2140f7ba061bc542108bb570028bde02fe34e05daa60bd`. |
| 3 | This commit | Affected targets built; 99 ordinary tests passed (packed 16, Batch 83). All build/test exit codes were zero. Results: `validation/step3-status.tsv`; the initial missing-header build failure is preserved in `validation/step3-build-initial-failed.log`. The tested patch SHA256 is `8f2fff3e2f3427285b684448911396942b7eaedb9fa00793963a86d5a4614206`, plus the factory test include fix (file SHA256 `88afbd7ac79480185d71792772acd7acd09ea4d627fc02007749d32b062a5208`). |
| 4-6 | Pending | Pending. |

Remote builds use the NPU configuration in
`build/cmake.linux-aarch64-cpython-311` and device 15. The baseline at
`9fbc85c9` passed 219 ordinary tests; step 1 adds two packed-input tests.
Bounded test invocations and per-step logs are under `validation/`. Baseline
results and exclusions are recorded separately from branch validation.
