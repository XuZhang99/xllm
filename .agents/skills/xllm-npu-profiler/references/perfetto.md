<!-- Copyright 2026 The xLLM Authors. SPDX-License-Identifier: Apache-2.0 -->

# Viewing and analyzing traces in Perfetto

## Load the actual file

1. Open `https://ui.perfetto.dev` with browser controls and use **Open trace file**
   to select the downloaded timeline, or use supported file drag-and-drop.
2. Wait for parsing and confirm a nonempty timeline, process/thread or device
   stream tracks, and selectable events. Record import warnings. The welcome
   page alone does not mean the trace was loaded.
3. Start with an overview, then locate steady requests/steps. Search for kernel,
   HCCL, runtime API, or MSTX names that actually appear. Pin relevant tracks,
   zoom into a window, and inspect slice start times, durations, and arguments.
   Save overview and interval screenshots with rank, phase, and window in filenames.
4. Retain the original local trace. Sharing or public uploading is not required
   to open it. The website URL does not contain the local trace; deliver its
   actual local path as well.

Start with a representative rank. Inspect other ranks when investigating
communication tails or load imbalance. Do not manually concatenate JSON files
to construct a global timeline across ranks.

## Large files or unavailable file selection

Perfetto can connect to a local native Trace Processor. Check the installed tool's
help and the [official large-trace guide](https://perfetto.dev/docs/visualization/large-traces).
Run it on the **same local machine as the browser**, for example:

```bash
# Use a separate tools directory; check for an existing installation first.
curl -fL https://get.perfetto.dev/trace_processor -o trace_processor
chmod +x trace_processor
./trace_processor --httpd /absolute/path/to/msprof_timestamp.json
```

Some versions also offer `trace_processor server http <trace>`; follow the installed
version's help. Open Perfetto, select the detected local accelerator, and confirm
it loaded the intended trace. The default endpoint is `127.0.0.1:9001`; do not bind
the service publicly to work around file selection. If browser controls run on
another machine, verify whether their localhost is the machine holding the trace.
Report the limitation if no connection is possible. Stop the trace processor
started for this task when finished.

## Cross-check with SQL

Inspect `slice` and `track` in the Perfetto Query/SQL panel first. This query lists
hotspots grouped by track and name; it does not directly give kernel-only shares:

```sql
SELECT s.track_id, t.name AS track, s.name,
       COUNT(*) AS calls,
       SUM(s.dur) / 1e6 AS total_ms,
       AVG(s.dur) / 1e3 AS avg_us
FROM slice AS s
LEFT JOIN track AS t ON t.id = s.track_id
WHERE s.dur > 0
GROUP BY s.track_id, t.name, s.name
ORDER BY total_ms DESC
LIMIT 40;
```

Perfetto SQL uses nanoseconds for `ts/dur`; source Chrome JSON commonly uses
microseconds. Do not mix units. Select the rank/device/stream and phase before
filtering `track_id` and `ts`. Clip slices crossing window boundaries to the
selected interval. Nested scopes and concurrent streams cannot simply be summed
to obtain busy time. If no slices appear, inspect import warnings and raw events
instead of drawing conclusions from an empty table.

Ascend JSON may trigger `slice_spill_overlapping_complete_event`: complete events
on one thread overlap without proper nesting, so Perfetto places them on overflow
tracks. Record importer explanations and counts, and compare source events with
imported slices. Do not delete events to suppress warnings or interpret display
overflow as additional physical streams or parallelism.

## Record reproducible evidence

For each bottleneck, record the following in `timeline_notes.md`:

```text
Trace / SHA-256 / rank / device:
Phase and identification evidence:
Selected tracks:
Window [start, end], units, and time origin:
Observation: event names, call counts, durations, gaps, or overlap
Screenshot paths / SQL and filters:
Interpretation: confirmed facts, candidate causes, missing evidence
Next steps: source locations, testable changes, comparison without profiling
```

For decode gaps, identify adjacent device tasks, activity on other streams, host
activity, and synchronization waits. For HCCL, distinguish total communication
time from communication time not overlapped by compute. For graph replay,
distinguish initial capture/compilation from steady replay. Mark unmeasurable
fields as not covered; fixed percentage thresholds do not establish causality.

UI reference: [Official Perfetto UI documentation](https://perfetto.dev/docs/visualization/perfetto-ui).
