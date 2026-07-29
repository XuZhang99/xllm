# Copyright 2025-2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from dataclasses import dataclass
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from types import ModuleType
from typing import NoReturn, Sequence, TextIO

from scripts.logger import logger


@dataclass
class ServerProcess:
    rank: int
    process: subprocess.Popen
    log_file: TextIO | None


def _package_binary_path() -> str:
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "xllm")


def _installed_binary_path() -> str | None:
    # When `xllm serve` runs from the repo root, `import xllm` is shadowed by
    # the source-tree `xllm/` package (cwd precedes site-packages on sys.path).
    # That directory has no compiled binary -- it only exists in the installed
    # wheel -- so _package_binary_path() misses. Scan sys.path for an installed
    # `xllm/xllm` executable so the command still works from the source tree.
    this_dir = os.path.dirname(os.path.realpath(__file__))
    for entry in sys.path:
        package_dir = os.path.realpath(os.path.join(entry or os.getcwd(), "xllm"))
        if package_dir == this_dir:
            continue
        candidate = os.path.join(package_dir, "xllm")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def _resolve_binary_path(binary_path: str | None) -> str:
    if binary_path:
        path = os.path.realpath(os.path.expanduser(binary_path))
    else:
        path = _package_binary_path()
        if not os.path.isfile(path):
            fallback = _installed_binary_path()
            if fallback is not None:
                logger.info(
                    "xllm binary not found next to the source-tree package; "
                    "using the installed binary at %s.",
                    fallback,
                )
                path = fallback
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"xllm server binary was not found: {path}. "
            "Build and install the wheel before using `xllm serve`."
        )
    if not os.access(path, os.X_OK):
        raise PermissionError(f"xllm server binary is not executable: {path}")
    return path


def _format_command(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _ensure_python_model_path() -> None:
    # The Python model executor (--model_impl=python) imports the 'xllm.python'
    # subpackage. --python_model_path (or XLLM_PYTHON_MODEL_PATH when the flag
    # is empty) must point at the directory containing the 'xllm' package. For
    # a wheel install that is site-packages — the parent of this launcher's
    # directory. The embedded interpreter does not reliably pick up venv
    # site-packages on its own, so default the env var explicitly; an explicit
    # --python_model_path or a pre-set env var still takes precedence.
    os.environ.setdefault(
        "XLLM_PYTHON_MODEL_PATH",
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
    )


_VISIBLE_DEVICE_ENV_VARS = (
    "ASCEND_RT_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "MLU_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "MUSA_VISIBLE_DEVICES",
)


def _auto_tuning_config_dir() -> str:
    # Profiles live next to this launcher (xllm/auto_config), which resolves
    # the same way in the source tree and in a wheel install.
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "auto_config")


def _extract_model_path(extra_args: Sequence[str]) -> str | None:
    # --model is forwarded to the binary, so it is not parsed by the launcher
    # and must be recovered from the passthrough args. Support both
    # `--model VALUE` and `--model=VALUE`.
    for index, arg in enumerate(extra_args):
        if arg == "--model":
            if index + 1 < len(extra_args):
                return extra_args[index + 1]
            return None
        if arg.startswith("--model="):
            return arg[len("--model=") :]
    return None


def _read_model_type(
    parser: argparse.ArgumentParser,
    model_path: str,
) -> str:
    config_path = os.path.join(
        os.path.realpath(os.path.expanduser(model_path)), "config.json"
    )
    try:
        with open(config_path, "r", encoding="utf-8") as config_file:
            model_config = json.load(config_file)
    except FileNotFoundError:
        parser.error(f"auto-tuning: model config.json not found: {config_path}")
    except json.JSONDecodeError as error:
        parser.error(f"auto-tuning: failed to parse {config_path}: {error}")
    except OSError as error:
        parser.error(f"auto-tuning: failed to read {config_path}: {error}")

    if not isinstance(model_config, dict):
        parser.error(f"auto-tuning: {config_path} must contain a JSON object")

    # Mirror C++ util::get_model_type: prefer model_type, fall back to
    # model_name for configs that only carry model_name.
    model_type = model_config.get("model_type") or model_config.get("model_name")
    if not isinstance(model_type, str) or not model_type:
        parser.error(
            f"auto-tuning: {config_path} must contain a string "
            "`model_type` or `model_name`"
        )
    return model_type


def _load_tuning_module(
    parser: argparse.ArgumentParser,
    py_path: str,
    model_type: str,
) -> ModuleType:
    # Load the profile by file path so we never trigger xllm/__init__'s lazy
    # xllm_export .so loading. The module still resolves `from scripts.logger
    # import logger` via the normal import path.
    spec = importlib.util.spec_from_file_location(
        f"xllm.auto_config.{model_type}", py_path
    )
    if spec is None or spec.loader is None:
        parser.error(f"auto-tuning: failed to load tuning module: {py_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        parser.error(f"auto-tuning: failed to import {py_path}: {error}")
    if not callable(getattr(module, "tune", None)):
        parser.error(
            f"auto-tuning: {py_path} must define a callable `tune(base_config, "
            "context)`"
        )
    return module


def _count_visible_devices() -> int | None:
    for env_var in _VISIBLE_DEVICE_ENV_VARS:
        value = os.environ.get(env_var)
        if value is None:
            continue
        entries = [entry for entry in value.split(",") if entry.strip() != ""]
        return len(entries)
    return None


def _generate_tuned_config(
    parser: argparse.ArgumentParser,
    extra_args: Sequence[str],
) -> str:
    model_path = _extract_model_path(extra_args)
    if not model_path:
        parser.error("auto-tuning requires --model <path>")

    model_type = _read_model_type(parser, model_path)

    config_dir = _auto_tuning_config_dir()
    base_json_path = os.path.join(config_dir, f"{model_type}.json")
    tuning_py_path = os.path.join(config_dir, f"{model_type}.py")
    if not os.path.isfile(base_json_path) or not os.path.isfile(tuning_py_path):
        parser.error(
            f"auto-tuning is not supported for model_type `{model_type}`: "
            f"expected both {base_json_path} and {tuning_py_path} to exist."
        )

    try:
        with open(base_json_path, "r", encoding="utf-8") as base_file:
            base_config = json.load(base_file)
    except (OSError, json.JSONDecodeError) as error:
        parser.error(f"auto-tuning: failed to read {base_json_path}: {error}")
    if not isinstance(base_config, dict):
        parser.error(f"auto-tuning: {base_json_path} must contain a JSON object")

    module = _load_tuning_module(parser, tuning_py_path, model_type)

    detect_hardware = getattr(module, "detect_hardware", None)
    hardware = detect_hardware() if callable(detect_hardware) else None
    context = {
        "model_path": model_path,
        "model_type": model_type,
        "visible_device_count": _count_visible_devices(),
        "hardware": hardware,
    }

    try:
        tuned_config = module.tune(base_config, context)
    except Exception as error:
        parser.error(f"auto-tuning: {tuning_py_path} tune() failed: {error}")
    if not isinstance(tuned_config, dict):
        parser.error(f"auto-tuning: {tuning_py_path} tune() must return a dict")

    output_path = os.path.join(os.getcwd(), f"{model_type}.tuned.json")
    try:
        with open(output_path, "w", encoding="utf-8") as output_file:
            json.dump(tuned_config, output_file, indent=2)
            output_file.write("\n")
    except OSError as error:
        parser.error(
            f"auto-tuning: failed to write tuned config {output_path}: {error}"
        )

    logger.info("auto-tuning: wrote tuned config for %s to %s", model_type, output_path)
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"{os.path.basename(sys.argv[0]) or 'xllm'} serve",
        description=(
            "Launch the packaged xLLM server binary. Unknown arguments are "
            "forwarded to the xllm binary unchanged."
        ),
        allow_abbrev=False,
        # Handle -h/--help ourselves so we can also print the server binary's
        # own help (xllm --help) instead of stopping at the launcher options.
        add_help=False,
    )
    parser.add_argument(
        "-h",
        "--help",
        dest="show_help",
        action="store_true",
        help="Show this launcher help and the xllm server options, then exit.",
    )
    parser.add_argument(
        "--config_json_file",
        "--config-json-file",
        dest="config_json_file",
        default=None,
        help=(
            "JSON config file forwarded to xllm. port and nnodes are used by "
            "this launcher."
        ),
    )
    parser.add_argument(
        "--enable-auto-tuning-gflags",
        "--enable_auto_tuning_gflags",
        dest="enable_auto_tuning",
        action="store_true",
        help=(
            "Generate an optimal JSON config for the model's model_type and "
            "launch with it. The tuned config is written to the current "
            "working directory and forwarded via --config_json_file. Mutually "
            "exclusive with --config_json_file."
        ),
    )
    parser.add_argument(
        "--port",
        "--start-port",
        "--start_port",
        dest="start_port",
        type=int,
        default=8010,
        help="Base service port. Local multi-rank launch uses port + rank.",
    )
    parser.add_argument(
        "--nnodes",
        type=int,
        default=1,
        help="Total number of xllm ranks.",
    )
    parser.add_argument(
        "--node_rank",
        "--node-rank",
        dest="node_rank",
        type=int,
        default=None,
        help="Launch only this rank. If omitted, local ranks 0..nnodes-1 are launched.",
    )
    parser.add_argument(
        "--log-dir",
        "--log_dir",
        dest="log_dir",
        default="log",
        help="Directory for per-rank logs. Use --no-log-files to inherit the console.",
    )
    parser.add_argument(
        "--no-log-files",
        "--no_log_files",
        dest="log_dir",
        action="store_const",
        const=None,
        help="Do not redirect server stdout/stderr to log files.",
    )
    parser.add_argument(
        "--binary-path",
        "--binary_path",
        default=None,
        help="Override the packaged xllm binary path. Mainly useful for development.",
    )
    parser.add_argument(
        "--dry-run",
        "--dry_run",
        action="store_true",
        help="Print the commands that would be launched and exit.",
    )
    return parser


def _load_config_json(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> dict[str, object] | None:
    if args.config_json_file is None or args.config_json_file == "":
        return None

    config_path = os.path.realpath(os.path.expanduser(args.config_json_file))
    try:
        with open(config_path, "r", encoding="utf-8") as config_file:
            config_json = json.load(config_file)
    except FileNotFoundError:
        parser.error(f"--config_json_file does not exist: {config_path}")
    except json.JSONDecodeError as error:
        parser.error(f"failed to parse --config_json_file {config_path}: {error}")
    except OSError as error:
        parser.error(f"failed to read --config_json_file {config_path}: {error}")

    if not isinstance(config_json, dict):
        parser.error("--config_json_file must contain a JSON object")

    args.config_json_file = str(config_path)
    return config_json


def _read_json_int(
    parser: argparse.ArgumentParser,
    config_json: dict[str, object],
    key: str,
    default_value: int,
) -> int:
    if key not in config_json or config_json[key] is None:
        return default_value

    value = config_json[key]
    if isinstance(value, bool) or not isinstance(value, int):
        parser.error(f"--config_json_file field `{key}` must be an integer")
    return value


def _apply_config_json_overrides(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    config_json: dict[str, object] | None,
) -> None:
    if config_json is None:
        return

    args.start_port = _read_json_int(parser, config_json, "port", args.start_port)
    args.nnodes = _read_json_int(parser, config_json, "nnodes", args.nnodes)


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.nnodes < 1:
        parser.error("--nnodes must be greater than 0")
    if args.start_port < 1 or args.start_port > 65535:
        parser.error("--port/--start-port must be in range [1, 65535]")
    if args.node_rank is not None and (
        args.node_rank < 0 or args.node_rank >= args.nnodes
    ):
        parser.error("--node-rank must be in range [0, nnodes)")

    launches_all_local_ranks = args.node_rank is None
    if launches_all_local_ranks and args.nnodes > 1:
        if args.start_port + args.nnodes - 1 > 65535:
            parser.error("--port + --nnodes - 1 must be less than or equal to 65535")


def _resolve_ranks(args: argparse.Namespace) -> list[int]:
    if args.node_rank is not None:
        return [args.node_rank]
    return list(range(args.nnodes))


def _resolve_port(
    args: argparse.Namespace,
    rank: int,
    launches_all_local_ranks: bool,
) -> int:
    if launches_all_local_ranks:
        return args.start_port + rank
    return args.start_port


def _build_command(
    binary_path: str,
    args: argparse.Namespace,
    rank: int,
    extra_args: Sequence[str],
    launches_all_local_ranks: bool,
) -> list[str]:
    port = _resolve_port(args, rank, launches_all_local_ranks)

    command = [binary_path]
    if args.config_json_file is not None:
        command.append(f"--config_json_file={args.config_json_file}")
    command.append(f"--port={port}")
    command.append(f"--nnodes={args.nnodes}")
    command.append(f"--node_rank={rank}")
    command.extend(extra_args)
    return command


def _open_log_file(log_dir: str | None, rank: int) -> TextIO | None:
    if log_dir is None:
        return None
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"node_{rank}.log")
    return open(log_path, "w", encoding="utf-8")


def _start_process(command: Sequence[str], rank: int, log_dir: str | None) -> ServerProcess:
    log_file = _open_log_file(log_dir, rank)
    try:
        process = subprocess.Popen(
            list(command),
            stdout=log_file if log_file is not None else None,
            stderr=subprocess.STDOUT if log_file is not None else None,
        )
    except BaseException:
        if log_file is not None:
            log_file.close()
        raise
    return ServerProcess(rank, process, log_file)


def _terminate_processes(processes: Sequence[ServerProcess]) -> None:
    for server_process in processes:
        if server_process.process.poll() is None:
            server_process.process.terminate()

    deadline = time.time() + 15
    for server_process in processes:
        process = server_process.process
        if process.poll() is not None:
            continue
        timeout = max(0.0, deadline - time.time())
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _close_logs(processes: Sequence[ServerProcess]) -> None:
    for server_process in processes:
        if server_process.log_file is not None:
            server_process.log_file.close()


def _probe_server_ready(
    port: int,
    stop_event: threading.Event,
    poll_interval_s: float = 2.0,
) -> None:
    # Only rank 0 (the master node) serves the HTTP API and /health, so this is
    # the readiness signal for the whole cluster: /health returns 200 once the
    # model is loaded and all workers report healthy. Poll indefinitely; a slow
    # model load must not be reported as a failure. The thread is a daemon and
    # also stops as soon as the main loop signals process exit.
    health_url = f"http://127.0.0.1:{port}/health"
    while not stop_event.is_set():
        try:
            with urllib.request.urlopen(health_url, timeout=2) as response:
                if response.status == 200:
                    logger.info(
                        "xllm server started successfully, serving on port %s "
                        "(health: %s).",
                        port,
                        health_url,
                    )
                    return
        except (urllib.error.URLError, OSError):
            # Not accepting connections yet, or /health still reporting 503
            # (workers connecting / model loading). Keep waiting.
            pass
        stop_event.wait(poll_interval_s)


def _start_readiness_probe(port: int) -> tuple[threading.Thread, threading.Event]:
    stop_event = threading.Event()
    thread = threading.Thread(
        target=_probe_server_ready,
        args=(port, stop_event),
        name="xllm-readiness-probe",
        daemon=True,
    )
    thread.start()
    return thread, stop_event


def _wait_for_processes(processes: Sequence[ServerProcess]) -> int:
    try:
        while True:
            for server_process in processes:
                return_code = server_process.process.poll()
                if return_code is None:
                    continue
                if len(processes) > 1:
                    logger.warning(
                        "xllm rank %s exited with code %s; terminating "
                        "remaining ranks.",
                        server_process.rank,
                        return_code,
                    )
                    _terminate_processes(processes)
                return return_code
            time.sleep(1)
    except KeyboardInterrupt:
        logger.warning("Interrupted; terminating xllm server processes.")
        _terminate_processes(processes)
        return 130


def _install_signal_handlers() -> None:
    def _raise_keyboard_interrupt(signum: int, frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _raise_keyboard_interrupt)
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)


def _print_binary_help(parser: argparse.ArgumentParser, binary_path: str) -> None:
    # Flush our buffered stdout first: when piped, Python stdout is block
    # buffered while the child writes straight to the fd, which would otherwise
    # print the binary help before the launcher help.
    sys.stdout.flush()
    try:
        subprocess.run([binary_path, "--help"], check=False)
    except OSError as error:
        parser.error(f"failed to run xllm binary help {binary_path}: {error}")


def launch_server(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args, extra_args = parser.parse_known_args(argv)

    if args.show_help:
        parser.print_help()
        # Also surface the server binary's own options (HelpFormatter output).
        try:
            binary_path = _resolve_binary_path(args.binary_path)
        except (FileNotFoundError, PermissionError) as error:
            parser.exit(status=0, message=f"\n{error}\n")
        print("\nxllm server options:\n")
        _print_binary_help(parser, binary_path)
        return 0

    if args.enable_auto_tuning:
        if args.config_json_file:
            parser.error(
                "--enable-auto-tuning-gflags and --config_json_file are "
                "mutually exclusive"
            )
        args.config_json_file = _generate_tuned_config(parser, extra_args)

    config_json = _load_config_json(parser, args)
    _apply_config_json_overrides(parser, args, config_json)
    _validate_args(parser, args)

    binary_path = _resolve_binary_path(args.binary_path)

    _ensure_python_model_path()

    launches_all_local_ranks = args.node_rank is None
    ranks = _resolve_ranks(args)
    commands = [
        _build_command(binary_path, args, rank, extra_args, launches_all_local_ranks)
        for rank in ranks
    ]

    for rank, command in zip(ranks, commands):
        logger.info("rank %s: %s", rank, _format_command(command))

    if args.dry_run:
        return 0

    _install_signal_handlers()
    processes: list[ServerProcess] = []
    readiness_stop_event: threading.Event | None = None
    try:
        for rank, command in zip(ranks, commands):
            processes.append(_start_process(command, rank, args.log_dir))
            if args.log_dir is not None:
                logger.info(
                    "rank %s log: %s",
                    rank,
                    os.path.join(args.log_dir, f"node_{rank}.log"),
                )
        # Only rank 0 serves the HTTP API, so probe its port for readiness.
        if 0 in ranks:
            rank_0_port = _resolve_port(args, 0, launches_all_local_ranks)
            _, readiness_stop_event = _start_readiness_probe(rank_0_port)
        return _wait_for_processes(processes)
    except BaseException:
        _terminate_processes(processes)
        raise
    finally:
        if readiness_stop_event is not None:
            readiness_stop_event.set()
        _close_logs(processes)


def _exec_binary(argv: Sequence[str]) -> NoReturn:
    # Replace this process with the packaged xllm binary so `xllm <args>`
    # behaves exactly like invoking the binary directly (same pid, signals,
    # stdio, and exit code). Any argument other than the `serve` subcommand is
    # forwarded verbatim, which keeps `xllm --model ...` working for users who
    # start the server through the binary directly.
    try:
        binary_path = _resolve_binary_path(None)
    except (FileNotFoundError, PermissionError) as error:
        logger.error("%s", error)
        raise SystemExit(1)

    _ensure_python_model_path()
    os.execv(binary_path, [binary_path, *argv])


def main(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)

    if args and args[0] == "serve":
        raise SystemExit(launch_server(args[1:]))

    # Everything else is handed straight to the xllm binary, including no args
    # and -h/--help, so `xllm` is a transparent alias for the server binary.
    _exec_binary(args)


if __name__ == "__main__":
    main()
