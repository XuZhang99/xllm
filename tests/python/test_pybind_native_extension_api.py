# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import json
import signal
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from xllm.pybind import embedding, llm, utils, vlm
from xllm.pybind.args import ArgumentParser


def test_extension_family_cli() -> None:
    parser = ArgumentParser().parser
    assert parser.parse_args([]).extension_impl == ""
    assert parser.parse_args(["--extension_impl", "custom"]).extension_impl == "custom"


@pytest.mark.parametrize(
    "api_module,api_name,backend",
    [(llm, "LLM", "llm"), (llm, "LLM", "vlm"), (vlm, "VLM", "vlm"), (embedding, "Embedding", "llm")],
)
def test_constructor_keeps_extension_selection_per_instance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, api_module: ModuleType, api_name: str, backend: str
) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen2_vl" if backend == "vlm" else "qwen3"}), encoding="utf-8"
    )
    master = MagicMock()
    monkeypatch.setattr(api_module, "VLMMaster" if backend == "vlm" else "LLMMaster", master)
    monkeypatch.setattr(signal, "signal", MagicMock())
    monkeypatch.setattr(utils, "get_free_port", lambda: 26001)
    monkeypatch.setattr(utils.xllm_export, "get_model_backend", lambda _: backend)
    monkeypatch.setattr(utils.xllm_export, "configure_cpp_chat_template", MagicMock())
    constructor = getattr(api_module, api_name)

    selected = constructor(model=str(tmp_path), extension_impl="custom")
    selected_options = master.call_args.args[0]
    default = constructor(model=str(tmp_path))
    default_options = master.call_args.args[0]

    assert selected.master is master.return_value
    assert default.master is master.return_value
    assert isinstance(selected_options, api_module.Options)
    assert isinstance(default_options, api_module.Options)
    assert selected_options is not default_options
    assert selected_options.extension_impl == "custom"
    assert default_options.extension_impl == ""
