# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) or Cursor (cursor.com) when working with code in this repository.

## Project Overview

xLLM is an efficient LLM inference framework, specifically optimized for Chinese AI accelerators, enabling enterprise-grade deployment with enhanced efficiency and reduced cost.

## Quick Reference

| Task | Command |
| ---- | ------- |
| Initialize submodules | `git submodule update --init` |
| Build xLLM binary | `python setup.py build` |
| Build xLLM wheel | `python setup.py bdist_wheel` |
| Test xLLM | `python setup.py test` |
| Test specific unit test | `python setup.py test --test-name <test_name>` |
| Build xLLM binary for A3 machine | `python setup.py build --device a3` |
| Build xLLM wheel for A3 machine | `python setup.py bdist_wheel --device a3` |
| Install pre-commit hooks | `pre-commit install` |




## Quick Start for Development

### Setup xLLM

```bash
git clone https://github.com/jd-opensource/xllm
cd xllm

# install pre-commit hooks for the first time
pip install pre-commit
pre-commit install

git submodule update --init
```

### Build xLLM

Build xLLM binary or wheel, but add `--device a3` for A3 machine.

```bash
# build bin
python setup.py build

# build wheel
python setup.py bdist_wheel

# build xllm so
python setup.py build --generate-so
```

### Unit Test


```bash
# test all unit tests
python setup.py test

# test specific unit test
python setup.py 

```


## Directory Structure


├── xllm/
|   : main source folder
│   ├── api_service/               # code for api services
│   ├── c_api/                     # code for c api
│   ├── cc_api/                    # code for cc api 
│   ├── core/  
│   │   : xllm core features folder
│   │   ├── common/                
│   │   ├── distributed_runtime/   # code for distributed and pd serving
│   │   ├── framework/             # code for execution orchestration
│   │   ├── kernels/               # adaption for npu kernels adaption
│   │   ├── layers/                # model layers impl
│   │   ├── platform/              # adaption for various platform
│   │   ├── runtime/               # code for worker and executor
│   │   ├── scheduler/             # code for batch and pd scheduler
│   │   └── util/
│   ├── function_call              # code for tool call parser
│   ├── models/                    # models impl
│   ├── parser/                    # parser reasoning
│   ├── processors/                # code for vlm pre-processing
│   ├── proto/                     # communication protocol
│   ├── pybind/                    # code for python bind
|   └── server/                    # xLLM server
├── examples/                      # examples of calling xLLM
├── tools/                         # code for npu time generations
└── xllm.cpp                       # entrypoint of xLLM


