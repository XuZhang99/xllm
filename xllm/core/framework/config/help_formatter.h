/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <gflags/gflags.h>

#include <iostream>
#include <sstream>
#include <string>

#include "core/framework/config/xllm_config.h"

namespace xllm {

class HelpFormatter {
 public:
  static std::string generate_help() {
    std::ostringstream oss;

    oss << "USAGE: xllm --model <PATH> [OPTIONS]\n\n";

    oss << "REQUIRED OPTIONS:\n";
    oss << "  --model <PATH>: Path to the model directory. This is "
           "the only required flag.\n\n";

    oss << "HELP OPTIONS:\n";
    oss << "  -h, --help: Display this help message and exit.\n\n";

    // Print flags(options) by category
    for (const OptionCategory& option_category :
         XllmConfig::option_categories()) {
      std::ostringstream category_oss;

      for (const std::string& option_name : option_category.option_names) {
        google::CommandLineFlagInfo option_info;
        if (google::GetCommandLineFlagInfo(option_name.c_str(), &option_info)) {
          category_oss << "  --" << option_info.name;
          if (!option_info.description.empty()) {
            category_oss << ": " << option_info.description;
          }
          category_oss << "\n";
        }
      }

      std::string category_help = category_oss.str();
      if (!category_help.empty()) {
        oss << option_category.category_name << ":\n";
        oss << category_help << "\n";
      }
    }

    oss << "For more information and all available options, visit:\n";
    oss << "  https://github.com/jd-opensource/xllm/blob/main/xllm/core/"
           "framework/config/\n";
    oss << "Documentation: "
           "https://docs.xllm-ai.com/en/cli_reference/\n";

    return oss.str();
  }

  static void print_help() { std::cout << generate_help(); }

  static void print_usage() {
    std::cout << "USAGE: xllm --model <PATH> [OPTIONS]\n";
    std::cout << "Try 'xllm --help' for more information.\n";
  }

  static void print_error(const std::string& error_msg) {
    std::cerr << "Error: " << error_msg << "\n\n";
    print_usage();
  }
};

}  // namespace xllm
