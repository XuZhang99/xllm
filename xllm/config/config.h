#include <nlohmann/json.hpp>
#include <string_view>

namespace xllm {

bool is_flag_default(const std::string& name) {
  gflags::CommandLineFlagInfo info;
  CHECK(gflags::GetCommandLineFlagInfo(name.c_str(), &info));
  return info.is_default;
}

inline constexpr std::string_view kQwen3Config =
#include "qwen3/config.json.inc"
    ;

// nlohmann::json qwen3_config = nlohmann::json::parse(kQwen3Config);

std::unordered_map<std::string, std::string_view> config_map = {
    {"qwen3", kQwen3Config},
};

nlohmann::json get_config(const std::string& model_type) {
  if (config_map.find(model_type) == config_map.end()) {
    LOG(FATAL) << "Model type " << model_type << " not found";
    return nlohmann::json();
  }
  JsonReader reader;
  reader.parse_text(config_map.at(model_type));
  return reader.data();
}

}  // namespace xllm