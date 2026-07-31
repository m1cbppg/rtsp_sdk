// Adapted from NVIDIA-AI-IOT/deepstream_lpr_app (MIT).
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

#include "nvdsinfer.h"

namespace {
std::once_flag dictionary_once;
std::vector<std::string> dictionary;
bool dictionary_ok = false;

void load_dictionary() {
  const char* configured = std::getenv("LPR_DICT_PATH");
  const std::string path =
      configured == nullptr ? "/app/models/lpr/ch_lp_characters.txt"
                            : configured;
  std::ifstream input(path);
  std::string line;
  while (std::getline(input, line)) {
    if (!line.empty()) {
      dictionary.push_back(line);
    }
  }
  dictionary_ok = !dictionary.empty();
}
}  // namespace

extern "C" bool NvDsInferParseCustomNVPlate(
    const std::vector<NvDsInferLayerInfo>& output_layers,
    const NvDsInferNetworkInfo& network_info,
    float classifier_threshold,
    std::vector<NvDsInferAttribute>& attributes,
    std::string& attribute_string) {
  std::call_once(dictionary_once, load_dictionary);
  if (!dictionary_ok) {
    return false;
  }

  const int* indices = nullptr;
  const float* confidences = nullptr;
  for (const auto& layer : output_layers) {
    if (layer.isInput) {
      continue;
    }
    if (layer.dataType == NvDsInferDataType::INT32 && indices == nullptr) {
      indices = static_cast<const int*>(layer.buffer);
    } else if (
        layer.dataType == NvDsInferDataType::FLOAT &&
        confidences == nullptr) {
      confidences = static_cast<const float*>(layer.buffer);
    }
  }
  if (indices == nullptr || confidences == nullptr) {
    return false;
  }

  const int blank = static_cast<int>(dictionary.size());
  const int sequence_length = static_cast<int>(network_info.width / 4);
  int previous = -1;
  float confidence_sum = 0.0F;
  unsigned int characters = 0;
  attribute_string.clear();
  for (int index = 0; index < sequence_length; ++index) {
    const int value = indices[index];
    if (value < 0 || value > blank) {
      previous = value;
      continue;
    }
    if (value != previous && value != blank) {
      attribute_string += dictionary[static_cast<std::size_t>(value)];
      confidence_sum += confidences[index];
      ++characters;
    }
    previous = value;
  }
  const float mean_confidence =
      characters == 0 ? 0.0F : confidence_sum / characters;
  if (characters < 3 || mean_confidence < classifier_threshold) {
    attribute_string.clear();
    return true;
  }

  NvDsInferAttribute attribute{};
  attribute.attributeIndex = 0;
  attribute.attributeValue = 1;
  attribute.attributeConfidence = mean_confidence;
  attribute.attributeLabel = ::strdup(attribute_string.c_str());
  attributes.push_back(attribute);
  return true;
}
