#ifndef TWIST_CLI_MODEL_GEN_H
#define TWIST_CLI_MODEL_GEN_H

#include "simdjson.h"
#include "tsl/ordered_map.h"
#include <utility>
#include <vector>

std::vector<std::pair<int, double>> load_diffusion(const simdjson::dom::element &doc);
tsl::ordered_map<std::string, std::pair<double, double>> load_transforms(const simdjson::dom::element &doc);
void compile_model(const char *path, const char *compiler, const bool source_only);

#endif // TWIST_CLI_MODEL_GEN_H