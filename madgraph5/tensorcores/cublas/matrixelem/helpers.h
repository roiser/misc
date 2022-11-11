#ifndef HELPERS_H
#define HELPERS_H

void make_json(const std::vector<float> &cublas_t,
               const std::vector<float> &device_t, int ncol, int nevt,
               int threads, int blocks) {
  std::ofstream json;
  int singledouble = 32;
#if defined(DOUBLEPRECISION)
  singledouble = 64;
#endif
  std::string filename = std::to_string(singledouble) + "_" +
                         std::to_string(ncol) + "_" + std::to_string(threads) +
                         "_" + std::to_string(blocks) + ".json";
  json.open(filename);
  // clang-format off
  json << "{" << std::endl
       << "  \"precision\": " << singledouble << "," << std::endl
       << "  \"numevents\": " << nevt << "," << std::endl
       << "  \"numcolors\": " << ncol << "," << std::endl
       << "  \"numblocks\": " << blocks << "," << std::endl
       << "  \"numthreads\": " << threads << "," << std::endl
       << "  \"cublas\": {" << std::endl
       << "    \"avg\": " << std::reduce(cublas_t.begin(), cublas_t.end(), 0.0) / cublas_t.size() << "," << std::endl
       << "    \"min\": " << *std::min_element(cublas_t.begin(), cublas_t.end()) << "," << std::endl
       << "    \"max\": " << *std::max_element(cublas_t.begin(), cublas_t.end()) << std::endl
       << "  }," << std::endl
       << "  \"device\": {" << std::endl
       << "    \"avg\": " << std::reduce(device_t.begin(), device_t.end(), 0.0) / device_t.size() << "," << std::endl
       << "    \"min\": " << *std::min_element(device_t.begin(), device_t.end()) << "," << std::endl
       << "    \"max\": " << *std::max_element(device_t.begin(), device_t.end()) << std::endl
       << "  }" << std::endl
       << "}" << std::endl;
  // clang-format on
}

#endif // HELPERS_H
