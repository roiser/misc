#ifndef HELPERS_H
#define HELPERS_H

void make_json(const std::vector<float> &cub_t, const std::vector<float> &dev_t,
               int ncol, int nevt, int threads, int blocks) {
  std::ofstream json;
  int singledouble = 32;
#if defined(DOUBLEPRECISION)
  singledouble = 64;
#endif
  std::string filename = std::to_string(singledouble) + "_" +
                         std::to_string(ncol) + "_" + std::to_string(threads) +
                         "_" + std::to_string(blocks) + ".json";
  json.open(filename);
  double cuavg = std::reduce(cub_t.begin(), cub_t.end(), 0.0) / cub_t.size(),
         cumin = *std::min_element(cub_t.begin(), cub_t.end()),
         cumax = *std::max_element(cub_t.begin(), cub_t.end()),
         oravg = std::reduce(dev_t.begin(), dev_t.end(), 0.0) / dev_t.size(),
         ormin = *std::min_element(dev_t.begin(), dev_t.end()),
         ormax = *std::max_element(dev_t.begin(), dev_t.end());
  json << "{" << std::endl
       << "  \"precision\": " << singledouble << "," << std::endl
       << "  \"numevents\": " << nevt << "," << std::endl
       << "  \"numcolors\": " << ncol << "," << std::endl
       << "  \"numblocks\": " << blocks << "," << std::endl
       << "  \"numthreads\": " << threads << "," << std::endl
       << "  \"cublas\": {" << std::endl
       << "    \"avg\": " << cuavg << "," << std::endl
       << "    \"min\": " << cumin << "," << std::endl
       << "    \"max\": " << cumax << std::endl
       << "  }," << std::endl
       << "  \"device\": {" << std::endl
       << "    \"avg\": " << oravg << "," << std::endl
       << "    \"min\": " << ormin << "," << std::endl
       << "    \"max\": " << ormax << std::endl
       << "  }" << std::endl
       << "}" << std::endl;
  json.close();
  std::cout << std::endl << "factor: " << oravg / cuavg << std::endl;
}

#endif // HELPERS_H
