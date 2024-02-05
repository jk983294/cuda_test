#include <algorithm>
#include <cassert>
#include <iostream>
#include <my_cuda.h>
#include <my_host.h>

void verify_result(std::vector<int> &a, std::vector<int> &b,
                   std::vector<int> &c) {
  for (uint64_t i = 0; i < a.size(); i++) {
    assert(c[i] == a[i] + b[i]);
  }
}


int main() {
  // Array size of 2^16 (65536 elements)
  constexpr int N = 1 << 16;

  // Vectors for holding the host-side (CPU-side) data
  std::vector<int> a;
  a.reserve(N);
  std::vector<int> b;
  b.reserve(N);
  std::vector<int> c;
  c.reserve(N);

  // Initialize random numbers in each array
  for (int i = 0; i < N; i++) {
    a.push_back(rand() % 100);
    b.push_back(rand() % 100);
  }

  MyCuda mc;
  mc.init(N);
  mc.cuda_entry(a.data(), b.data(), c.data(), N);

  // Check result for errors
  verify_result(a, b, c);

  std::cout << "COMPLETED SUCCESSFULLY\n";
  return 0;
}