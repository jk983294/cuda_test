#include <cstdint>

struct MyCuda {
    ~MyCuda();
    void init(int N_);
    void cuda_entry(const int * a, const int * b, int * c, int N);

    int N{0};
    uint64_t bytes{0};
    int *d_a{nullptr};
    int *d_b{nullptr};
    int *d_c{nullptr};
};

