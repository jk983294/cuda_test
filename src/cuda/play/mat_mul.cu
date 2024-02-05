#include <getopt.h>
#include <omp.h>
#include <chrono>
#include <iostream>
#include <random>

using namespace std;
using namespace std::chrono;

__global__ void matrixMul(const double* a, double* b, uint64_t n_col, uint64_t len) {
    // Compute each thread's global row and column index
    uint64_t row = blockIdx.y * blockDim.y + threadIdx.y;
    uint64_t col = blockIdx.x * blockDim.x + threadIdx.x;

    // Iterate over row, and down column
    if (row < n_col && col < n_col) {
        double val = 0;
        for (uint64_t k = 0; k < len; k++) {
            val += a[row * len + k] * a[col * len + k];
        }
        b[row * n_col + col] = val;
    }
}

int main(int argc, char** argv) {
    constexpr uint64_t n_col = 160;
    constexpr uint64_t n_ukey = 500;
    constexpr uint64_t n_tick = 240;
    constexpr uint64_t len = n_ukey * n_tick;
    int rounds = 1;

    int opt;
    while ((opt = getopt(argc, argv, "r:h")) != -1) {
        switch (opt) {
            case 'r':
                rounds = std::stoi(optarg);
                break;
            case 'h':
            default:
                return 0;
        }
    }

    int max_threads = omp_get_max_threads();
    omp_set_num_threads(max_threads);

    std::vector<double> datum(n_col * len, NAN);
    {
        random_device rd;  // non-deterministic generator
#pragma omp parallel for
        for (int i = 0; i < n_col; ++i) {
            mt19937 generator(rd());
            normal_distribution<double> uid(0, 1);
            for (int j = 0; j < len; j++) {
                datum[i * len + j] = uid(generator);
            }
        }
    }

    double *d_datum, *d_XTX;
    cudaMalloc(&d_datum, n_col * len * sizeof(double));
    cudaMalloc(&d_XTX, n_col * n_col * sizeof(double));

    uint32_t THREADS = 32;
    uint32_t BLOCKS = (n_col + THREADS - 1) / THREADS;

    // Use dim3 structs for block  and grid dimensions
    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS, BLOCKS);

    std::vector<std::pair<int, int>> m_xx_seq;
    m_xx_seq.reserve(n_col * n_col / 2);
    for (int xi = 0; xi < n_col; xi++) {
        for (int xi1 = 0; xi1 <= xi; xi1++) {
            m_xx_seq.emplace_back(xi, xi1);
        }
    }

    cout << "init finish" << endl;

    std::vector<double> XTX(n_col * n_col, 0);
    std::vector<double> h_XTX(n_col * n_col, 0);
    uint64_t calc_time = 0;
    steady_clock::time_point t0 = steady_clock::now();
    for (int k = 0; k < rounds; k++) {
#pragma omp parallel for
        for (size_t i = 0; i < m_xx_seq.size(); i++) {
            int xi = m_xx_seq[i].first;
            int xi1 = m_xx_seq[i].second;
            const double* vec = datum.data() + xi * len;
            const double* vec1 = datum.data() + xi1 * len;
            double val = 0;
            for (size_t ni = 0; ni < len; ni++) {
                val += vec[ni] * vec1[ni];
            }
            if (xi == xi1) {
                XTX[xi * n_col + xi1] += val;
            } else {
                XTX[xi * n_col + xi1] += val;
                XTX[xi1 * n_col + xi] += val;
            }
        }

        cudaMemcpy(d_datum, datum.data(), n_col * len * sizeof(double), cudaMemcpyHostToDevice);
        steady_clock::time_point t0_0 = steady_clock::now();
        matrixMul<<<blocks, threads>>>(d_datum, d_XTX, n_col, len);
        steady_clock::time_point t0_1 = steady_clock::now();
        calc_time += nanoseconds{t0_1 - t0_0}.count();
        cudaMemcpy(h_XTX.data(), d_XTX, n_col * n_col * sizeof(double), cudaMemcpyDeviceToHost);
    }
    steady_clock::time_point t1 = steady_clock::now();
    auto t = nanoseconds{t1 - t0}.count();
    double avg_t = (double)t / 1000. / 1000. / 1000. / rounds;
    double avg_t1 = (double)calc_time / 1000. / 1000. / 1000. / rounds;

    uint64_t mismatch = 0;
    for (size_t i = 0; i < n_col; i++) {
        for (size_t j = 0; j < n_col; j++) {
            double a = XTX[i * n_col + j];
            double b = h_XTX[i * n_col + j];
            if (std::abs(a - b) > 1e-6) {
                mismatch++;
                // printf("mismatch %f %f %zu %zu\n", a, b, i, j);
            }
        }
    }

    printf("r=%d, mismatch=%zu, %f,%f,%f,%f\n", rounds, mismatch, avg_t, avg_t1, avg_t1 / avg_t, h_XTX[0]);
    return 0;
}
