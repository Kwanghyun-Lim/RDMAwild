#include "utils.hpp"

#include <cblas.h>
#define _USE_MATH_DEFINES  // this is for PI in cmath
#include <cmath>
#include <iostream>
#include <random>

void utils::zero_arr(double* arr, const size_t size) {
    for(size_t i = 0; i < size; ++i, ++arr) {
        *arr = 0;
    }
}

void utils::softmax(double* x, const size_t m, const size_t n) {
    for(size_t i = 0; i < n; ++i) {
        double* y = x + i;
        double sum_exp = 0;

        double max;
        for(size_t j = 0; j < m; ++j, y += n) {
            if(j == 0) {
                max = *y;
            } else {
                if(*y > max) {
                    max = *y;
                }
            }
        }

        y = x + i;
        for(size_t j = 0; j < m; ++j, y += n) {
            *y -= max;
        }

        y = x + i;
        for(size_t j = 0; j < m; ++j, y += n) {
            *y = std::exp(*y);
            sum_exp += *y;
        }

        y = x + i;
        for(size_t j = 0; j < m; ++j, y += n) {
            *y /= sum_exp;
            // std::cout << *y << " ";
        }
    }
}

void utils::submatrix_multiply(CBLAS_TRANSPOSE TransA,
                               CBLAS_TRANSPOSE TransB,
                               double* A, double* B, double* C,
                               int ai, int aj, int bi, int bj,
                               int m, int n, int k, int lda, int ldb,
                               double alpha, double beta) {
    cblas_dgemm(CblasRowMajor, TransA, TransB,
                m, n, k, alpha,
                A + (ai * lda) + aj, lda,
                B + (bi * ldb) + bj, ldb,
                beta, C, n);
}

std::string utils::fullpath(const std::string& dir, const std::string& filename) {
    return dir + "/" + filename;
}

utils::dataset::dataset(const std::string data_path, const uint32_t num_parts,
                        const uint32_t part_num) : data_path(data_path), num_parts(num_parts), part_num(part_num) {
    // std::cout << "[dataset] data_path " << data_path << std::endl;
}
