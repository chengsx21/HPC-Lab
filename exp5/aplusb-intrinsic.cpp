#include "aplusb.h"
#include <x86intrin.h>

void a_plus_b_intrinsic(float* a, float* b, float* c, int n) {
    for (int idx = 0; idx < n; idx += 8) {
        auto aa = _mm256_loadu_ps(a + idx);
        auto bb = _mm256_loadu_ps(b + idx);
        auto cc = _mm256_add_ps(aa, bb);
        _mm256_storeu_ps(c + idx, cc);
    }
}