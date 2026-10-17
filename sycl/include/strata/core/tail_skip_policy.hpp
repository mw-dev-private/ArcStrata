#pragma once
#include <cstdint>
namespace strata::core {
// An absent token occurrence has no contribution. Any occurrence above the tail vetoes the skip.
inline bool tail_skip_candidate(const int32_t* ids, int64_t tokens, int64_t k,
                                int32_t expert, bool resident, int rank) {
    if (resident || rank <= 0 || rank >= 10 || k != 10 || tokens <= 0) return false;
    bool seen = false;
    for (int64_t t = 0; t < tokens; ++t)
        for (int64_t r = 0; r < k; ++r)
            if (ids[t * k + r] == expert) {
                seen = true;
                if (r < rank) return false;
            }
    return seen;
}
}
