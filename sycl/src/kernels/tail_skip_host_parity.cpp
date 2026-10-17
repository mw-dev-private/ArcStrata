#include "strata/core/tail_skip_policy.hpp"
#include <array>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

static bool reference(const std::vector<int32_t>& ids, int tokens, int expert, bool resident, int rank) {
    if (resident || rank <= 0 || rank >= 10) return false;
    std::array<bool, 10> occupied{};
    for (int t = 0; t < tokens; ++t)
        for (int r = 0; r < 10; ++r)
            occupied[r] = occupied[r] || ids[t * 10 + r] == expert;
    bool found = false;
    for (int r = 0; r < 10; ++r) {
        if (occupied[r] && r < rank) return false;
        found = found || occupied[r];
    }
    return found;
}
int main() {
    std::mt19937 rng(42);
    int checks = 0;
    for (int tokens : {1, 2, 5, 16, 32}) {
        for (int trial = 0; trial < 100; ++trial) {
            std::vector<int32_t> ids(tokens * 10);
            for (auto& e : ids) e = rng() % 64;
            // Directed boundaries: resident tail, all-tail, one head occurrence, absent-in-other-token.
            ids[7] = 60; ids[8] = 61; ids[9] = 62;
            if (tokens > 1) { ids[10] = 61; ids[17] = 60; }
            for (int rank = 0; rank <= 10; ++rank)
                for (int expert = 0; expert <= 64; ++expert) {
                    const bool resident = expert % 3 == 2;
                    const bool got = strata::core::tail_skip_candidate(ids.data(), tokens, 10, expert, resident, rank);
                    if (got != reference(ids, tokens, expert, resident, rank)) throw std::runtime_error("rank mismatch");
                    ++checks;
                }
            // Independent weighted-sum check: null skipped rows, preserve original router weights bytewise.
            std::vector<float> weights(ids.size()), rows(ids.size());
            for (size_t i = 0; i < ids.size(); ++i) { weights[i] = float(10 - i % 10) / 55; rows[i] = float(ids[i] + 1); }
            const auto saved = weights;
            for (size_t i = 0; i < ids.size(); ++i)
                if (strata::core::tail_skip_candidate(ids.data(), tokens, 10, ids[i], ids[i] % 3 == 2, 7)) rows[i] = 0;
            if (std::memcmp(weights.data(), saved.data(), weights.size() * sizeof(float))) throw std::runtime_error("weights changed");
            for (int t = 0; t < tokens; ++t) {
                double got = 0, expected = 0;
                for (int r = 0; r < 10; ++r) {
                    const int i = t * 10 + r;
                    got += double(weights[i]) * rows[i];
                    if (!reference(ids, tokens, ids[i], ids[i] % 3 == 2, 7)) expected += double(saved[i]) * float(ids[i] + 1);
                }
                if (got != expected) throw std::runtime_error("zero contribution mismatch");
            }
        }
    }
    std::cout << "PASS: " << checks << " rank decisions; weighted sums and unchanged weights\n";
}
