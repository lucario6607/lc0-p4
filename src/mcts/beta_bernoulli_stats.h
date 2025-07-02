--- START OF FILE beta_bernoulli_stats.h ---

#pragma once

#include <random>
#include <cmath>
#include <algorithm>
#include <limits>

namespace lczero {

class BetaBernoulliStats {
public:
    BetaBernoulliStats(float alpha_prior = 1.0f, float beta_prior = 1.0f)
        : alpha_(alpha_prior), beta_(beta_prior) {}

    void Update(float value) {
        // Convert [-1, 1] to [0, 1] win probability.
        float p = (value + 1.0f) / 2.0f;
        alpha_ += p;
        beta_ += (1.0f - p);
    }

    float Sample(std::mt19937& rng) const {
        // Handle small alpha/beta by uniform sampling.
        if (alpha_ < std::numeric_limits<float>::epsilon() || beta_ < std::numeric_limits<float>::epsilon()) {
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            return dist(rng);
        }
        std::gamma_distribution<float> dist_alpha(alpha_, 1.0f);
        std::gamma_distribution<float> dist_beta(beta_, 1.0f);
        float x = dist_alpha(rng);
        float y = dist_beta(rng);
        return x / (x + y);
    }

    float GetMean() const {
        return alpha_ / (alpha_ + beta_);
    }

    float GetVariance() const {
        float sum = alpha_ + beta_;
        if (sum == 0.0f) return 0.0f; // Avoid division by zero
        return (alpha_ * beta_) / (sum * sum * (sum + 1.0f));
    }

    float GetAlpha() const {
        return alpha_;
    }

    float GetBeta() const {
        return beta_;
    }

private:
    float alpha_;
    float beta_;
};

} // namespace lczero
