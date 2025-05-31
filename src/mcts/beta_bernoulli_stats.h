#pragma once

#include <random>
#include <vector> // Added for std::vector
#include <cmath>
#include <algorithm>
#include <limits>
#include <numeric> // Added for std::accumulate (though commented out in provided snippet, good to have for GetPosteriorMean)

// Forward declarations
class Node; // Assuming Node will be defined elsewhere, e.g. "mcts/node.h"
class Move; // Assuming Move will be defined elsewhere, e.g. "chess/move.h" or similar

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
        if (x + y < std::numeric_limits<float>::epsilon()) return 0.5f; // Avoid division by zero if both are ~0
        return x / (x + y);
    }

    float GetMean() const {
        if (alpha_ + beta_ < std::numeric_limits<float>::epsilon()) return 0.5f; // Avoid division by zero
        return alpha_ / (alpha_ + beta_);
    }

    float GetVariance() const {
        float sum = alpha_ + beta_;
        if (sum < std::numeric_limits<float>::epsilon()) return 0.0f; // Avoid division by zero
        if (sum + 1.0f < std::numeric_limits<float>::epsilon()) return 0.0f; // Avoid division by zero
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

// Models uncertainty in position value estimates
class ValueUncertaintyTracker {
public:
    ValueUncertaintyTracker() : sum_(0.0f), sum_squared_(0.0f), count_(0) {}

    void Update(float value) {
        sum_ += value;
        sum_squared_ += value * value;
        count_++;
    }

    float GetMean() const {
        return count_ > 0 ? sum_ / count_ : 0.0f;
    }

    float GetVariance() const {
        if (count_ < 2) return 1.0f;  // High uncertainty with few samples
        float mean = GetMean();
        // Ensure count_ is not zero before division, though count_ < 2 handles this for positive counts.
        if (count_ == 0) return 1.0f;
        return std::max(0.0f, (sum_squared_ / count_) - (mean * mean)); // Variance cannot be negative
    }

    float GetStandardError() const {
        if (count_ < 2) return 1.0f;
         // Ensure count_ is not zero before division
        if (count_ == 0) return 1.0f;
        float variance = GetVariance();
        return std::sqrt(std::max(0.0f, variance / count_)); // Ensure non-negative before sqrt
    }

    // Sample from approximate posterior (Gaussian approximation)
    float SampleValue(std::mt19937& rng) const {
        float mean = GetMean();
        float std_err = GetStandardError();

        // Ensure std_err is positive and not too small to avoid issues with normal_distribution
        if (std_err < 1e-5f) { // Using a slightly larger epsilon for practical purposes
            return mean;
        }
        std::normal_distribution<float> dist(mean, std_err);
        return dist(rng);
    }

    int GetCount() const { return count_; }

private:
    float sum_;
    float sum_squared_;
    int count_;
};

// Bayesian policy updater using Dirichlet distribution
class BayesianPolicyTracker {
public:
    // Default constructor for cases where concentration is set later
    BayesianPolicyTracker() : concentration_(10.0f) {}

    explicit BayesianPolicyTracker(float concentration)
        : concentration_(concentration) {}

    void Initialize(const std::vector<float>& nn_policy) {
        alpha_params_.clear();
        alpha_params_.reserve(nn_policy.size());

        for (float p : nn_policy) {
            // Convert policy probability to Dirichlet parameter
            alpha_params_.push_back(std::max(0.0f, p) * concentration_ + 0.1f);  // Ensure p is not negative, small epsilon to avoid zeros
        }
    }

    void UpdateBestMove(size_t move_index, float confidence = 1.0f) {
        if (move_index < alpha_params_.size()) {
            alpha_params_[move_index] += confidence;
        }
    }

    std::vector<float> SamplePolicy(std::mt19937& rng) const {
        if (alpha_params_.empty()) return {};

        std::vector<float> samples;
        samples.reserve(alpha_params_.size());

        // Sample from Gamma distributions (Dirichlet is normalized Gammas)
        float sum = 0.0f;
        for (float alpha : alpha_params_) {
            // Ensure alpha is positive for gamma_distribution
            float safe_alpha = std::max(alpha, 1e-5f); // Using a slightly larger epsilon
            std::gamma_distribution<float> gamma(safe_alpha, 1.0f);
            float sample = gamma(rng);
            samples.push_back(sample);
            sum += sample;
        }

        // Normalize to get valid probability distribution
        if (sum > 1e-5f) { // Avoid division by zero or very small sum
            for (float& sample : samples) {
                sample /= sum;
            }
        } else if (!samples.empty()) {
            // Fallback: uniform distribution if sum is too small
            float uniform_prob = 1.0f / samples.size();
            for (float& sample : samples) {
                sample = uniform_prob;
            }
        }

        return samples;
    }

    std::vector<float> GetPosteriorMean() const {
        if (alpha_params_.empty()) return {};

        float sum_alpha = 0.0f;
        for (float alpha : alpha_params_) { // Manual sum to avoid accumulate if <numeric> is problematic
            sum_alpha += alpha;
        }

        std::vector<float> means;
        means.reserve(alpha_params_.size());

        if (sum_alpha > 1e-5f) { // Avoid division by zero or very small sum
            for (float alpha : alpha_params_) {
                means.push_back(alpha / sum_alpha);
            }
        } else if (!alpha_params_.empty()) {
            // Fallback: uniform distribution if sum_alpha is too small
            float uniform_prob = 1.0f / alpha_params_.size();
            for (size_t i = 0; i < alpha_params_.size(); ++i) { // Loop to fill with uniform_prob
                 means.push_back(uniform_prob);
            }
        }

        return means;
    }

    void SetConcentration(float concentration) {
        concentration_ = concentration;
    }

    float GetConcentration() const { // Added getter for concentration
        return concentration_;
    }

private:
    std::vector<float> alpha_params_;
    float concentration_;
};

} // namespace lczero
