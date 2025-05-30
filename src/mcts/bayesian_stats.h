#pragma once

#include <random>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric> // Required for std::accumulate

// Forward declaration if Move is a class/struct from elsewhere
// class Move; // Assuming Move is defined elsewhere, e.g. chess/move.h

namespace lczero { // Assuming these classes belong in the lczero namespace

// Models uncertainty in position value estimates
class ValueUncertaintyTracker {
public:
    ValueUncertaintyTracker() : sum_(0.0f), sum_squared_(0.0f), count_(0) {}

    void Update(float value);

    float GetMean() const;

    float GetVariance() const;

    float GetStandardError() const;

    // Sample from approximate posterior (Gaussian approximation)
    float SampleValue(std::mt19937& rng) const;

    int GetCount() const { return count_; }

private:
    float sum_;
    float sum_squared_;
    int count_;
};

// Bayesian policy updater using Dirichlet distribution
class BayesianPolicyTracker {
public:
    explicit BayesianPolicyTracker(float concentration = 10.0f)
        : concentration_(concentration) {}

    void Initialize(const std::vector<float>& nn_policy);

    void UpdateBestMove(size_t move_index, float confidence = 1.0f);

    std::vector<float> SamplePolicy(std::mt19937& rng) const;

    std::vector<float> GetPosteriorMean() const;

    // Getter for alpha_params_ size for checking if initialized
    bool IsInitialized() const { return !alpha_params_.empty(); }

private:
    std::vector<float> alpha_params_;
    float concentration_;
};

} // namespace lczero
