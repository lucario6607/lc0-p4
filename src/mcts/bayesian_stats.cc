#include "bayesian_stats.h" // Self-header

namespace lczero {

// ValueUncertaintyTracker methods
void ValueUncertaintyTracker::Update(float value) {
    sum_ += value;
    sum_squared_ += value * value;
    count_++;
}

float ValueUncertaintyTracker::GetMean() const {
    return count_ > 0 ? sum_ / count_ : 0.0f;
}

float ValueUncertaintyTracker::GetVariance() const {
    if (count_ < 2) return 1.0f;  // High uncertainty with few samples
    float mean = GetMean();
    return (sum_squared_ / count_) - (mean * mean);
}

float ValueUncertaintyTracker::GetStandardError() const {
    if (count_ < 2) return 1.0f; // High uncertainty for low counts
    // Ensure variance is non-negative before sqrt
    float variance = GetVariance();
    if (variance < 0.0f) variance = 0.0f;
    return std::sqrt(variance / count_);
}

float ValueUncertaintyTracker::SampleValue(std::mt19937& rng) const {
    float mean = GetMean();
    float std_err = GetStandardError();

    // Ensure std_err is positive for normal_distribution
    if (std_err <= 0.0f && count_ > 0) { // if count_ is 0, mean is 0, std_err is 1.0 (or should be if GetStandardError handles count < 2 properly)
        // If std_err is zero (or negative due to float precision with very low variance),
        // and we have samples, it implies certainty. Return the mean.
        // Or, if std_err is problematic, return mean to avoid issues with distribution.
        return mean;
    }
    if (count_ == 0) { // No samples yet, std_err will be 1.0 by default from GetStandardError
         std_err = 1.0f; // Default high uncertainty if no samples
    }


    std::normal_distribution<float> dist(mean, std_err);
    return dist(rng);
}

// BayesianPolicyTracker methods
void BayesianPolicyTracker::Initialize(const std::vector<float>& nn_policy) {
    alpha_params_.clear();
    alpha_params_.reserve(nn_policy.size());

    for (float p : nn_policy) {
        // Convert policy probability to Dirichlet parameter
        alpha_params_.push_back(p * concentration_ + 0.1f);  // Small epsilon to avoid zeros
    }
}

void BayesianPolicyTracker::UpdateBestMove(size_t move_index, float confidence) {
    if (move_index < alpha_params_.size()) {
        alpha_params_[move_index] += confidence;
    }
}

std::vector<float> BayesianPolicyTracker::SamplePolicy(std::mt19937& rng) const {
    if (alpha_params_.empty()) return {};

    std::vector<float> samples;
    samples.reserve(alpha_params_.size());

    float sum = 0.0f;
    for (float alpha : alpha_params_) {
        // Ensure alpha is positive for gamma_distribution
        float current_alpha = alpha > 0.0f ? alpha : 0.1f; // Use small epsilon if alpha is not positive
        std::gamma_distribution<float> gamma(current_alpha, 1.0f);
        float sample = gamma(rng);
        samples.push_back(sample);
        sum += sample;
    }

    if (sum > 0.0f) {
        for (float& sample : samples) {
            sample /= sum;
        }
    } else if (!samples.empty()) {
        // If sum is zero (e.g. all samples were zero), distribute uniformly as a fallback
        // This case should be rare with positive alpha values.
        float uniform_prob = 1.0f / samples.size();
        for (float& sample : samples) {
            sample = uniform_prob;
        }
    }

    return samples;
}

std::vector<float> BayesianPolicyTracker::GetPosteriorMean() const {
    if (alpha_params_.empty()) return {};

    float sum_alphas = std::accumulate(alpha_params_.begin(), alpha_params_.end(), 0.0f);

    if (sum_alphas == 0.0f && !alpha_params_.empty()) { // Should not happen if alphas are initialized with epsilon
        std::vector<float> uniform_means(alpha_params_.size(), 1.0f / alpha_params_.size());
        return uniform_means;
    }
    if (sum_alphas == 0.0f && alpha_params_.empty()) { // Actually empty
        return {};
    }

    std::vector<float> means;
    means.reserve(alpha_params_.size());

    for (float alpha : alpha_params_) {
        means.push_back(alpha / sum_alphas);
    }

    return means;
}

} // namespace lczero
