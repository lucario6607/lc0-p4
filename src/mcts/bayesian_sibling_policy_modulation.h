// bayesian_sibling_policy_modulation.h
#ifndef LCZERO_BAYESIAN_SIBLING_POLICY_MODULATION_H_
#define LCZERO_BAYESIAN_SIBLING_POLICY_MODULATION_H_

#include <vector>
#include <cmath>
#include <unordered_map>
#include "../neural/cache.h"
#include "node.h"

namespace lczero {

struct BayesianStats {
    float alpha = 1.0f;  // Beta distribution parameters for wins
    float beta = 1.0f;   // Beta distribution parameters for losses
    int update_count = 0;
    
    void Update(float outcome) {
        // Convert Q-value outcome to win/loss probability
        if (outcome > 0.0f) {
            alpha += outcome;
            beta += (1.0f - outcome);
        } else {
            // Handle negative Q-values (losses from perspective)
            alpha += (1.0f + outcome) / 2.0f;
            beta += (1.0f - outcome) / 2.0f;
        }
        update_count++;
    }
    
    float GetMean() const { 
        return alpha / (alpha + beta); 
    }
    
    float GetVariance() const { 
        float sum = alpha + beta;
        return (alpha * beta) / (sum * sum * (sum + 1.0f));
    }
    
    float GetConfidenceInterval(float confidence = 0.95f) const {
        if (update_count < 2) return 1.0f; // Maximum uncertainty for few samples
        
        float std_dev = std::sqrt(GetVariance());
        // Approximate inverse normal CDF
        float z_score = (confidence >= 0.95f) ? 1.96f : 1.64f; // 95% or 90%
        return 2.0f * z_score * std_dev;
    }
    
    float GetUncertaintyMeasure() const {
        // Higher values indicate more uncertainty
        float ci_width = GetConfidenceInterval();
        float sample_uncertainty = 1.0f / (1.0f + std::log(1.0f + update_count));
        return ci_width * sample_uncertainty;
    }
    
    // Thompson Sampling for exploration
    float SampleThompson() const {
        // Simple approximation of Beta distribution sampling
        float mean = GetMean();
        float variance = GetVariance();
        if (variance < 1e-6f) return mean;
        
        // Use normal approximation for Beta when alpha, beta > 5
        if (alpha > 5.0f && beta > 5.0f) {
            float std_dev = std::sqrt(variance);
            // Generate pseudo-random sample (in practice, use proper RNG)
            static thread_local float last_random = 0.5f;
            last_random = std::fmod(last_random * 1.618033988749f, 1.0f); // Golden ratio for pseudo-randomness
            float z = (last_random - 0.5f) * 3.0f; // Approximate normal
            return std::max(0.0f, std::min(1.0f, mean + z * std_dev));
        }
        
        return mean;
    }
};

struct BayesianSiblingPolicyModulationParams {
    // Base parameters (same as original)
    bool enabled = false;
    float q_diff_threshold = 0.1f;
    float policy_boost_factor = 1.2f;
    float policy_damp_factor = 0.8f;
    float min_visits_for_modulation = 5.0f;
    float confidence_weight = 0.5f;
    float temporal_decay = 0.95f;
    bool use_policy_groups = true;
    float high_policy_threshold = 0.1f;
    bool adaptive_thresholds = true;
    
    // Bayesian-specific parameters
    bool use_bayesian_confidence = true;
    float uncertainty_exploration_weight = 0.4f;  // How much to boost uncertain moves
    float bayesian_update_decay = 0.95f;          // Decay older observations
    float min_confidence_for_dampening = 0.8f;    // Only dampen if we're confident
    bool use_thompson_sampling = false;           // Use Thompson sampling for exploration
    float credible_interval_threshold = 0.3f;    // Width threshold for high uncertainty
    float bayesian_prior_strength = 2.0f;        // Strength of prior beliefs (alpha + beta)
};

class BayesianSiblingPolicyModulator {
public:
    explicit BayesianSiblingPolicyModulator(const BayesianSiblingPolicyModulationParams& params)
        : params_(params) {}

    // Main function with Bayesian enhancements
    float CalculateEffectivePolicy(const Node* parent, const Edge& edge) const;

    // Update Bayesian statistics
    void UpdateBayesianStats(const Node* parent, const Edge& edge, float outcome);
    void UpdateModulationStats(const Node* parent);

private:
    struct BayesianSiblingGroup {
        std::vector<const Edge*> edges;
        float avg_q = 0.0f;
        float total_policy = 0.0f;
        int total_visits = 0;
        BayesianStats group_stats;
        float uncertainty_score = 0.0f;
    };

    // Core Bayesian modulation methods
    float CalculateBayesianConfidenceModulation(const Node* parent, const Edge& edge) const;
    float CalculateUncertaintyExplorationBonus(const Node* parent, const Edge& edge) const;
    float CalculateThompsonSamplingModulation(const Node* parent, const Edge& edge) const;
    float CalculateBayesianGroupModulation(const Node* parent, const Edge& edge) const;
    
    // Enhanced helper methods
    BayesianStats* GetOrCreateBayesianStats(const Edge& edge) const;
    std::vector<BayesianSiblingGroup> GroupSiblingsByBayesianUncertainty(const Node* parent) const;
    float CalculateBayesianQVariance(const Node* parent) const;
    bool ShouldApplyBayesianModulation(const Node* parent, const Edge& edge) const;
    
    // Bayesian-specific calculations
    float CalculateCredibleIntervalOverlap(const BayesianStats& stats1, const BayesianStats& stats2) const;
    float EstimateProbabilityOfSuperiority(const Edge& edge1, const Edge& edge2) const;
    
    const BayesianSiblingPolicyModulationParams& params_;
    
    // Bayesian state tracking
    mutable std::unordered_map<const Edge*, BayesianStats> edge_bayesian_stats_;
    mutable std::unordered_map<const Node*, std::vector<BayesianSiblingGroup>> cached_bayesian_groups_;
    mutable std::unordered_map<const Node*, float> cached_bayesian_variance_;
};

} // namespace lczero

#endif // LCZERO_BAYESIAN_SIBLING_POLICY_MODULATION_H_
