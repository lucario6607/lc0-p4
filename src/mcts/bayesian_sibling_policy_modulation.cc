// bayesian_sibling_policy_modulation.cc
#include "bayesian_sibling_policy_modulation.h"
#include <algorithm>
#include <numeric>
#include <cmath> // For std::sqrt, std::exp, std::abs, M_PI

#ifndef M_PI // Define M_PI if not defined (e.g., on Windows with MSVC)
#define M_PI 3.14159265358979323846
#endif

namespace lczero {

float BayesianSiblingPolicyModulator::CalculateEffectivePolicy(
    const Node* parent, const Edge& edge) const {
    
    if (!params_.enabled || !parent || parent->GetNumEdges() < 2) {
        return edge.GetP();
    }

    if (!ShouldApplyBayesianModulation(parent, edge)) {
        return edge.GetP();
    }

    float base_policy = edge.GetP();
    float modulation_factor = 1.0f;

    // Apply Bayesian confidence modulation
    if (params_.use_bayesian_confidence) {
        modulation_factor *= CalculateBayesianConfidenceModulation(parent, edge);
    }

    // Apply uncertainty exploration bonus
    modulation_factor *= CalculateUncertaintyExplorationBonus(parent, edge);

    // Apply Thompson sampling if enabled
    if (params_.use_thompson_sampling) {
        modulation_factor *= CalculateThompsonSamplingModulation(parent, edge);
    }

    // Apply Bayesian group-based modulation
    if (params_.use_policy_groups) {
        modulation_factor *= CalculateBayesianGroupModulation(parent, edge);
    }

    // Clamp modulation factor
    modulation_factor = std::max(0.1f, std::min(4.0f, modulation_factor));

    return base_policy * modulation_factor;
}

float BayesianSiblingPolicyModulator::CalculateBayesianConfidenceModulation(
    const Node* parent, const Edge& edge) const {
    
    BayesianStats* stats = GetOrCreateBayesianStats(edge);
    if (!stats || stats->update_count < 2) return 1.0f;
    
    float confidence_interval_width = stats->GetConfidenceInterval();
    float uncertainty = stats->GetUncertaintyMeasure();
    
    // If we have high uncertainty about this move, explore it more
    if (confidence_interval_width > params_.credible_interval_threshold) {
        return 1.0f + uncertainty * params_.uncertainty_exploration_weight;
    }
    
    // If we're confident this move is poor, dampen it
    float current_q = edge.GetQ(0.0f);
    float mean_estimated_q = stats->GetMean() * 2.0f - 1.0f; // Convert [0,1] back to [-1,1]
    
    if (confidence_interval_width < params_.credible_interval_threshold * 0.5f && 
        mean_estimated_q < current_q - params_.q_diff_threshold &&
        stats->update_count >= params_.min_confidence_for_dampening * 2) {
        return params_.policy_damp_factor;
    }
    
    return 1.0f;
}

float BayesianSiblingPolicyModulator::CalculateUncertaintyExplorationBonus(
    const Node* parent, const Edge& edge) const {
    
    BayesianStats* stats = GetOrCreateBayesianStats(edge);
    if (!stats) return 1.0f;
    
    float uncertainty = stats->GetUncertaintyMeasure();
    
    // Compare uncertainty with siblings
    float avg_sibling_uncertainty = 0.0f;
    int sibling_count = 0;
    
    for (const auto& sibling_edge : parent->Edges()) { // Renamed inner loop variable
        if (&sibling_edge != &edge && sibling_edge.GetN() >= params_.min_visits_for_modulation) {
            BayesianStats* sibling_stats = GetOrCreateBayesianStats(sibling_edge);
            if (sibling_stats) {
                avg_sibling_uncertainty += sibling_stats->GetUncertaintyMeasure();
                sibling_count++;
            }
        }
    }
    
    if (sibling_count == 0) return 1.0f;
    avg_sibling_uncertainty /= sibling_count;
    
    // If this move is more uncertain than siblings, give it an exploration bonus
    if (uncertainty > avg_sibling_uncertainty * 1.2f) {
        float bonus = 1.0f + (uncertainty - avg_sibling_uncertainty) * params_.uncertainty_exploration_weight;
        return std::min(bonus, params_.policy_boost_factor);
    }
    
    return 1.0f;
}

float BayesianSiblingPolicyModulator::CalculateThompsonSamplingModulation(
    const Node* parent, const Edge& edge) const {
    
    BayesianStats* stats = GetOrCreateBayesianStats(edge);
    if (!stats || stats->update_count < 2) return 1.0f;
    
    float thompson_sample = stats->SampleThompson();
    float current_q_normalized = (edge.GetQ(0.0f) + 1.0f) / 2.0f; // Convert to [0,1]
    
    // If Thompson sample suggests this move is better than current estimate, boost it
    if (thompson_sample > current_q_normalized + 0.1f) {
        float boost = 1.0f + (thompson_sample - current_q_normalized) * 0.5f;
        return std::min(boost, params_.policy_boost_factor);
    }
    
    return 1.0f;
}

float BayesianSiblingPolicyModulator::CalculateBayesianGroupModulation(
    const Node* parent, const Edge& edge) const {
    
    auto groups = GroupSiblingsByBayesianUncertainty(parent);
    if (groups.size() < 2) return 1.0f;
    
    // Find which group this edge belongs to
    BayesianSiblingGroup* current_group = nullptr;
    for (auto& group : groups) {
        if (std::find(group.edges.begin(), group.edges.end(), &edge) != group.edges.end()) {
            current_group = &group;
            break;
        }
    }
    
    if (!current_group) return 1.0f;
    
    // High-uncertainty, low-policy group might contain hidden gems
    if (current_group->uncertainty_score > 0.5f && 
        current_group->total_policy < params_.high_policy_threshold) {
        return params_.policy_boost_factor;
    }
    
    // High-policy group with high confidence in poor performance - dampen
    if (current_group->total_policy > params_.high_policy_threshold &&
        current_group->uncertainty_score < 0.3f &&
        current_group->avg_q < -0.1f) { // Assuming avg_q is in [-1, 1]
        return params_.policy_damp_factor;
    }
    
    return 1.0f;
}

BayesianStats* BayesianSiblingPolicyModulator::GetOrCreateBayesianStats(const Edge& edge) const {
    auto it = edge_bayesian_stats_.find(&edge);
    if (it == edge_bayesian_stats_.end()) {
        // Initialize with weak prior
        BayesianStats stats;
        stats.alpha = params_.bayesian_prior_strength / 2.0f;
        stats.beta = params_.bayesian_prior_strength / 2.0f;
        edge_bayesian_stats_[&edge] = stats;
        return &edge_bayesian_stats_.at(&edge); // Use .at() for const correctness if map is const
    }
    return &it->second;
}

std::vector<BayesianSiblingPolicyModulator::BayesianSiblingGroup> 
BayesianSiblingPolicyModulator::GroupSiblingsByBayesianUncertainty(const Node* parent) const {
    
    std::vector<BayesianSiblingGroup> groups;
    BayesianSiblingGroup high_uncertainty_group, low_uncertainty_group;
    
    for (const auto& current_edge : parent->Edges()) { // Renamed inner loop variable
        if (current_edge.GetN() < params_.min_visits_for_modulation) continue;
        
        BayesianStats* stats = GetOrCreateBayesianStats(current_edge);
        float uncertainty = stats ? stats->GetUncertaintyMeasure() : 1.0f;
        
        if (uncertainty > params_.credible_interval_threshold) {
            high_uncertainty_group.edges.push_back(&current_edge);
            high_uncertainty_group.total_policy += current_edge.GetP();
            high_uncertainty_group.total_visits += current_edge.GetN();
            high_uncertainty_group.uncertainty_score += uncertainty;
        } else {
            low_uncertainty_group.edges.push_back(&current_edge);
            low_uncertainty_group.total_policy += current_edge.GetP();
            low_uncertainty_group.total_visits += current_edge.GetN();
            low_uncertainty_group.uncertainty_score += uncertainty;
        }
    }
    
    // Calculate average Q and normalize uncertainty scores
    auto finalize_group = [](BayesianSiblingGroup& group) {
        if (group.edges.empty()) return;
        
        float q_sum = 0.0f;
        for (const auto* edge_ptr : group.edges) { // Corrected: Iterate over pointers
            q_sum += edge_ptr->GetQ(0.0f);
        }
        group.avg_q = q_sum / group.edges.size();
        group.uncertainty_score /= group.edges.size();
    };
    
    finalize_group(high_uncertainty_group);
    finalize_group(low_uncertainty_group);
    
    if (!high_uncertainty_group.edges.empty()) groups.push_back(high_uncertainty_group);
    if (!low_uncertainty_group.edges.empty()) groups.push_back(low_uncertainty_group);
    
    return groups;
}

bool BayesianSiblingPolicyModulator::ShouldApplyBayesianModulation(
    const Node* parent, const Edge& edge) const {
    
    if (edge.GetN() < params_.min_visits_for_modulation) return false;
    
    // Need at least one other sibling with Bayesian stats
    int siblings_with_stats = 0;
    for (const auto& sibling_edge : parent->Edges()) { // Renamed inner loop variable
        if (sibling_edge.GetN() >= params_.min_visits_for_modulation) {
            // Avoid checking the edge against itself if it's part of parent->Edges()
            if (&sibling_edge == &edge) continue; 
            
            BayesianStats* stats = GetOrCreateBayesianStats(sibling_edge);
            if (stats && stats->update_count > 0) {
                siblings_with_stats++;
            }
        }
    }
    
    return siblings_with_stats >= 1;
}

void BayesianSiblingPolicyModulator::UpdateBayesianStats(
    const Node* parent, const Edge& edge, float outcome) {
    
    BayesianStats* stats = GetOrCreateBayesianStats(edge);
    if (stats) {
        // Apply temporal decay to older observations
        stats->alpha *= params_.bayesian_update_decay;
        stats->beta *= params_.bayesian_update_decay;
        
        // Update with new outcome
        stats->Update(outcome);
    }
}

void BayesianSiblingPolicyModulator::UpdateModulationStats(const Node* parent) {
    // Clear caches periodically
    if (cached_bayesian_groups_.size() > 1000) { // Example threshold
        cached_bayesian_groups_.clear();
        // cached_bayesian_variance_.clear(); // This was in the .h but not used in .cc, consider removing if not needed
    }
    
    // Decay old Bayesian statistics to prevent staleness
    for (auto& pair : edge_bayesian_stats_) {
        BayesianStats& stats = pair.second;
        // Consider if this decay is always desired, or configurable
        // if (stats.update_count > 100) { // Original condition
        //    stats.alpha *= 0.99f; 
        //    stats.beta *= 0.99f;
        // }
    }
}

float BayesianSiblingPolicyModulator::EstimateProbabilityOfSuperiority(
    const Edge& edge1, const Edge& edge2) const {
    
    BayesianStats* stats1 = GetOrCreateBayesianStats(edge1);
    BayesianStats* stats2 = GetOrCreateBayesianStats(edge2);
    
    if (!stats1 || !stats2 || stats1->update_count < 1 || stats2->update_count < 1) return 0.5f;
    
    float mean1 = stats1->GetMean();
    float mean2 = stats2->GetMean();
    float var1 = stats1->GetVariance();
    float var2 = stats2->GetVariance();
    
    float diff_mean = mean1 - mean2;
    float diff_var = var1 + var2;
    
    if (diff_var < 1e-7f) { 
        return (diff_mean > 0) ? 1.0f : ((diff_mean < 0) ? 0.0f : 0.5f);
    }
    
    float z_score = diff_mean / std::sqrt(diff_var);
    
    // Standard normal CDF approximation (Abramowitz and Stegun formula 26.2.17)
    float p = 0.2316419f;
    float b1 = 0.319381530f;
    float b2 = -0.356563782f;
    float b3 = 1.781477937f;
    float b4 = -1.821255978f;
    float b5 = 1.330274429f;

    float t = 1.0f / (1.0f + p * std::abs(z_score));
    float pdf_val = (1.0f / std::sqrt(2.0f * M_PI)) * std::exp(-0.5f * z_score * z_score);
    float cdf_val_tail = pdf_val * (b1*t + b2*t*t + b3*t*t*t + b4*t*t*t*t + b5*t*t*t*t*t);

    float prob = (z_score >= 0) ? (1.0f - cdf_val_tail) : cdf_val_tail; // Changed > to >= for z_score = 0 case
    
    return std::max(0.0f, std::min(1.0f, prob));
}

float BayesianSiblingPolicyModulator::CalculateBayesianQVariance(const Node* parent) const {
    if (!parent || parent->GetNumEdges() == 0) return 0.0f;

    float total_weighted_visits = 0; // Using a more descriptive name
    float sum_weighted_q = 0;
    float sum_weighted_q_squared = 0;
    int valid_edges_count = 0; // Count of edges with valid stats

    for (const auto& edge : parent->Edges()) {
        BayesianStats* stats = GetOrCreateBayesianStats(edge);
        // Ensure stats exist and have been updated at least once
        if (stats && stats->update_count > 0) { 
            float mean_q = stats->GetMean(); // Q is in [0,1] from BayesianStats
            // Weight by update_count from BayesianStats for consistency, or edge.GetN()
            float weight = static_cast<float>(stats->update_count); 

            sum_weighted_q += mean_q * weight;
            sum_weighted_q_squared += mean_q * mean_q * weight;
            total_weighted_visits += weight;
            valid_edges_count++;
        }
    }

    if (total_weighted_visits < 1e-6f || valid_edges_count < 2) return 0.0f;

    float overall_mean_q = sum_weighted_q / total_weighted_visits;
    float variance = (sum_weighted_q_squared / total_weighted_visits) - (overall_mean_q * overall_mean_q);
    
    // Ensure variance is not negative due to floating point inaccuracies
    return std::max(0.0f, variance); 
}

float BayesianSiblingPolicyModulator::CalculateCredibleIntervalOverlap(
    const BayesianStats& stats1, const BayesianStats& stats2) const {
    if (stats1.update_count < 1 || stats2.update_count < 1) return 0.0f;

    float mean1 = stats1.GetMean();
    // float var1 = stats1.GetVariance(); // Not directly used in this CI overlap version
    float ci_width1 = stats1.GetConfidenceInterval(0.95f);
    float lower1 = mean1 - ci_width1 / 2.0f;
    float upper1 = mean1 + ci_width1 / 2.0f;

    float mean2 = stats2.GetMean();
    // float var2 = stats2.GetVariance(); // Not directly used
    float ci_width2 = stats2.GetConfidenceInterval(0.95f);
    float lower2 = mean2 - ci_width2 / 2.0f;
    float upper2 = mean2 + ci_width2 / 2.0f;

    float overlap_lower = std::max(lower1, lower2);
    float overlap_upper = std::min(upper1, upper2);
    float overlap_length = std::max(0.0f, overlap_upper - overlap_lower);

    // Normalize by the minimum of the two interval widths for a more sensitive overlap measure
    float min_ci_width = std::min(ci_width1, ci_width2);
    if (min_ci_width < 1e-6f) { // Avoid division by zero if one interval is tiny
        return (overlap_length > 0) ? 1.0f : 0.0f;
    }

    // Return overlap as a fraction of the smaller credible interval
    return std::min(1.0f, overlap_length / min_ci_width); 
}

} // namespace lczero
