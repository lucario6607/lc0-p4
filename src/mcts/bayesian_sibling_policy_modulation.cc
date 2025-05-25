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
    if (!stats || stats->update_count < 2) return 1.0f; // No stats, no modulation.
    
    float confidence_interval_width = stats->GetConfidenceInterval();
    float uncertainty = stats->GetUncertaintyMeasure();
    
    // If we have high uncertainty about this move, explore it more
    if (confidence_interval_width > params_.credible_interval_threshold) {
        return 1.0f + uncertainty * params_.uncertainty_exploration_weight;
    }
    
    // Find the Q-value of the node corresponding to the given edge.
    float current_q_from_node = 0.0f; // Default Q-value.
    bool node_found_and_visited = false;
    // Need to iterate using Node::ConstIterator as this member function is const.
    for (Node::ConstIterator it = parent->Edges().begin(); it != parent->Edges().end(); ++it) {
        if (it.edge() == &edge) {
            // it.node() gives Node*. it.GetN() checks visits of this Node.
            // it.GetQ(default_q, draw_score) gets Q from this Node.
            if (it.node() != nullptr && it.GetN() > 0) {
                 // Using 0.0f for default_q and 0.0f for draw_score.
                 // The draw_score might need to be configurable via SearchParams later.
                current_q_from_node = it.GetQ(0.0f, 0.0f); 
                node_found_and_visited = true;
            }
            break;
        }
    }

    if (!node_found_and_visited) {
        // If we couldn't find the node or it has no visits,
        // we can't reliably use its Q for dampening.
        return 1.0f; // No modulation.
    }
    
    // If we're confident this move is poor, dampen it
    // float current_q = edge.GetQ(0.0f); // OLD LINE
    float current_q = current_q_from_node; // NEW LINE: Use Q from the actual node
    
    float mean_estimated_q = stats->GetMean() * 2.0f - 1.0f; // Convert [0,1] from stats to [-1,1]
    
    // Ensure current_q (from node, typically [-1,1]) and mean_estimated_q (converted to [-1,1]) are comparable.
    
    if (confidence_interval_width < params_.credible_interval_threshold * 0.5f && 
        mean_estimated_q < current_q - params_.q_diff_threshold &&
        stats->update_count >= static_cast<int>(params_.min_confidence_for_dampening * 2)) { // Ensure comparison is int vs float
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
    
    // Iterate over sibling edges using Node::ConstIterator
    for (Node::ConstIterator sibling_it = parent->Edges().begin(); 
         sibling_it != parent->Edges().end(); 
         ++sibling_it) {
        
        // sibling_it is an EdgeAndNode. sibling_it.edge() gives Edge*.
        // &edge is the address of the input Edge object.
        if (sibling_it.edge() != &edge && 
            sibling_it.GetN() >= static_cast<uint32_t>(params_.min_visits_for_modulation)) {
            
            // Get stats for the sibling edge. sibling_it.edge() is Edge*, so dereference for const Edge&.
            BayesianStats* sibling_stats = GetOrCreateBayesianStats(*sibling_it.edge());
            if (sibling_stats) {
                avg_sibling_uncertainty += sibling_stats->GetUncertaintyMeasure();
                sibling_count++;
            }
        }
    }
    
    if (sibling_count == 0) return 1.0f; // No valid siblings to compare against.
    avg_sibling_uncertainty /= sibling_count;
    
    // If this move is more uncertain than siblings, give it an exploration bonus
    if (uncertainty > avg_sibling_uncertainty * 1.2f) { // 1.2f is an arbitrary factor, consider parameterizing
        float bonus = 1.0f + (uncertainty - avg_sibling_uncertainty) * params_.uncertainty_exploration_weight;
        // Ensure policy_boost_factor is positive and > 1 for actual boosting
        return std::min(bonus, params_.policy_boost_factor > 1.0f ? params_.policy_boost_factor : 1.5f ); 
    }
    
    return 1.0f;
}

float BayesianSiblingPolicyModulator::CalculateThompsonSamplingModulation(
    const Node* parent, const Edge& edge) const {
    
    BayesianStats* stats = GetOrCreateBayesianStats(edge);
    // If no stats or not enough data for Thompson sampling, return neutral factor.
    if (!stats || stats->update_count < 2) return 1.0f; 
    
    float thompson_sample = stats->SampleThompson(); // This is in [0,1] range

    // Find the Q-value of the node corresponding to the given edge.
    float current_q_from_node_minus1_to_1 = 0.0f; // Default Q-value.
    bool node_found_and_visited = false;
    // Need to iterate using Node::ConstIterator as this member function is const.
    for (Node::ConstIterator it = parent->Edges().begin(); it != parent->Edges().end(); ++it) {
        if (it.edge() == &edge) {
            if (it.node() != nullptr && it.GetN() > 0) {
                // Using 0.0f for default_q and 0.0f for draw_score.
                current_q_from_node_minus1_to_1 = it.GetQ(0.0f, 0.0f); 
                node_found_and_visited = true;
            }
            break;
        }
    }

    if (!node_found_and_visited) {
        // If node not found or not visited, cannot get its Q value.
        return 1.0f; // Neutral modulation.
    }
    
    // Normalize current Q from [-1,1] to [0,1] for comparison with Thompson sample.
    float current_q_normalized = (current_q_from_node_minus1_to_1 + 1.0f) / 2.0f;
    
    // If Thompson sample suggests this move is better than current estimate, boost it
    // The 0.1f threshold is arbitrary, consider parameterizing.
    if (thompson_sample > current_q_normalized + 0.1f) { 
        float boost = 1.0f + (thompson_sample - current_q_normalized) * 0.5f; // 0.5f factor also arbitrary
        // Ensure policy_boost_factor is positive and > 1.0 for actual boosting, otherwise use a default.
        return std::min(boost, params_.policy_boost_factor > 1.0f ? params_.policy_boost_factor : 1.5f);
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
    BayesianSiblingGroup high_uncertainty_group; // Gets default initialized members (avg_q=0, etc.)
    BayesianSiblingGroup low_uncertainty_group;
    
    float high_unc_q_sum = 0.0f;
    int high_unc_member_count = 0; // Renamed to avoid conflict with BayesianStats::update_count
    float low_unc_q_sum = 0.0f;
    int low_unc_member_count = 0;

    for (Node::ConstIterator it = parent->Edges().begin(); it != parent->Edges().end(); ++it) {
        if (it.GetN() < static_cast<uint32_t>(params_.min_visits_for_modulation)) continue;
        
        BayesianStats* stats = GetOrCreateBayesianStats(*it.edge()); // Pass const Edge&
        float uncertainty = stats ? stats->GetUncertaintyMeasure() : 1.0f; // Default to max uncertainty if no stats
        
        float current_q = 0.0f; // Default Q
        if (it.node() != nullptr && it.GetN() > 0) { // Check node exists and has visits
            current_q = it.GetQ(0.0f, 0.0f); // Get Q from Node, assume default_q=0, draw_score=0
        }

        if (uncertainty > params_.credible_interval_threshold) {
            high_uncertainty_group.edges.push_back(it.edge()); // it.edge() is const Edge*
            high_uncertainty_group.total_policy += it.GetP();
            high_uncertainty_group.total_visits += it.GetN();
            high_uncertainty_group.uncertainty_score += uncertainty;
            high_unc_q_sum += current_q;
            high_unc_member_count++;
        } else {
            low_uncertainty_group.edges.push_back(it.edge()); // it.edge() is const Edge*
            low_uncertainty_group.total_policy += it.GetP();
            low_uncertainty_group.total_visits += it.GetN();
            low_uncertainty_group.uncertainty_score += uncertainty;
            low_unc_q_sum += current_q;
            low_unc_member_count++;
        }
    }
    
    if (high_unc_member_count > 0) {
        high_uncertainty_group.avg_q = high_unc_q_sum / high_unc_member_count;
        high_uncertainty_group.uncertainty_score /= high_unc_member_count; // Normalize score
    }
    // else avg_q remains 0.0f as initialized, uncertainty_score also 0.0f

    if (low_unc_member_count > 0) {
        low_uncertainty_group.avg_q = low_unc_q_sum / low_unc_member_count;
        low_uncertainty_group.uncertainty_score /= low_unc_member_count; // Normalize score
    }
    // else avg_q remains 0.0f, uncertainty_score also 0.0f

    // The old finalize_group lambda is no longer needed as its logic is integrated.
    
    if (!high_uncertainty_group.edges.empty()) groups.push_back(high_uncertainty_group);
    if (!low_uncertainty_group.edges.empty()) groups.push_back(low_uncertainty_group);
    
    return groups;
}

bool BayesianSiblingPolicyModulator::ShouldApplyBayesianModulation(
    const Node* parent, const Edge& edge_param) const { // Renamed edge to edge_param
    
    // Find the node corresponding to edge_param and check its visits.
    bool current_edge_node_has_enough_visits = false;
    // Node::ConstIterator current_edge_it = parent->Edges().end(); // Not strictly needed to store

    for (Node::ConstIterator it = parent->Edges().begin(); it != parent->Edges().end(); ++it) {
        if (it.edge() == &edge_param) {
            // current_edge_it = it; // Found the iterator for the current edge
            if (it.GetN() >= static_cast<uint32_t>(params_.min_visits_for_modulation)) {
                current_edge_node_has_enough_visits = true;
            }
            break; 
        }
    }

    if (!current_edge_node_has_enough_visits) {
        return false; // Current edge's node doesn't have enough visits or not found.
    }
    
    // Need at least one OTHER sibling with Bayesian stats and enough visits
    int siblings_with_stats = 0;
    for (Node::ConstIterator sibling_it = parent->Edges().begin(); 
         sibling_it != parent->Edges().end(); 
         ++sibling_it) {
        
        // Skip the original edge itself
        if (sibling_it.edge() == &edge_param) {
            continue;
        }
        
        if (sibling_it.GetN() >= static_cast<uint32_t>(params_.min_visits_for_modulation)) {
            BayesianStats* stats = GetOrCreateBayesianStats(*sibling_it.edge()); // Pass const Edge&
            if (stats && stats->update_count > 0) {
                siblings_with_stats++;
            }
        }
    }
    
    return siblings_with_stats >= 1;
}

void BayesianSiblingPolicyModulator::UpdateBayesianStats(
    const Node* /*parent*/, const Edge& edge, float outcome) { // Marked parent as unused
    
    BayesianStats* stats = GetOrCreateBayesianStats(edge);
    if (stats) {
        // Apply temporal decay to older observations
        stats->alpha *= params_.bayesian_update_decay;
        stats->beta *= params_.bayesian_update_decay;
        
        // Update with new outcome
        stats->Update(outcome);
    }
}

void BayesianSiblingPolicyModulator::UpdateModulationStats(const Node* /*parent*/) { // Marked parent as unused
    // Clear caches periodically
    if (cached_bayesian_groups_.size() > 1000) {
        cached_bayesian_groups_.clear();
        cached_bayesian_variance_.clear();
    }
    
    // Decay old Bayesian statistics to prevent staleness
    for (auto& pair : edge_bayesian_stats_) {
        BayesianStats& stats = pair.second;
        if (stats.update_count > 100) { // Only decay if there are enough updates
            stats.alpha *= 0.99f; // Use a fixed decay or make it a parameter
            stats.beta *= 0.99f;
        }
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

    float total_visits_weight = 0.0f; // Using 'weight' in name to clarify it's for weighted average
    float mean_q_sum_weighted = 0.0f;
    float mean_q_sq_sum_weighted = 0.0f;
    int count = 0;

    for (Node::ConstIterator it = parent->Edges().begin(); it != parent->Edges().end(); ++it) {
        // Get Bayesian stats for the current edge
        BayesianStats* stats = GetOrCreateBayesianStats(*it.edge()); // Pass const Edge&

        if (stats && stats->update_count > 0) {
            float mean_q_for_edge = stats->GetMean(); // Q is already in [0,1] range from BayesianStats
            
            // Use actual visits from the node for weighting, if available and makes sense.
            // Or, could use stats->update_count if that's more representative of the stat's reliability.
            // Let's stick to node visits as per the original placeholder's intent.
            float visits_for_weighting = static_cast<float>(it.GetN());

            if (visits_for_weighting > 0) { // Only consider edges/nodes that have been visited
                mean_q_sum_weighted += mean_q_for_edge * visits_for_weighting;
                mean_q_sq_sum_weighted += mean_q_for_edge * mean_q_for_edge * visits_for_weighting;
                total_visits_weight += visits_for_weighting;
                count++;
            }
        }
    }

    if (total_visits_weight < 1e-6f || count < 2) return 0.0f; // Not enough data or not enough distinct visited children

    float overall_mean_q = mean_q_sum_weighted / total_visits_weight;
    float variance = (mean_q_sq_sum_weighted / total_visits_weight) - (overall_mean_q * overall_mean_q);
    
    return std::max(0.0f, variance); // Variance cannot be negative
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
