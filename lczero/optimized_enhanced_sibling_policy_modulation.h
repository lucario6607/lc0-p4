// optimized_enhanced_sibling_policy_modulation.h
#ifndef LCZERO_OPTIMIZED_ENHANCED_SIBLING_POLICY_MODULATION_H_
#define LCZERO_OPTIMIZED_ENHANCED_SIBLING_POLICY_MODULATION_H_

#include <vector>
#include <cmath>
#include <unordered_map>
#include <memory>
#include <array>
#include <atomic>
#include <thread>
#include <numeric> // Required for std::accumulate, std::iota
#include <algorithm> // Required for std::sort, std::clamp, std::min
#include <mutex>   // Required for std::mutex and std::lock_guard (will be added in next step)

// Forward declarations if Node and Edge are from elsewhere
// namespace lczero { class Node; class Edge; }
// Assuming Node and Edge are defined in these headers for now:
#include "neural/cache.h" // Should contain lczero::NeuralCache (not directly used here but common)
#include "mcts/node.h"    // Should contain lczero::Node and lczero::Edge

namespace lczero {

// Optimized parameter structure with better organization
struct OptimizedSPMParams {
    // Core modulation parameters
    bool enabled = false;
    float q_diff_threshold = 0.1f;
    float policy_boost_factor = 1.25f;
    float policy_damp_factor = 0.8f;
    uint32_t min_visits_for_modulation = 5;
    
    // Enhanced features with better defaults
    struct {
        bool enabled = true;
        float grandparent_weight = 0.2f;  // Reduced from 0.3f for stability
        float cross_subtree_weight = 0.3f;
        uint32_t min_subtree_visits = 10;
    } multi_level;
    
    struct {
        bool enabled = true;
        uint32_t max_clusters = 3;  // Reduced for efficiency
        float silhouette_threshold = 0.4f;
        uint32_t min_cluster_size = 2;
    } clustering;
    
    struct {
        bool enabled = true;
        uint32_t max_depth = 8;  // Reduced memory usage
        float decay_factor = 0.85f;
        float prediction_weight = 0.15f;
        uint32_t min_samples_for_prediction = 4;
    } history;
    
    struct {
        bool enabled = true;
        float min_confidence = 0.5f;
        float effect_size_threshold = 0.25f;
        uint32_t min_samples = 6;
    } statistics;
    
    struct {
        float momentum_alpha = 0.75f;
        float exploration_boost = 1.3f;
        uint32_t cache_size_limit = 1000;
        uint32_t cleanup_frequency = 500;
    } performance;
};

// Compact data structures for better cache performance
struct CompactPolicyCluster {
    std::vector<uint16_t> edge_indices;  // Store indices instead of pointers
    float avg_policy;
    float avg_q;
    float q_variance;
    uint32_t total_visits;
    float confidence;
};

struct CompactMoveHistory {
    struct Entry {
        float q_value;
        uint16_t visits;
        uint8_t confidence_scaled;  // 0-255 for 0.0-1.0
        
        float GetConfidence() const { return confidence_scaled / 255.0f; }
        void SetConfidence(float conf) { confidence_scaled = static_cast<uint8_t>(conf * 255); }
    };
    
    std::array<Entry, 8> entries;  // Fixed-size circular buffer
    uint8_t current_index = 0;
    uint8_t size = 0;
    
    void AddEntry(float q, uint16_t visits, float confidence);
    float CalculateTrend() const;
    float PredictPerformance() const; // Declaration, implementation missing in provided code
};

class OptimizedSiblingPolicyModulator {
public:
    explicit OptimizedSiblingPolicyModulator(const OptimizedSPMParams& params)
        : params_(params), cleanup_counter_(0) {
        // Pre-allocate common containers to reduce allocations
        temp_q_values_.reserve(64);
        temp_policies_.reserve(64);
        temp_clusters_.reserve(params.clustering.max_clusters);
    }

    // Main interface - optimized for hot path
    float CalculateEffectivePolicy(const Node* parent, const Edge& edge) const;
    
    // Batch processing for better cache utilization
    void ProcessNodeBatch(const std::vector<const Node*>& nodes); // Declaration, implementation missing
    
    // Memory management
    void PeriodicCleanup();
    void ClearCaches() { 
        // This will require locking in the next step
        cached_data_.clear(); 
        move_histories_.clear();
        policy_momentum_.clear();
    }
    
    // Debugging and analysis
    struct ModulationBreakdown {
        float base_policy;
        float final_policy;
        float basic_modulation = 1.0f;
        float multi_level_modulation = 1.0f;
        float cluster_modulation = 1.0f;
        float history_modulation = 1.0f;
        float confidence_factor = 1.0f;
        bool statistically_significant = true;
    };
    
    ModulationBreakdown AnalyzeModulation(const Node* parent, const Edge& edge) const; // Declaration, implementation missing

private:
    struct CachedNodeData {
        float avg_q = 0.0f;
        float q_variance = 0.0f;
        float exploration_efficiency = 0.0f;
        std::vector<CompactPolicyCluster> clusters;
        uint64_t last_update_visits = 0;  // For cache invalidation
        
        bool IsValid(uint64_t current_visits) const {
            return current_visits > 0 && 
                   std::abs(static_cast<int64_t>(current_visits - last_update_visits)) < 20;
        }
    };

    // Core calculation methods - optimized implementations
    float CalculateBasicModulation(const Node* parent, const Edge& edge,
                                 const CachedNodeData& cached_data) const;
    
    float CalculateMultiLevelModulation(const Node* parent, const Edge& edge) const; // Declaration, implementation missing
    
    float CalculateClusterModulation(const Node* parent, const Edge& edge,
                                   const std::vector<CompactPolicyCluster>& clusters) const; // Declaration, implementation missing
    
    float CalculateHistoryModulation(const Node* parent, const Edge& edge) const; // Declaration, implementation missing
    
    // Optimized helper methods
    CachedNodeData& GetOrUpdateCachedData(const Node* parent) const;
    
    std::vector<CompactPolicyCluster> CreateOptimizedClusters(const Node* parent) const;
    
    uint32_t FindOptimalClusterCount(const std::vector<float>& policies) const; // Declaration, implementation missing
    
    float CalculateStatisticalSignificance(const Node* parent, const Edge& edge) const; // Declaration, implementation missing
    
    void UpdateMoveHistory(const Edge& edge, float q_value, uint32_t visits, float confidence) const;
    
    // Thread-safe operations
    mutable std::unordered_map<const Node*, CachedNodeData> cached_data_;
    mutable std::mutex cached_data_mutex_;
    mutable std::unordered_map<const Edge*, CompactMoveHistory> move_histories_;
    mutable std::mutex move_histories_mutex_;
    mutable std::unordered_map<const Edge*, float> policy_momentum_;
    mutable std::mutex policy_momentum_mutex_;
    
    // Temporary containers to reduce allocations
    mutable std::vector<float> temp_q_values_;
    mutable std::vector<float> temp_policies_;
    mutable std::vector<CompactPolicyCluster> temp_clusters_;
    
    const OptimizedSPMParams& params_;
    mutable std::atomic<uint32_t> cleanup_counter_;
};

// Optimized implementations
inline float OptimizedSiblingPolicyModulator::CalculateEffectivePolicy(
    const Node* parent, const Edge& edge) const {
    
    if (!params_.enabled || !parent || parent->GetNumEdges() < 2) {
        return edge.GetP();
    }
    
    // Early exit for low-visit edges to avoid noise
    if (edge.GetN() < params_.min_visits_for_modulation) {
        return edge.GetP();
    }
    
    // Get or compute cached data
    // Note: GetOrUpdateCachedData and other map accesses will be mutex-protected in the next step.
    const CachedNodeData& cached_data = GetOrUpdateCachedData(parent); // Internally locked
    
    // Quick statistical significance check
    if (params_.statistics.enabled) {
        // Assuming CalculateStatisticalSignificance is implemented or will be.
        // float significance = CalculateStatisticalSignificance(parent, edge);
        // if (significance < params_.statistics.effect_size_threshold) {
        //     return edge.GetP();
        // }
        // For now, let's assume it's not critical for this step or will be stubbed.
    }
    
    float base_policy = edge.GetP();
    float modulation = 1.0f;
    
    // Apply modulations in order of computational cost (cheapest first)
    modulation *= CalculateBasicModulation(parent, edge, cached_data);
    
    if (params_.clustering.enabled && !cached_data.clusters.empty()) {
        // modulation *= CalculateClusterModulation(parent, edge, cached_data.clusters);
        // Assuming CalculateClusterModulation is implemented or will be.
    }
    
    if (params_.history.enabled) {
        // modulation *= CalculateHistoryModulation(parent, edge);
        // Assuming CalculateHistoryModulation is implemented or will be.
    }
    
    if (params_.multi_level.enabled) {
        // modulation *= CalculateMultiLevelModulation(parent, edge);
        // Assuming CalculateMultiLevelModulation is implemented or will be.
    }
    
    // Apply momentum smoothing
    float final_modulation_value = modulation; // Store current modulation before locking
    { // Scope for policy_momentum_mutex_
        std::lock_guard<std::mutex> lock(policy_momentum_mutex_);
        auto momentum_it = policy_momentum_.find(&edge);
        if (momentum_it != policy_momentum_.end()) {
            final_modulation_value = params_.performance.momentum_alpha * momentum_it->second +
                                     (1.0f - params_.performance.momentum_alpha) * modulation;
        }
        policy_momentum_[&edge] = final_modulation_value; // Update with the potentially modified value
    }
    modulation = final_modulation_value; // Assign back to modulation
    
    // Confidence-based clamping
    float confidence = cached_data.exploration_efficiency; // exploration_efficiency might not be fully calculated if dependencies are missing
    float max_change = 0.5f + confidence * 1.0f;  // More confident = larger changes allowed
    modulation = std::clamp(modulation, 1.0f - max_change, 1.0f + max_change);
    
    // Update history tracking
    float effective_policy = base_policy * modulation;
    UpdateMoveHistory(edge, edge.GetQ(0.0f), edge.GetN(), confidence);
    
    // Periodic cleanup
    if ((++cleanup_counter_) % params_.performance.cleanup_frequency == 0) {
        // This const_cast is a code smell often related to mutable members being modified in const methods.
        // The actual cleanup modifies maps, which will need locking.
        const_cast<OptimizedSiblingPolicyModulator*>(this)->PeriodicCleanup();
    }
    
    return effective_policy;
}

inline OptimizedSiblingPolicyModulator::CachedNodeData& 
OptimizedSiblingPolicyModulator::GetOrUpdateCachedData(const Node* parent) const {
    std::lock_guard<std::mutex> lock(cached_data_mutex_);
    auto& cached_data = cached_data_[parent]; // This insertion needs protection
    uint64_t current_visits = parent->GetN();
    
    if (cached_data.IsValid(current_visits)) {
        return cached_data;
    }
    
    // Recompute cached data
    temp_q_values_.clear();
    temp_policies_.clear();
    
    for (const auto& edge_item : parent->Edges()) { // Assuming Edges() returns a container of Edge or Edge*
        // If Edges() returns Edge*, then use edge_item->GetN(), etc.
        // For now, assuming it returns references or objects directly.
        if (edge_item.GetN() >= params_.min_visits_for_modulation) {
            temp_q_values_.push_back(edge_item.GetQ(0.0f));
            temp_policies_.push_back(edge_item.GetP());
        }
    }
    
    if (temp_q_values_.empty()) {
        cached_data.last_update_visits = current_visits;
        // exploration_efficiency and other fields remain default
        return cached_data;
    }
    
    // Calculate statistics efficiently
    cached_data.avg_q = std::accumulate(temp_q_values_.begin(), temp_q_values_.end(), 0.0f) 
                       / temp_q_values_.size();
    
    float q_var_sum = 0.0f;
    for (float q : temp_q_values_) {
        float diff = q - cached_data.avg_q;
        q_var_sum += diff * diff;
    }
    cached_data.q_variance = (temp_q_values_.size() > 0) ? (q_var_sum / temp_q_values_.size()) : 0.0f;
    
    // Calculate exploration efficiency using Spearman correlation approximation
    if (temp_q_values_.size() >= 3 && temp_q_values_.size() == temp_policies_.size()) { // ensure policies also populated
        std::vector<size_t> q_indices(temp_q_values_.size());
        std::iota(q_indices.begin(), q_indices.end(), 0);
        // Ensure temp_policies_ is captured correctly if it's a member
        // Also, check if temp_q_values_ is what's intended for sorting.
        std::sort(q_indices.begin(), q_indices.end(), 
                 [&](size_t a, size_t b) { return temp_q_values_[a] > temp_q_values_[b]; });
        
        float rank_correlation = 0.0f;
        if (q_indices.size() > 1) { // Avoid division by zero if size is 1
            for (size_t i = 0; i < q_indices.size(); ++i) {
                float policy_rank = static_cast<float>(i) / (q_indices.size() - 1);
                float actual_policy = temp_policies_[q_indices[i]];
                rank_correlation += std::abs(policy_rank - actual_policy);
            }
            cached_data.exploration_efficiency = 1.0f - (rank_correlation / q_indices.size());
        } else {
             cached_data.exploration_efficiency = 0.5f; // Default for single element
        }
    } else {
        cached_data.exploration_efficiency = 0.5f; // Default if not enough data
    }
    
    // Update clusters if enabled
    if (params_.clustering.enabled) {
        cached_data.clusters = CreateOptimizedClusters(parent);
    }
    
    cached_data.last_update_visits = current_visits;
    return cached_data;
}

inline float OptimizedSiblingPolicyModulator::CalculateBasicModulation(
    const Node* parent, const Edge& edge, const CachedNodeData& cached_data) const {
    
    float edge_q = edge.GetQ(0.0f);
    float q_diff = edge_q - cached_data.avg_q;
    
    if (std::abs(q_diff) < params_.q_diff_threshold) {
        return 1.0f;
    }
    
    float modulation = 1.0f;
    bool is_high_policy = edge.GetP() > 0.05f;  // Simplified threshold
    
    if (q_diff > 0 && !is_high_policy) {
        // Low policy, high performance - boost
        modulation = params_.policy_boost_factor;
    } else if (q_diff < 0 && is_high_policy) {
        // High policy, low performance - dampen
        modulation = params_.policy_damp_factor;
    }
    
    // Scale by confidence
    float confidence_scale = std::min(1.0f, static_cast<float>(edge.GetN()) / 20.0f);
    return 1.0f + (modulation - 1.0f) * confidence_scale;
}

inline std::vector<OptimizedSiblingPolicyModulator::CompactPolicyCluster> 
OptimizedSiblingPolicyModulator::CreateOptimizedClusters(const Node* parent) const {
    
    temp_policies_.clear(); // mutable member
    std::vector<uint16_t> valid_indices;
    valid_indices.reserve(parent->GetNumEdges()); // Pre-reserve
    
    uint16_t index = 0;
    for (const auto& edge_item : parent->Edges()) { // Assuming Edges() returns container of Edge or Edge*
        if (edge_item.GetN() >= params_.min_visits_for_modulation) {
            temp_policies_.push_back(edge_item.GetP());
            valid_indices.push_back(index);
        }
        ++index;
    }
    
    if (valid_indices.size() < params_.clustering.min_cluster_size * 2 || temp_policies_.empty()) {
        return {};  // Not enough data for clustering
    }
    
    // uint32_t num_clusters = FindOptimalClusterCount(temp_policies_);
    // FindOptimalClusterCount is not implemented. Using max_clusters as a placeholder.
    uint32_t num_clusters = std::min(params_.clustering.max_clusters, static_cast<uint32_t>(temp_policies_.size() / params_.clustering.min_cluster_size));
    if (num_clusters == 0 && !temp_policies_.empty()) num_clusters = 1; // Ensure at least one cluster if there's data
    if (num_clusters == 0) return {};


    std::vector<CompactPolicyCluster> clusters(num_clusters);
    
    // Simple k-means style clustering - sort indices based on their policy values in temp_policies_
    // Need to ensure temp_policies_ corresponds to valid_indices correctly.
    // The current temp_policies_ is populated based on edges that meet min_visits_for_modulation.
    // The valid_indices stores original edge indices.
    // We need to sort valid_indices based on the policy of the edge they point to.
    
    // Create a temporary vector of pairs or structs to sort based on policy values
    // associated with original edge indices.
    std::vector<std::pair<float, uint16_t>> policy_edge_pairs;
    policy_edge_pairs.reserve(valid_indices.size());
    for(uint16_t edge_idx : valid_indices) {
        policy_edge_pairs.push_back({parent->Edges()[edge_idx].GetP(), edge_idx});
    }

    std::sort(policy_edge_pairs.begin(), policy_edge_pairs.end(),
             [](const auto& a, const auto& b) { return a.first > b.first; }); // Sort by policy descending

    // Distribute sorted edge_indices among clusters
    for (size_t i = 0; i < policy_edge_pairs.size(); ++i) {
        size_t cluster_idx = (i * num_clusters) / policy_edge_pairs.size();
        clusters[cluster_idx].edge_indices.push_back(policy_edge_pairs[i].second);
    }
    
    // Calculate cluster statistics
    for (auto& cluster : clusters) {
        if (cluster.edge_indices.empty()) continue;
        
        float policy_sum = 0.0f, q_sum = 0.0f;
        uint32_t visit_sum = 0;
        
        for (uint16_t idx : cluster.edge_indices) {
            const auto& edge = parent->Edges()[idx]; // Access edge using original index
            policy_sum += edge.GetP();
            q_sum += edge.GetQ(0.0f);
            visit_sum += edge.GetN();
        }
        
        cluster.avg_policy = policy_sum / cluster.edge_indices.size();
        cluster.avg_q = q_sum / cluster.edge_indices.size();
        cluster.total_visits = visit_sum;
        
        // Calculate variance
        float q_var_sum = 0.0f;
        for (uint16_t idx : cluster.edge_indices) {
            float diff = parent->Edges()[idx].GetQ(0.0f) - cluster.avg_q; // Access edge using original index
            q_var_sum += diff * diff;
        }
        cluster.q_variance = q_var_sum / cluster.edge_indices.size();
        // Simplified confidence, ensure total_visits > 0
        cluster.confidence = (cluster.total_visits > 0) ? 
            static_cast<float>(visit_sum) / (1.0f + cluster.q_variance * 10.0f) : 0.0f;
    }
    
    return clusters;
}

inline void OptimizedSiblingPolicyModulator::CompactMoveHistory::AddEntry(
    float q, uint16_t visits, float confidence) {
    
    entries[current_index] = {q, visits, 0}; // q_value, visits, confidence_scaled
    entries[current_index].SetConfidence(confidence);
    
    current_index = (current_index + 1) % entries.size();
    if (size < entries.size()) {
        ++size;
    }
}

inline float OptimizedSiblingPolicyModulator::CompactMoveHistory::CalculateTrend() const {
    if (size < 3) return 0.0f;
    
    float recent_avg = 0.0f, older_avg = 0.0f;
    uint8_t recent_count = 0, older_count = 0;
    
    uint8_t split_point = size * 2 / 3; // e.g. if size=8, split=5. i=5,6,7 are recent
    for (uint8_t i = 0; i < size; ++i) {
        // Correct indexing for circular buffer: (start_index + i) % capacity
        // Here, start_index is (current_index - size + entries.size()) % entries.size()
        uint8_t actual_idx = (current_index + entries.size() - size + i) % entries.size();
        
        if (i >= split_point) {
            recent_avg += entries[actual_idx].q_value;
            ++recent_count;
        } else {
            older_avg += entries[actual_idx].q_value;
            ++older_count;
        }
    }
    
    if (recent_count == 0 || older_count == 0) return 0.0f; // Should not happen if size >= 3
    
    return (recent_avg / recent_count) - (older_avg / older_count);
}

// Definition for UpdateMoveHistory (was only declared)
inline void OptimizedSiblingPolicyModulator::UpdateMoveHistory(
    const Edge& edge, float q_value, uint32_t visits, float confidence) const {
    std::lock_guard<std::mutex> lock(move_histories_mutex_);
    move_histories_[&edge].AddEntry(q_value, static_cast<uint16_t>(visits), confidence);
}


inline void OptimizedSiblingPolicyModulator::PeriodicCleanup() {
    std::lock_guard<std::mutex> lock_cached(cached_data_mutex_);
    std::lock_guard<std::mutex> lock_histories(move_histories_mutex_);
    std::lock_guard<std::mutex> lock_momentum(policy_momentum_mutex_);

    // Remove stale cache entries
    if (cached_data_.size() > params_.performance.cache_size_limit) {
        // This is not the most performant way to remove N random items or oldest items.
        // For simplicity, we'll keep it, but it could be slow.
        // A more targeted LRU or random removal might be better.
        // Erasing a range from unordered_map can be slow.
        // Let's remove a fixed number of elements by iterating if map is too large.
        size_t num_to_remove = cached_data_.size() / 4;
        auto it = cached_data_.begin();
        for(size_t i = 0; i < num_to_remove && it != cached_data_.end(); ++i) {
            it = cached_data_.erase(it); // erase current and advance iterator
        }
    }
    
    // Clean up rarely used move histories
    if (move_histories_.size() > params_.performance.cache_size_limit) {
        // Simple cleanup - could be more sophisticated (e.g. LRU based on node visits)
        // For now, clearing a portion like cached_data_
        size_t num_to_remove = move_histories_.size() / 4;
        auto it = move_histories_.begin();
        for(size_t i = 0; i < num_to_remove && it != move_histories_.end(); ++i) {
            it = move_histories_.erase(it);
        }
    }
    
    // Clean up momentum tracking
    if (policy_momentum_.size() > params_.performance.cache_size_limit) {
        size_t num_to_remove = policy_momentum_.size() / 4;
        auto it = policy_momentum_.begin();
        for(size_t i = 0; i < num_to_remove && it != policy_momentum_.end(); ++i) {
            it = policy_momentum_.erase(it);
        }
    }
}


// ClearCaches method - Modified to acquire locks
inline void OptimizedSiblingPolicyModulator::ClearCaches() {
    std::lock_guard<std::mutex> lock_cached(cached_data_mutex_);
    std::lock_guard<std::mutex> lock_histories(move_histories_mutex_);
    std::lock_guard<std::mutex> lock_momentum(policy_momentum_mutex_);

    cached_data_.clear();
    move_histories_.clear();
    policy_momentum_.clear();
}


// Factory function for easy integration
inline std::unique_ptr<OptimizedSiblingPolicyModulator> 
CreateOptimizedSPM(const OptimizedSPMParams& params = OptimizedSPMParams{}) {
    return std::make_unique<OptimizedSiblingPolicyModulator>(params);
}

// Stub implementations for missing methods to make the header self-contained for now
// These would typically be implemented elsewhere or defined if they are simple enough.
inline float OptimizedSiblingPolicyModulator::CompactMoveHistory::PredictPerformance() const {
    // Placeholder: Actual prediction logic would be more complex.
    if (size == 0) return 0.0f;
    float sum_q = 0.0f;
    for (uint8_t i = 0; i < size; ++i) {
        uint8_t actual_idx = (current_index + entries.size() - size + i) % entries.size();
        sum_q += entries[actual_idx].q_value;
    }
    return sum_q / size; // Simple average as placeholder
}

inline void OptimizedSiblingPolicyModulator::ProcessNodeBatch(const std::vector<const Node*>& nodes) {
    // Placeholder: Batch processing logic would go here.
    // Example: iterate over nodes and call GetOrUpdateCachedData to pre-warm cache.
    for (const Node* node : nodes) {
        if (node) {
            GetOrUpdateCachedData(node);
        }
    }
}

inline OptimizedSiblingPolicyModulator::ModulationBreakdown 
OptimizedSiblingPolicyModulator::AnalyzeModulation(const Node* parent, const Edge& edge) const {
    ModulationBreakdown breakdown;
    breakdown.base_policy = edge.GetP();
    // Placeholder: Actual analysis would compute all modulation factors.
    // This is a simplified version.
    if (!params_.enabled || !parent || parent->GetNumEdges() < 2 || edge.GetN() < params_.min_visits_for_modulation) {
        breakdown.final_policy = edge.GetP();
        return breakdown;
    }
    const CachedNodeData& cached_data = GetOrUpdateCachedData(parent);
    breakdown.basic_modulation = CalculateBasicModulation(parent, edge, cached_data);
    // ... other modulations ...
    breakdown.final_policy = breakdown.base_policy * breakdown.basic_modulation; // Simplified
    return breakdown;
}

inline float OptimizedSiblingPolicyModulator::CalculateMultiLevelModulation(const Node* parent, const Edge& edge) const {
    // Placeholder
    return 1.0f;
}

inline float OptimizedSiblingPolicyModulator::CalculateClusterModulation(
    const Node* parent, const Edge& edge, const std::vector<CompactPolicyCluster>& clusters) const {
    // Placeholder
    // Find which cluster this edge belongs to, then apply logic.
    // This requires edge_indices in CompactPolicyCluster to be original indices.
    // For now, returning neutral modulation.
    return 1.0f;
}

inline float OptimizedSiblingPolicyModulator::CalculateHistoryModulation(const Node* parent, const Edge& edge) const {
    // Placeholder
    // auto it = move_histories_.find(&edge);
    // if (it != move_histories_.end() && it->second.size > 0) {
    //    return 1.0f + it->second.CalculateTrend() * params_.history.prediction_weight; // Example
    // }
    return 1.0f;
}

inline uint32_t OptimizedSiblingPolicyModulator::FindOptimalClusterCount(const std::vector<float>& policies) const {
    // Placeholder: Actual logic would use silhouette scores or other methods.
    if (policies.empty()) return 0;
    return std::min(params_.clustering.max_clusters, static_cast<uint32_t>(policies.size() / params_.clustering.min_cluster_size));
}

inline float OptimizedSiblingPolicyModulator::CalculateStatisticalSignificance(const Node* parent, const Edge& edge) const {
    // Placeholder: Actual statistical calculation (e.g., t-test, chi-squared) needed.
    // For now, return a value that passes the threshold to allow flow.
    return params_.statistics.effect_size_threshold + 0.1f;
}


} // namespace lczero

#endif // LCZERO_OPTIMIZED_ENHANCED_SIBLING_POLICY_MODULATION_H_
