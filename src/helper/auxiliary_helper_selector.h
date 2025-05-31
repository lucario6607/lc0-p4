#ifndef AUXILIARY_HELPER_SELECTOR_H
#define AUXILIARY_HELPER_SELECTOR_H

#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <algorithm>
#include <optional>
#include <string>

// Forward declarations for Leela's existing structures
namespace lczero {
    class Node;
    // class Position; // Full definition will be included
    class Move;
    // Assuming NodeTree is needed for context, or Search for relevance checks
    class NodeTree;
}

#include "chess/position.h" // For lczero::Position full definition
#include "helper/helper_engine_pool.h" // For engine_pool_

// Struct for requests in the priority queue
struct HelperRequest {
    lczero::Node* node;          // Pointer to the Leela Node
    lczero::Position position;   // The position corresponding to the node
    int depth;                   // Depth of the node in the search or game
    std::chrono::steady_clock::time_point enqueue_time;
    float priority_score;

    HelperRequest() : node(nullptr),
                      position(lczero::ChessBoard(lczero::ChessBoard::kStartposFen), 0, 0),
                      depth(0),
                      enqueue_time(std::chrono::steady_clock::now()),
                      priority_score(0.0f) {}

    // Priority queue comparison (higher score = higher priority)
    bool operator<(const HelperRequest& other) const {
        return priority_score < other.priority_score;
    }
};

class AuxiliaryHelperSelector {
public:
    AuxiliaryHelperSelector();
    ~AuxiliaryHelperSelector();

    // Main interface for adding nodes
    void AddRootNode(lczero::Node* root, lczero::NodeTree* tree_context); // Added tree_context
    void AddDivergenceNode(lczero::Node* node, float eval_difference, lczero::NodeTree* tree_context);
    void AddPVComparisonNode(lczero::Node* node, const std::vector<lczero::Move>& leela_pv,
                             const std::vector<lczero::Move>& helper_pv, lczero::NodeTree* tree_context);

    std::optional<std::pair<HelperRequest, int>> GetNextRequest();
    void OnHelperComplete(lczero::Node* node, bool success);

    // Configuration setters
    void SetMaxDepth(int depth);
    void SetEvalThreshold(float threshold);
    void SetMaxQueueSize(int size);
    void SetMaxConcurrentHelpers(int count);
    void SetMaxHelperInstances(int count);
    void SetHelperEnginePath(const std::string& path);
    void SetEngineOptions(const std::vector<std::string>& options);
    void SetEngineRestartThreshold(int queries);

    // Status and monitoring
    int GetActiveEngineCount() const;
    int GetAvailableEngineCount() const;
    int GetQueueSize() const;
    float GetSuccessRate() const;

    // Public getters for specific config values needed by LeelaAuxiliaryIntegration
    int GetMaxConcurrentHelpersConfig() const { return config_.max_concurrent_helpers; }
    int GetMaxHelperInstancesConfig() const { return config_.max_helper_instances; }
    const std::string& GetHelperEnginePathConfig() const { return config_.helper_engine_path; }

    // Allow LeelaAuxiliaryIntegration to initialize the pool
    void InitializeEnginePool();

private:
    enum class NodeSource {
        ROOT,
        DIVERGENCE,
        PV_COMPARISON,
        // TACTICAL_PATTERN // Not explicitly used in Add methods from issue, but present in CalculatePriorityScore
    };

    struct Config {
        int max_depth = 20;
        float base_probability = 0.4f;
        float eval_threshold = 0.15f;
        int max_queue_size = 1000;
        int max_concurrent_helpers = 3;
        int max_helper_instances = 4;
        std::chrono::milliseconds max_age_ms{30000};
        float tactical_position_bonus = 0.3f;
        float recent_divergence_bonus = 0.2f;

        std::string helper_engine_path = "stockfish"; // Default, should be configurable
        std::vector<std::string> engine_options;
        int engine_restart_threshold = 1000;
    } config_;

    std::priority_queue<HelperRequest> request_queue_;
    std::unordered_set<lczero::Node*> queued_nodes_;
    std::unordered_map<lczero::Node*, int> processing_nodes_; // Node -> engine_idx
    std::unordered_map<lczero::Node*, std::chrono::steady_clock::time_point> helper_start_times_;

    std::unique_ptr<HelperEnginePool> engine_pool_;
    std::vector<bool> engine_busy_;

    struct Statistics {
        int total_requests = 0;
        int successful_queries = 0;
        int timeouts = 0;
        // float avg_response_time_ms = 100.0f; // Not used in provided code, consider adding later
        std::unordered_map<int, int> depth_success_rate; // Needs population
    } stats_;

    std::mt19937 rng_;
    std::uniform_real_distribution<float> uniform_dist_{0.0f, 1.0f};

    // Internal methods
    // void InitializeEnginePool(); // Moved to public
    void ShutdownEnginePool();
    void ReconfigureEnginePool();

    bool ShouldEnqueue(lczero::Node* node, NodeSource source, lczero::NodeTree* tree_context);
    void EnqueueNode(lczero::Node* node, NodeSource source, lczero::NodeTree* tree_context, float importance = 0.0f);
    float CalculatePriorityScore(lczero::Node* node, NodeSource source, lczero::NodeTree* tree_context, float importance);
    float CalculateDepthSamplingProbability(int depth);
    float CalculatePVDivergenceImportance(const std::vector<lczero::Move>& leela_pv,
                                        const std::vector<lczero::Move>& helper_pv);
    void CleanupExpiredRequests(lczero::NodeTree* tree_context);
    void UpdateAdaptiveParameters();
    int GetAvailableEngine();

    // Placeholder implementations - these are critical and need proper Leela integration
    bool IsNodeStillRelevant(lczero::Node* node, lczero::NodeTree* tree_context);
    int GetNodeDepth(lczero::Node* node, lczero::NodeTree* tree_context);
    lczero::Position GetNodePosition(lczero::Node* node, lczero::NodeTree* tree_context);
    int GetNodeVisits(lczero::Node* node);
    std::chrono::steady_clock::time_point GetNodeCreationTime(lczero::Node* node);
    bool IsTacticalPosition(lczero::Node* node, lczero::NodeTree* tree_context);
};

#endif // AUXILIARY_HELPER_SELECTOR_H
