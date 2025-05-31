#include "helper/leela_auxiliary_integration.h"
#include "chess/position.h" // For lczero::GetFen
#include "mcts/node.h"      // For lczero::NodeTree (used as parameter)

LeelaAuxiliaryIntegration::LeelaAuxiliaryIntegration(int max_engines, const std::string& engine_path) {
    selector_.SetMaxHelperInstances(max_engines);
    selector_.SetHelperEnginePath(engine_path);

    std::vector<std::string> options = {
        "name Threads value 1",
        "name Hash value 64",
        "name MultiPV value 1"
    };
    selector_.SetEngineOptions(options);
    // Ensure engine pool is initialized after settings are applied
    selector_.InitializeEnginePool();
    std::cout << "LeelaAuxiliaryIntegration initialized." << std::endl;
}

void LeelaAuxiliaryIntegration::OnRootAnalysis(lczero::Node* root, lczero::NodeTree* tree_context) {
    selector_.AddRootNode(root, tree_context);
    // std::cout << "OnRootAnalysis called." << std::endl;
}

void LeelaAuxiliaryIntegration::OnPVDivergence(lczero::Node* node, lczero::NodeTree* tree_context,
                                           const std::vector<lczero::Move>& leela_pv,
                                           const std::vector<lczero::Move>& helper_pv) {
    selector_.AddPVComparisonNode(node, leela_pv, helper_pv, tree_context);
    // std::cout << "OnPVDivergence called." << std::endl;
}

void LeelaAuxiliaryIntegration::OnEvaluationDifference(lczero::Node* node, lczero::NodeTree* tree_context,
                                                   float leela_eval, float helper_eval) {
    float difference = std::abs(leela_eval - helper_eval);
    selector_.AddDivergenceNode(node, difference, tree_context);
    // std::cout << "OnEvaluationDifference called with diff: " << difference << std::endl;
}

void LeelaAuxiliaryIntegration::ProcessHelperRequests(lczero::NodeTree* tree_context) {
    while (auto request_pair = selector_.GetNextRequest()) {
        auto& [request_obj, engine_idx] = *request_pair;
        SendToHelperEngine(request_obj.node, request_obj.position, engine_idx, request_obj.depth, tree_context);
    }
}

void LeelaAuxiliaryIntegration::SetEngineCount(int count) {
    selector_.SetMaxHelperInstances(count);
}

void LeelaAuxiliaryIntegration::SetConcurrentLimit(int limit) {
    selector_.SetMaxConcurrentHelpers(limit);
}

void LeelaAuxiliaryIntegration::SetEnginePath(const std::string& path) {
    selector_.SetHelperEnginePath(path);
}

void LeelaAuxiliaryIntegration::ConfigureStockfish(int threads, int hash_mb, bool use_syzygy,
                                               const std::string& syzygy_path) {
    std::vector<std::string> options = {
        "name Threads value " + std::to_string(threads),
        "name Hash value " + std::to_string(hash_mb),
        "name MultiPV value 1"
    };
    if (use_syzygy && !syzygy_path.empty()) {
        options.push_back("name SyzygyPath value " + syzygy_path);
    }
    selector_.SetEngineOptions(options);
}

void LeelaAuxiliaryIntegration::PrintStatus() const {
    std::cout << "--- Auxiliary Helper Status ---" << std::endl;
    std::cout << "  Active Engines: " << selector_.GetActiveEngineCount() << " / Max Concurrent: " << selector_.GetMaxConcurrentHelpersConfig() << std::endl;
    std::cout << "  Total Instances Configured: " << selector_.GetMaxHelperInstancesConfig() << std::endl;
    std::cout << "  Queue Size: " << selector_.GetQueueSize() << std::endl;
    std::cout << "  Success Rate: " << (selector_.GetSuccessRate() * 100.0f) << "%" << std::endl;
    std::cout << "  Engine Path: " << selector_.GetHelperEnginePathConfig() << std::endl;
}

void LeelaAuxiliaryIntegration::SendToHelperEngine(lczero::Node* node, const lczero::Position& position,
                                               int engine_id, int request_depth, lczero::NodeTree* tree_context) {
    std::string fen = lczero::GetFen(position);
    // std::cout << "[LeelaAuxIntegration] Attempting to send to engine " << engine_id << " FEN: " << fen << " Depth: " << request_depth << std::endl;

    // This is where the actual call to the selector's engine pool to send UCI commands would go.
    // For this to work, AuxiliaryHelperSelector needs to expose methods like:
    // - bool SendPositionToEngine(int engine_idx, const std::string& fen);
    // - bool RequestAnalysisFromEngine(int engine_idx, int depth, int time_ms);
    // These methods would then call the corresponding methods on its private HelperEnginePool instance.

    // For now, this function will just log that it's been called.
    // The actual sending of commands is deferred until AuxiliaryHelperSelector's interface is updated.
    std::cout << "[INFO] SendToHelperEngine called for engine " << engine_id << ". FEN: " << fen << ", Depth: " << request_depth << std::endl;
    std::cout << "       (Actual UCI command sending is deferred pending AuxiliaryHelperSelector interface changes)" << std::endl;

    // Placeholder: If this function were to directly interact with the pool (which it can't due to privacy),
    // it might look like this, assuming selector_ had public methods `PoolSendPosition` and `PoolRequestAnalysis`:
    /*
    if (selector_.PoolSendPosition(engine_id, fen)) {
        if (selector_.PoolRequestAnalysis(engine_id, request_depth, 1000)) { // 1000ms default time
            // std::cout << "Sent position and requested analysis for engine " << engine_id << std::endl;
        } else {
            // std::cerr << "Failed to request analysis from engine " << engine_id << std::endl;
            selector_.OnHelperComplete(node, false);
        }
    } else {
        // std::cerr << "Failed to send position to engine " << engine_id << std::endl;
        selector_.OnHelperComplete(node, false);
    }
    */
}
