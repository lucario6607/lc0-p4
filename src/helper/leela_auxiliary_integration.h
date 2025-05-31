#ifndef LEELA_AUXILIARY_INTEGRATION_H
#define LEELA_AUXILIARY_INTEGRATION_H

#include "helper/auxiliary_helper_selector.h"
#include "mcts/node.h"
#include "chess/position.h"
#include "chess/board.h"

#include <string>
#include <vector>
#include <iostream>

namespace lczero {
    class NodeTree;
}

class LeelaAuxiliaryIntegration {
public:
    LeelaAuxiliaryIntegration(int max_engines = 4, const std::string& engine_path = "stockfish");

    void OnRootAnalysis(lczero::Node* root, lczero::NodeTree* tree_context);
    void OnPVDivergence(lczero::Node* node, lczero::NodeTree* tree_context, const std::vector<lczero::Move>& leela_pv,
                       const std::vector<lczero::Move>& helper_pv);
    void OnEvaluationDifference(lczero::Node* node, lczero::NodeTree* tree_context, float leela_eval, float helper_eval);
    void ProcessHelperRequests(lczero::NodeTree* tree_context);

    void SetEngineCount(int count);
    void SetConcurrentLimit(int limit);
    void SetEnginePath(const std::string& path);
    void ConfigureStockfish(int threads = 1, int hash_mb = 64, bool use_syzygy = false,
                           const std::string& syzygy_path = "");

    void PrintStatus() const;
    // Temporary for potential debugging or simplified access during development
    const AuxiliaryHelperSelector& GetSelectorForTesting() const { return selector_; }
    AuxiliaryHelperSelector& GetSelectorForTesting() { return selector_; }

private:
    AuxiliaryHelperSelector selector_;

    void SendToHelperEngine(lczero::Node* node, const lczero::Position& position, int engine_id, int request_depth, lczero::NodeTree* tree_context);
};

#endif // LEELA_AUXILIARY_INTEGRATION_H
