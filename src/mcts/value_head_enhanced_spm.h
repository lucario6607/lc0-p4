#ifndef LCZERO_MCTS_VALUE_HEAD_ENHANCED_SPM_H_
#define LCZERO_MCTS_VALUE_HEAD_ENHANCED_SPM_H_

#include <vector>
#include <cmath>
#include <unordered_map>
#include <memory>
#include <numeric>
#include <algorithm>

#include "../utils/optionsdict.h"    // For OptionsDict and OptionId
#include "../utils/optionsparser.h"   // For OptionsParser and TypeOption classes in AddValueHeadSPMOptions
#include "../neural/cache.h"        // Path relative to src/mcts/ (already correct)
#include "../neural/network.h"      // Path relative to src/mcts/ (already correct)
#include "node.h"                   // Path relative to src/mcts/ (already correct)

namespace lczero {

// OptionIds for ValueHeadEnhancedSPMParams based on "original issue" specification
inline static const OptionId kSPMEnabledId{
    "spm-enabled", "Enable Enhanced SPM for Value Head evaluation.", false};
inline static const OptionId kSPMValueHeadWeightId{
    "spm-value-head-weight", "Weight applied to the SPM-adjusted value component.", 0.5f, 0.0f, 1.0f};
inline static const OptionId kSPMOpeningThresholdPlyId{
    "spm-opening-threshold-ply", "Number of plies to define the opening phase for SPM.", 40, 0, 200};
inline static const OptionId kSPMEndgameThresholdPlyId{
    "spm-endgame-threshold-ply", "Number of plies to define the endgame phase for SPM.", 160, 0, 400};
inline static const OptionId kSPMQDiffThresholdId{
    "spm-q-diff-threshold", "Q-value difference threshold for SPM to consider a position uncertain.", 0.1f, 0.0f, 1.0f};
inline static const OptionId kSPMMinGamesId{
    "spm-min-games", "Minimum number of games with similar characteristics for SPM to be active.", 100, 0, 10000};
inline static const OptionId kSPMCriticalityThresholdId{
    "spm-criticality-threshold", "Threshold for determining if a position is critical.", 0.05f, 0.0f, 1.0f};
inline static const OptionId kSPMEndgamePositionBoostId{
    "spm-endgame-position-boost", "Flat Q value boost for positions classified as endgame by SPM.", 0.1f, 0.0f, 1.0f};

// Definition for ValueHeadEnhancedSPMParams struct
// MUST EXACTLY match the one in the original issue.
struct ValueHeadEnhancedSPMParams {
  bool enabled;                     // Default: false
  float value_head_weight;        // Default: 0.5f
  int opening_threshold_ply;      // Default: 40
  int endgame_threshold_ply;      // Default: 160
  float q_diff_threshold;           // Default: 0.1f
  int min_games;                    // Default: 100
  float criticality_threshold;      // Default: 0.05f
  float endgame_position_boost;   // Default: 0.1f

  // Constructor to initialize ALL fields from OptionsDict
  explicit ValueHeadEnhancedSPMParams(const OptionsDict& options) :
    enabled(options.GetOrDefault<bool>(kSPMEnabledId, false)),
    value_head_weight(options.GetOrDefault<float>(kSPMValueHeadWeightId, 0.5f)),
    opening_threshold_ply(options.GetOrDefault<int>(kSPMOpeningThresholdPlyId, 40)),
    endgame_threshold_ply(options.GetOrDefault<int>(kSPMEndgameThresholdPlyId, 160)),
    q_diff_threshold(options.GetOrDefault<float>(kSPMQDiffThresholdId, 0.1f)),
    min_games(options.GetOrDefault<int>(kSPMMinGamesId, 100)),
    criticality_threshold(options.GetOrDefault<float>(kSPMCriticalityThresholdId, 0.05f)),
    endgame_position_boost(options.GetOrDefault<float>(kSPMEndgamePositionBoostId, 0.1f)) {
  }
};

class ValueHeadEnhancedSPM {
 public:
  ValueHeadEnhancedSPM(const ValueHeadEnhancedSPMParams& params) : params_(params) {}

  enum class PositionType {
    UNKNOWN = 0,
    OPENING = 1,
    MIDGAME = 2,
    ENDGAME = 3,
    CRITICAL = 4 // Typically for positions requiring deep tactical calculation
  };

  // Removed the constructor taking SearchConfig, now uses ValueHeadEnhancedSPMParams
  // ValueHeadEnhancedSPM(const SearchConfig& config) : config_(config) {}

  float CalculateValueHeadConfidence(Node* node, float nn_value) {
    // Dummy implementation - to be filled with actual logic
    if (!params_.enabled) return nn_value; // Or some default confidence
    // Actual confidence calculation logic here based on params_ and node state
    return nn_value; // Placeholder
  }

  PositionType GetPositionType(Node* node) {
    // Dummy implementation - to be filled with actual logic
    // This requires Node to have a way to get current ply count or game phase indicators
    // int ply = node->GetPly(); // Assuming Node has GetPly() or similar method
    // if (ply < params_.opening_threshold_ply) return PositionType::OPENING;
    // if (ply > params_.endgame_threshold_ply) return PositionType::ENDGAME;
    // Add logic for CRITICAL based on params_.criticality_threshold
    // Add logic for ENDGAME based on params_.endgame_position_boost if applicable here
    return PositionType::MIDGAME; // Default or more complex logic
  }

  void UpdateSPMStats(Node* node, float game_result) {
    // Dummy implementation - to be filled with actual logic for updating SPM statistics
    // This would involve using GetPositionType(node) and updating spm_scores_
    if (!params_.enabled) return;
    PositionType type = GetPositionType(node);
    // spm_scores_[type].push_back(game_result); // Example
  }

  float GetSPMAdjustedValue(Node* node, float nn_value) {
    // Dummy implementation - to be filled with actual logic
    if (!params_.enabled) return nn_value;
    PositionType type = GetPositionType(node);
    // float spm_adjustment = 0.0f; // Calculate based on spm_scores_[type]
    // float adjusted_value = nn_value * (1.0f - params_.value_head_weight) + spm_adjustment * params_.value_head_weight;
    // if (type == PositionType::ENDGAME) {
    // adjusted_value += params_.endgame_position_boost; // Apply boost
    // }
    // return adjusted_value;
    return nn_value; // Placeholder
  }

 private:
  const ValueHeadEnhancedSPMParams& params_;
  std::unordered_map<PositionType, std::vector<float>> spm_scores_; // Stores historical scores per position type
};

// Function to add options to OptionsParser
inline void AddValueHeadSPMOptions(OptionsParser* options) {
  options->Add<BoolOption>(kSPMEnabledId, false);
  options->Add<FloatOption>(kSPMValueHeadWeightId, 0.5f, 0.0f, 1.0f);
  options->Add<IntOption>(kSPMOpeningThresholdPlyId, 40, 0, 200);
  options->Add<IntOption>(kSPMEndgameThresholdPlyId, 160, 0, 400);
  options->Add<FloatOption>(kSPMQDiffThresholdId, 0.1f, 0.0f, 1.0f);
  options->Add<IntOption>(kSPMMinGamesId, 100, 0, 10000);
  options->Add<FloatOption>(kSPMCriticalityThresholdId, 0.05f, 0.0f, 1.0f);
  options->Add<FloatOption>(kSPMEndgamePositionBoostId, 0.1f, 0.0f, 1.0f);
}

}  // namespace lczero

#endif  // LCZERO_MCTS_VALUE_HEAD_ENHANCED_SPM_H_
