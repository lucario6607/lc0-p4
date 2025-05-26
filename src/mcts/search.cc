/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2019 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#include "mcts/search.h" // Should now include the modified search (6).h content

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip> // For std::fixed, std::setprecision
#include <iostream>
#include <iterator>
#include <limits> // Required for std::numeric_limits
#include <memory>
#include <sstream> // For std::ostringstream
#include <thread>

// Assuming these are accessible, or mcts/node.h defines them.
// #include "mcts/node.h" // Already in search.h
#include "neural/encoder.h" // For policy vector processing if needed here
#include "utils/fastmath.h"
#include "utils/random.h"
#include "utils/spinhelper.h"


namespace lczero {

// Constants for Thompson Sampling default parameters (if not in SearchParams)
// These will be taken from SearchParams which gets them from OptionsDict
// const float kDefaultFpuValueTS = 0.0f; // Example, use params_.GetFpuValue()
// const float kDefaultPolicyTemperatureTS = 1.0f; // Example, use params_.GetPolicyTemperature()
// const bool kDefaultUsePolicyPriorsTS = true; // Example, use params_.GetUsePolicyPriors()


// **************************************************************************
// Implementations for Beta-Bernoulli Thompson Sampling related methods
// (Typically would be in mcts/node.cc or similar)
// **************************************************************************
#ifndef EDGE_DEFINED_EXTERNALLY // If Edge was defined in search.h
Node* Edge::GetOrSpawnNode(Node* parent, uint16_t index_in_parent) {
  if (!node_) {
    // TODO: Ensure Node constructor is compatible. Node needs parent and its index in parent.
    node_ = std::make_unique<Node>(parent, index_in_parent);
  }
  return node_.get();
}

std::string Edge::DebugString() const {
  std::ostringstream out;
  const auto& stats = beta_stats_;
  // Ensure move_.as_string() is available or use a placeholder
  // out << move_.as_string(); 
  out << "Move_TODO";
  out << " (V:" << stats.GetVisits()
      << " Q_trad:" << std::fixed << std::setprecision(3) << stats.GetMeanValue()
      << " BetaP:" << std::fixed << std::setprecision(3) << stats.GetBetaMeanProb()
      << " BetaV:" << std::fixed << std::setprecision(3) << stats.GetBetaMeanValue()
      << " Uncert:" << std::fixed << std::setprecision(5) << stats.GetUncertainty()
      << " PolicyP:" << std::fixed << std::setprecision(3) << p_ << ")";
  return out.str();
}
#endif

#ifndef NODE_DEFINED_EXTERNALLY // If Node was defined in search.h
// Thompson Sampling edge selection implementation
uint16_t Node::SelectChildThompsonSampling(std::mt19937& rng, const SearchParams& params, float parent_q_for_fpu) const {
  if (edges_.empty()) return 0; // Or some invalid index

  float best_sample = -std::numeric_limits<float>::infinity();
  uint16_t best_idx = 0; // Default to first edge if all else fails

  // FPU value: use parent's Q if absolute FPU is not set.
  // The FPU value here is from the child's perspective (value if child is chosen).
  // If parent_q_for_fpu is -0.2 (parent is slightly losing), FPU might be +0.2 for child.
  // A simple FPU could just be params.GetFpuValue(is_root).
  // Let's use a simplified FPU for now directly from params.
  // Leela's FPU: fmax(-node->GetQ(-draw_score) - value * std::sqrt(node->GetVisitedPolicy()), -1.0f);
  // This is complex. For TS, FPU is simpler: value to add to unvisited nodes.
  // The `parent_q_for_fpu` is Q from parent's perspective. Child's FPU base is `-parent_q_for_fpu`.
  float fpu_base_q = -parent_q_for_fpu; // Child's perspective of parent's state
  float fpu_value_abs = params.GetFpuRoot(); // Assuming GetFpuRoot or similar for simplicity
                                          // Or params.GetFpuValue(parent_ == nullptr);

  for (uint16_t i = 0; i < edges_.size(); ++i) {
    const auto& edge = edges_[i];
    const auto& stats = edge.GetBetaStats();
    float sample;

    if (stats.GetVisits() == 0) {
      // First Play Urgency:
      // For TS, FPU is often an optimistic value, or P(edge) + some_const
      // The sample here is in [0,1] prob space if BetaSample is.
      // Policy prior P is a probability. FPU should be scaled.
      // A common approach: sample = policy_prior + fpu_boost
      // Let's make FPU apply in the [-1,1] value space then convert to [0,1] prob for consistency
      float effective_fpu_value;
      if (params.GetFpuAbsolute(parent_ == nullptr)) { // is_root_node
          effective_fpu_value = fpu_value_abs;
      } else {
          // Relative FPU: child's Q if unvisited = parent_value + fpu_reduction_from_parent_value
          // So, child's Q = fpu_base_q - fpu_reduction. Let's use a simpler fixed fpu for now.
          effective_fpu_value = fpu_base_q + params.GetFpuReduction(); // Example, needs proper FPU logic
      }
      
      float unvisited_value_estimate = params.GetUsePolicyPriors(parent_ == nullptr) ?
                                       (2.0f * edge.GetP() - 1.0f) : // Convert policy prob to value
                                       0.0f; // Neutral value if no policy prior

      // Add FPU to the estimated value
      // Note: this FPU logic is a simplification. Leela's original GetFpu is more complex.
      // For Thompson Sampling, often FPU is directly added to the policy prior prob.
      // Here, if SampleBeta is [0,1], then unvisited should also be [0,1]
      float fpu_for_thompson = params.GetFpuValue(parent_ == nullptr); // This is the direct UCI FPU value
      
      if (params.GetUsePolicyPriors(parent_==nullptr)) {
          sample = edge.GetP() + fpu_for_thompson; // FPU as a direct boost to policy prob
      } else {
          sample = 0.5f + fpu_for_thompson; // Neutral prior + FPU boost
      }
      // Clamp sample to be a valid probability, though FPU might push it outside [0,1]
      sample = std::max(0.0f, std::min(1.0f, sample));


    } else {
      sample = stats.SampleBeta(rng); // This is a probability in [0, 1]
    }

    if (sample > best_sample) {
      best_sample = sample;
      best_idx = i;
    }
  }
  return best_idx;
}

void Node::BackupValueToAncestors(float value, const SearchParams& params) {
  // 'value' is from the perspective of the player whose turn it was at the evaluated/terminal node.
  // It's in [-1, 1] range.
  Node* current_node = this;
  bool flip_sign = false; // Value is from current_node's perspective initially

  while (current_node != nullptr) {
    current_node->n_.fetch_add(1, std::memory_order_relaxed);
    // The node's own Q, W, L values are updated by SearchWorker::DoBackupUpdateSingleNode's main loop.
    // This method focuses on Beta stat updates for edges.

    if (current_node->parent_ != nullptr) {
      // Value needs to be from the perspective of the player at parent_node when choosing current_node
      float value_for_parent_edge = flip_sign ? -value : value;
      Edge& edge_to_current_node = current_node->parent_->GetEdge(current_node->index_in_parent_);
      edge_to_current_node.GetBetaStats().Update(value_for_parent_edge);
    }

    current_node = current_node->parent_;
    flip_sign = !flip_sign; // Value perspective flips for the parent
  }
}


void Node::CreateEdges(const std::vector<Move>& moves, const Position& pos) {
  edges_.clear(); // Clear existing edges if any (e.g., if re-expanding)
  edges_.reserve(moves.size());
  // uint16_t edge_idx = 0; // Not needed if Edge constructor doesn't take index
  for (const auto& move : moves) {
    edges_.emplace_back(move); // Assumes Edge constructor takes Move
    // edges_.back().SetMove(move); // If default constructor used
    // edge_idx++;
  }
  // Note: NN policy probabilities will be set later by SearchWorker::ExtendNode
  // And then InitializeBetaPriors will be called.
}


// Initialize Beta parameters with policy priors
void Node::InitializeBetaPriors(const std::vector<float>& policy_probs, const SearchParams& params) {
  if (!params.GetUsePolicyPriors(parent_ == nullptr) || policy_probs.size() != edges_.size()) {
    return; // Skip if disabled or sizes don't match
  }

  for (size_t i = 0; i < edges_.size(); ++i) {
    if (policy_probs[i] > 0.0f) { // Only initialize for moves with some policy
      auto& edge = edges_[i];
      edge.SetP(policy_probs[i]); // Store the raw policy prob on the edge
      auto& stats = edge.GetBetaStats();

      // Initialize Beta parameters based on policy strength
      // Alpha = 1 (prior) + wins, Beta = 1 (prior) + losses
      // We can interpret policy_prob as an initial estimate of win probability
      // And policy_temperature as the "strength" or "number of virtual games" for this prior
      float prior_strength = params.GetPolicyTemperature(parent_ == nullptr);
      float policy_p = policy_probs[i]; // This is P(action) from NN, in [0,1]

      // Atomically initialize. If visits is 0, this is the first init.
      // If multiple threads try to init, only one should win, or they should do it carefully.
      // However, InitializeBetaPriors is typically called once when node is expanded, before parallel access for selection.
      if (stats.GetVisits() == 0) { // Only initialize if not visited yet
          stats.alpha.store(1.0 + policy_p * prior_strength, std::memory_order_relaxed);
          stats.beta.store(1.0 + (1.0f - policy_p) * prior_strength, std::memory_order_relaxed);
      }
    }
  }
}

// Placeholder for SortEdgesByPolicy, if used by Thompson Sampling variant.
// Typically, TS doesn't require sorted edges for selection but might for other parts.
void Node::SortEdgesByPolicy() {
    std::sort(edges_.begin(), edges_.end(), [](const Edge& a, const Edge& b){
        return a.GetP() > b.GetP();
    });
}

float Node::GetVisitedPolicy() const {
    float p_sum = 0.0f;
    for(const auto& edge : edges_) {
        if (edge.HasNode() && edge.GetNode()->GetN() > 0) {
            p_sum += edge.GetP();
        }
    }
    return p_sum;
}

// DebugString for Node, adapted for BetaBernoulliStats
std::string Node::DebugString() const { // Assuming Node::DebugString() is needed
  std::ostringstream out;
  out << "Node State - Visits: " << GetN(); // Add more details as needed
  // out << " Q_node: " << GetQ(0.0f); // If node has its own Q value
  out << " Edges:[";
  for (uint16_t i = 0; i < edges_.size(); ++i) {
    const auto& edge = edges_[i];
    if (i > 0) out << ", ";
    out << edge.DebugString(); // Relies on Edge::DebugString
  }
  out << "]";
  return out.str();
}

#endif // NODE_DEFINED_EXTERNALLY

// **************************************************************************
// End of Node/Edge method implementations
// **************************************************************************


namespace { // Anonymous namespace from search (13).cc
// Maximum delay between outputting "uci info" when nothing interesting happens.
const int kUciInfoMinimumFrequencyMs = 5000; // Original was 5000

MoveList MakeRootMoveFilter(const MoveList& searchmoves,
                            SyzygyTablebase* syzygy_tb,
                            const PositionHistory& history, bool fast_play,
                            std::atomic<int>* tb_hits, bool* dtz_success) {
  assert(tb_hits);
  assert(dtz_success);
  // Search moves overrides tablebase.
  if (!searchmoves.empty()) return searchmoves;
  const auto& board = history.Last().GetBoard();
  MoveList root_moves;
  if (!syzygy_tb || !board.castlings().no_legal_castle() ||
      (board.ours() | board.theirs()).count() > syzygy_tb->max_cardinality()) {
    return root_moves;
  }
  if (syzygy_tb->root_probe(
          history.Last(), fast_play || history.DidRepeatSinceLastZeroingMove(),
          &root_moves)) {
    *dtz_success = true;
    tb_hits->fetch_add(1, std::memory_order_acq_rel);
  } else if (syzygy_tb->root_probe_wdl(history.Last(), &root_moves)) {
    tb_hits->fetch_add(1, std::memory_order_acq_rel);
  }
  return root_moves;
}

class MEvaluator {
 public:
  MEvaluator()
      : enabled_{false},
        m_slope_{0.0f},
        m_cap_{0.0f},
        a_constant_{0.0f},
        a_linear_{0.0f},
        a_square_{0.0f},
        q_threshold_{0.0f},
        parent_m_{0.0f} {}

  MEvaluator(const SearchParams& params, const Node* parent = nullptr)
      : enabled_{true}, // Assuming MLH is generally enabled if params support it.
        m_slope_{params.GetMovesLeftSlope()},
        m_cap_{params.GetMovesLeftMaxEffect()},
        a_constant_{params.GetMovesLeftConstantFactor()},
        a_linear_{params.GetMovesLeftScaledFactor()},
        a_square_{params.GetMovesLeftQuadraticFactor()},
        q_threshold_{params.GetMovesLeftThreshold()},
        parent_m_{parent ? parent->GetM() : 0.0f}, // Assumes Node::GetM()
        parent_within_threshold_{parent ? WithinThreshold(parent, q_threshold_)
                                        : false} {}

  void SetParent(const Node* parent) {
    assert(parent);
    if (enabled_) {
      parent_m_ = parent->GetM(); // Assumes Node::GetM()
      parent_within_threshold_ = WithinThreshold(parent, q_threshold_);
    }
  }

  // Calculates the utility for favoring shorter wins and longer losses.
  // q is the value of the child from parent's perspective [-1, 1]
  float GetMUtility(const Node* child_node, float q_from_parent_perspective) const { // Takes Node*
    if (!enabled_ || !parent_within_threshold_ || !child_node) return 0.0f;
    // const float child_m = child_node->GetM(); // Assumes Node::GetM()
    // This needs to be an actual M value from the child node if it has one,
    // or estimated. For simplicity, let's assume it's available.
    float child_m = 0.0f; // Placeholder, needs real child M value
    if (child_node->HasNode()) child_m = child_node->GetNode()->GetM();


    float m_effect = std::clamp(m_slope_ * (child_m - parent_m_), -m_cap_, m_cap_);
    m_effect *= FastSign(-q_from_parent_perspective); // If parent is winning (q_f_p_p > 0), negative q_f_p_p means child is good for parent.
                                         // We want shorter wins (child_m < parent_m => m_effect becomes positive if q_f_p_p is large positive)
                                         // And longer losses (child_m > parent_m => m_effect becomes positive if q_f_p_p is large negative)

    float abs_q = std::abs(q_from_parent_perspective);
    if (q_threshold_ > 0.0f && q_threshold_ < 1.0f) {
      abs_q = std::max(0.0f, (abs_q - q_threshold_)) / (1.0f - q_threshold_);
    }
    m_effect *= a_constant_ + a_linear_ * abs_q + a_square_ * abs_q * abs_q; // Use abs_q for scaling factor
    return m_effect;
  }

  float GetMUtility(const EdgeAndNode& child_edge_and_node, float q_from_parent_perspective) const {
    if (!enabled_ || !parent_within_threshold_) return 0.0f;
    if (child_edge_and_node.GetN() == 0 || !child_edge_and_node.node()) return GetDefaultMUtility();
    return GetMUtility(child_edge_and_node.node(), q_from_parent_perspective);
  }


  // The M utility to use for unvisited nodes.
  float GetDefaultMUtility() const { return 0.0f; }

 private:
  static bool WithinThreshold(const Node* parent, float q_threshold) {
    return std::abs(parent->GetQ(0.0f)) > q_threshold; // Assumes Node::GetQ()
  }

  const bool enabled_;
  const float m_slope_;
  const float m_cap_;
  const float a_constant_;
  const float a_linear_;
  const float a_square_;
  const float q_threshold_;
  float parent_m_ = 0.0f;
  bool parent_within_threshold_ = false;
};

}  // namespace


Search::Search(NodeTree* dag, Network* network,
               std::unique_ptr<UciResponder> uci_responder,
               const MoveList& searchmoves,
               std::chrono::steady_clock::time_point start_time,
               std::unique_ptr<SearchStopper> stopper, bool infinite,
               bool ponder, const OptionsDict& options, NNCache* cache,
               SyzygyTablebase* syzygy_tb)
    : ok_to_respond_bestmove_(!infinite && !ponder),
      stopper_(std::move(stopper)),
      root_node_(dag->GetCurrentHead()),
      cache_(cache),
      dag_(dag),
      syzygy_tb_(syzygy_tb),
      played_history_(dag->GetPositionHistory()),
      network_(network),
      params_(options), // This initializes SearchParams from OptionsDict
      searchmoves_(searchmoves),
      start_time_(start_time),
      initial_visits_(root_node_->GetN()),
      root_move_filter_(MakeRootMoveFilter(
          searchmoves_, syzygy_tb_, played_history_,
          params_.GetSyzygyFastPlay(), &tb_hits_, &root_is_in_dtz_)),
      uci_responder_(std::move(uci_responder)),
      rng_(std::random_device{}()) // Initialize RNG for Search class
      {
  // Ensure SearchParams now includes Thompson Sampling parameters
  // params_.SetPolicyTemperature(options.GetOrDefault<float>("PolicyTemperature", 1.0f));
  // params_.SetUsePolicyPriors(options.GetOrDefault<bool>("UsePolicyPriors", true));
  // These would be set inside SearchParams constructor or via setters if SearchParams is augmented.

  if (params_.GetMaxConcurrentSearchers() != 0) {
    pending_searchers_.store(params_.GetMaxConcurrentSearchers(),
                             std::memory_order_release);
  }
  contempt_mode_ = params_.GetContemptMode();
  // Make sure the contempt mode is never "play" beyond this point.
  if (contempt_mode_ == ContemptMode::PLAY) {
    if (infinite) {
      // For infinite search disable contempt, only "white"/"black" make sense.
      contempt_mode_ = ContemptMode::NONE;
      // Issue a warning only if contempt mode would have an effect.
      if (params_.GetWDLRescaleDiff() != 0.0f) {
        std::vector<ThinkingInfo> info(1);
        info.back().comment =
            "WARNING: Contempt mode set to 'disable' as 'play' not supported "
            "for infinite search.";
        uci_responder_->OutputThinkingInfo(&info);
      }
    } else {
      // Otherwise set it to the root move's side, unless pondering.
      contempt_mode_ = played_history_.IsBlackToMove() != ponder
                           ? ContemptMode::BLACK
                           : ContemptMode::WHITE;
    }
  }
}

namespace { // Anonymous namespace from search (13).cc
void ApplyDirichletNoise(Node* node, float eps, double alpha, const SearchParams& params) {
  // This function applies noise to policy priors (P values on edges)
  // For Thompson Sampling, this noise should ideally affect the initial Beta parameters
  // if policy priors are used to initialize them.
  // If InitializeBetaPriors is called *after* this, it will use the noised P values.
  float total = 0;
  std::vector<float> dirichlet_noise_values; // Renamed from 'noise' to avoid conflict

  for (int i = 0; i < node->GetNumEdges(); ++i) {
    float eta = Random::Get().GetGamma(alpha, 1.0);
    dirichlet_noise_values.emplace_back(eta);
    total += eta;
  }

  if (total < std::numeric_limits<float>::min()) return;

  int noise_idx = 0;
  // Assuming node->Edges() gives access to modifiable Edges or EdgeAndNodes with modifiable Edges.
  // If Node::Iterator is used, it should allow modification.
  // Let's assume Node has a way to iterate and modify its edges' P values.
  for (uint16_t i = 0; i < node->GetNumEdges(); ++i) {
    Edge& edge = node->GetEdge(i); // Assumes Node::GetEdge(idx) exists
    edge.SetP(edge.GetP() * (1.0f - eps) + eps * dirichlet_noise_values[noise_idx++] / total);
  }
  // After P values are noised, if Beta priors depend on P, they will use these noised values.
  // Re-initialize Beta priors if they were already set, or ensure this runs before initialization.
  // Option: Call node->InitializeBetaPriors again if it's safe and desired.
  // For now, assume InitializeBetaPriors will be called after noise application if node is new.
}
}  // namespace

namespace { // Anonymous namespace from search (13).cc
// WDL conversion formula based on random walk model.
inline double WDLRescale(float& v, float& d, float wdl_rescale_ratio,
                         float wdl_rescale_diff, float sign, bool invert) {
  if (invert) {
    wdl_rescale_diff = -wdl_rescale_diff;
    wdl_rescale_ratio = 1.0f / wdl_rescale_ratio;
  }
  auto w = (1 + v - d) / 2;
  auto l = (1 - v - d) / 2;
  // Safeguard against numerical issues; skip WDL transformation if WDL is too
  // extreme.
  const float eps = 0.0001f;
  if (w > eps && d > eps && l > eps && w < (1.0f - eps) && d < (1.0f - eps) &&
      l < (1.0f - eps)) {
    auto a = FastLog(1 / l - 1);
    auto b = FastLog(1 / w - 1);
    auto s = 2 / (a + b);
    // Safeguard against unrealistically broad WDL distributions coming from
    // the NN. Could be made into a parameter, but probably unnecessary.
    const float max_reasonable_s = 1.4f;
    if (!invert) s = std::min(max_reasonable_s, s);
    auto mu = (a - b) / (a + b);
    auto s_new = s * wdl_rescale_ratio;
    if (invert) {
      std::swap(s, s_new);
      s = std::min(max_reasonable_s, s);
    }
    auto mu_new = mu + sign * s * s * wdl_rescale_diff;
    auto w_new = FastLogistic((-1.0f + mu_new) / s_new);
    auto l_new = FastLogistic((-1.0f - mu_new) / s_new);
    v = w_new - l_new;
    d = std::max(0.0f, 1.0f - w_new - l_new);
    return mu_new;
  }
  return 0;
}
}  // namespace

void Search::SendUciInfo() REQUIRES(nodes_mutex_) REQUIRES(counters_mutex_) {
  const auto max_pv = params_.GetMultiPv();
  // GetBestChildrenNoTemperature now sorts by visits then BetaMeanValue for TS
  const auto edges_and_nodes = GetBestChildrenNoTemperature(root_node_, max_pv, 0);
  const auto score_type = params_.GetScoreType();
  const auto per_pv_counters = params_.GetPerPvCounters();
  const auto display_cache_usage = params_.GetDisplayCacheUsage();
  const auto draw_score = GetDrawScore(false); // Draw score from root's perspective

  std::vector<ThinkingInfo> uci_infos;

  // Info common for all multipv variants.
  ThinkingInfo common_info;
  common_info.depth = cum_depth_ / (total_playouts_ ? total_playouts_ : 1);
  common_info.seldepth = max_depth_;
  common_info.time = GetTimeSinceStart();
  uint64_t total_nodes_reported; // Renamed from total_nodes to avoid conflict
  std::string reported_nodes_str = params_.GetReportedNodes(); // Renamed from reported_nodes
  if (reported_nodes_str == "nodes") {
    total_nodes_reported = total_low_nodes_;
  } else if (reported_nodes_str == "queries") {
    total_nodes_reported = total_nn_queries_;
  } else { // "playouts" || "legacy"
    total_nodes_reported = total_playouts_;
  }

  if (!per_pv_counters) {
    common_info.nodes = total_playouts_ + initial_visits_;
  }
  if (nps_start_time_) {
    const auto time_since_first_batch_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - *nps_start_time_)
            .count();
    if (time_since_first_batch_ms > 0) {
      common_info.nps = total_nodes_reported * 1000 / time_since_first_batch_ms;
    }
  }
  if (display_cache_usage && cache_) { // Added cache_ check
    common_info.hashfull =
        cache_->GetSize() * 1000LL / std::max(1LL, cache_->GetCapacity()); // Ensure GetCapacity() is not 0
  }
  common_info.tb_hits = tb_hits_.load(std::memory_order_acquire);

  int multipv = 0;
  // Default values for Q, WL, D if an edge has no node (should not happen for top PVs)
  // These are from the child's perspective.
  // Root's Q is from root's perspective. Child's Q is -root_node->GetQ(...)
  const float default_q_child_perspective = root_node_->GetN() > 0 ? -root_node_->GetQ(-draw_score) : 0.0f;
  const float default_wl_child_perspective = root_node_->GetN() > 0 ? -root_node_->GetWL() : 0.0f; // Assumes Node::GetWL()
  const float default_d_child_perspective = root_node_->GetN() > 0 ? root_node_->GetD() : 0.0f;   // Assumes Node::GetD()


  for (const auto& edge_and_node : edges_and_nodes) {
    ++multipv;
    uci_infos.emplace_back(common_info);
    auto& uci_info = uci_infos.back();

    const Edge* edge_ptr = edge_and_node.edge();
    if (!edge_ptr) continue; // Should not happen
    const Node* child_node = edge_and_node.node();

    // Get evaluation from child's perspective (value of the state after this move)
    // For Thompson Sampling, the primary value is BetaMean.
    // We need BetaMeanValue ([-1, 1]) from the child's perspective.
    float q_val_child, wl_val_child, d_val_child;

    if (child_node && child_node->GetN() > 0) {
        // Use BetaMeanValue from the edge leading to this child. This is already from child's perspective.
        wl_val_child = edge_ptr->GetBetaStats().GetBetaMeanValue();
        // For Q and D, we might need to use the child node's own NN eval if available,
        // or derive from BetaMean / traditional Q.
        // For simplicity, let's use traditional Q for q_val_child if needed by score_type.
        q_val_child = edge_ptr->GetBetaStats().GetMeanValue(); // Traditional Q
        d_val_child = child_node->GetD(); // Or some default/derived D
    } else { // Unvisited or no node, use defaults
        wl_val_child = default_wl_child_perspective;
        q_val_child = default_q_child_perspective;
        d_val_child = default_d_child_perspective;
    }


    float mu_uci = 0.0f;
    if (score_type == "WDL_mu" || (params_.GetWDLRescaleDiff() != 0.0f &&
                                   contempt_mode_ != ContemptMode::NONE)) {
      // WDLRescale expects v (wl) and d from child's perspective
      float temp_wl = wl_val_child;
      float temp_d = d_val_child;
      auto sign = ((contempt_mode_ == ContemptMode::BLACK) ==
                   played_history_.IsBlackToMove()) // This logic for sign might need review
                      ? 1.0f                       // It depends on whose perspective mu_uci is for
                      : -1.0f;
      // If wl_val_child is from child's perspective, and played_history.IsBlackToMove is parent,
      // then sign should probably be based on parent's color vs contempt target.
      // Assuming mu_uci is from parent's perspective, so wl_val_child (child's val) is used.
      mu_uci = WDLRescale(
          temp_wl, temp_d, params_.GetWDLRescaleRatio(),
          contempt_mode_ == ContemptMode::NONE
              ? 0
              : params_.GetWDLRescaleDiff() * params_.GetWDLEvalObjectivity(),
          sign, true); // Invert = true means we are converting from WDL to mu for reporting
    }

    // Score reported is from the perspective of the player to move (parent node)
    // So, if wl_val_child is from child's perspective, we need to negate it.
    // However, UCI score is conventionally positive for current player winning.
    // If edge_and_node.GetWL() was from parent's perspective, this would be simpler.
    // Let's assume wl_val_child is value of child state. UCI score wants value of making that move.
    // If child state is good (wl_val_child high), then move is good.

    if (edge_and_node.IsTerminal() && wl_val_child != 0.0f) { // IsTerminal from EdgeAndNode
      uci_info.mate = std::copysign(
          std::round(edge_and_node.GetM(0.0f) + 1) / 2 + (edge_and_node.IsTbTerminal() ? 100 : 0),
          wl_val_child); // Use child's WL for sign of mate
    } else if (score_type == "centipawn_with_drawscore") {
      uci_info.score = 90 * tan(1.5637541897 * q_val_child); // q_val_child is child's Q
    } else if (score_type == "centipawn") {
      uci_info.score = 90 * tan(1.5637541897 * wl_val_child); // Use BetaMeanValue (child's WL)
    } else if (score_type == "centipawn_2019") {
      uci_info.score = 295 * wl_val_child / (1 - 0.976953126 * std::pow(wl_val_child, 14));
    } else if (score_type == "centipawn_2018") {
      uci_info.score = 290.680623072 * tan(1.548090806 * wl_val_child);
    } else if (score_type == "win_percentage") {
      uci_info.score = wl_val_child * 5000 + 5000;
    } else if (score_type == "Q") {
      uci_info.score = q_val_child * 10000;
    } else if (score_type == "W-L") { // This usually refers to parent's perspective
      uci_info.score = wl_val_child * 10000; // Using child's perspective BetaMeanValue
    } else if (score_type == "WDL_mu") {
      const float centipawn_fallback_threshold = 0.996f;
      float centipawn_score = 90 * tan(1.5637541897 * wl_val_child);
      uci_info.score =
          mu_uci != 0.0f && std::abs(wl_val_child) + d_val_child < centipawn_fallback_threshold &&
                  (std::abs(mu_uci) < 1.0f ||
                   std::abs(centipawn_score) < std::abs(100 * mu_uci))
              ? 100 * mu_uci
              : centipawn_score;
    }


    // WDL stats: W, D, L probabilities from child's perspective
    // wl_val_child = W - L
    // d_val_child = D
    // W + L + D = 1  => W + L = 1 - D
    // W = (wl_val_child + 1 - d_val_child) / 2
    // L = (1 - d_val_child - wl_val_child) / 2
    auto wdl_w_prob = (wl_val_child + 1.0f - d_val_child) / 2.0f;
    auto wdl_l_prob = (1.0f - d_val_child - wl_val_child) / 2.0f;
    auto wdl_d_prob = d_val_child;

    auto wdl_w = std::max(0, static_cast<int>(std::round(1000.0f * wdl_w_prob)));
    auto wdl_l = std::max(0, static_cast<int>(std::round(1000.0f * wdl_l_prob)));
    auto wdl_d = std::max(0, static_cast<int>(std::round(1000.0f * wdl_d_prob)));
    
    // Normalize to sum to 1000
    int total_wdl = wdl_w + wdl_d + wdl_l;
    if (total_wdl != 1000 && total_wdl != 0) {
        wdl_w = static_cast<int>(std::round(static_cast<float>(wdl_w) * 1000.0f / total_wdl));
        wdl_d = static_cast<int>(std::round(static_cast<float>(wdl_d) * 1000.0f / total_wdl));
        wdl_l = 1000 - wdl_w - wdl_d; // Ensure sum is 1000
    } else if (total_wdl == 0) { // Avoid division by zero if all probs are tiny
        wdl_w = 333; wdl_d = 334; wdl_l = 333; // Default to roughly even
    }


    uci_info.wdl = ThinkingInfo::WDL{wdl_w, wdl_d, wdl_l};
    if (network_->GetCapabilities().has_mlh() && child_node) { // Check child_node
      uci_info.moves_left = static_cast<int>(
          // (1.0f + edge_and_node.GetM(1.0f + root_node_->GetM())) / 2.0f); // GetM is on EdgeAndNode
          (1.0f + child_node->GetM()) / 2.0f); // Assumes Node::GetM() for child
    }
    if (max_pv > 1) uci_info.multipv = multipv;
    if (per_pv_counters) uci_info.nodes = edge_ptr->GetBetaStats().GetVisits(); // Visits from BetaStats

    // PV construction needs care: moves are from alternating perspectives.
    bool current_player_is_black = played_history_.IsBlackToMove();
    auto temp_history = played_history_; // Copy for PV simulation
    Node* pv_node = root_node_;
    const Edge* pv_edge = edge_ptr; // Start with the current MultiPV edge

    for (int pv_depth = 0; pv_depth < 64; ++pv_depth) { // Limit PV length
        if (!pv_edge) break;
        Move current_move = pv_edge->GetMove(current_player_is_black);
        uci_info.pv.push_back(current_move);
        temp_history.Append(current_move); // Use the actual move played

        Node* next_node_in_pv = pv_edge->GetNode();
        if (!next_node_in_pv || !next_node_in_pv->HasChildren() || next_node_in_pv->IsTerminal()) {
            break;
        }
        
        // Select best child from next_node_in_pv (usually most visited for PV)
        // This needs a GetBestChildNoTemperature equivalent that takes Node*
        // For now, let's assume GetBestChildNoTemperature can be adapted or we use a simple most visited.
        uint32_t max_visits_pv = 0;
        const Edge* next_pv_edge = nullptr;
        if (next_node_in_pv->GetNumEdges() > 0) {
            // Simplified: find most visited. Real PV construction might be more complex.
            uint16_t best_child_idx = 0;
             for (uint16_t k=0; k<next_node_in_pv->GetNumEdges(); ++k) {
                 if (next_node_in_pv->GetEdge(k).GetBetaStats().GetVisits() > max_visits_pv) {
                     max_visits_pv = next_node_in_pv->GetEdge(k).GetBetaStats().GetVisits();
                     best_child_idx = k;
                 }
             }
            if (max_visits_pv > 0) { // Only continue if there's a visited child
                 next_pv_edge = &next_node_in_pv->GetEdge(best_child_idx);
            }
        }
        
        pv_edge = next_pv_edge;
        pv_node = next_node_in_pv; // Update pv_node, though not directly used in loop
        current_player_is_black = !current_player_is_black; // Flip player for next move in PV

        if (temp_history.Last().GetRepetitions() >= 2) break; // Stop PV on repetition
    }
  }


  if (!uci_infos.empty()) last_outputted_uci_info_ = uci_infos.front();
  if (current_best_edge_ && current_best_edge_.edge() && !edges_and_nodes.empty()) { // Check current_best_edge_.edge()
    last_outputted_info_edge_ = current_best_edge_.edge();
  }

  uci_responder_->OutputThinkingInfo(&uci_infos);
}

// Decides whether anything important changed in stats and new info should be
// shown to a user.
void Search::MaybeOutputInfo() {
  SharedMutex::Lock lock(nodes_mutex_);
  Mutex::Lock counters_lock(counters_mutex_);
  if (!bestmove_is_sent_ && current_best_edge_ && current_best_edge_.edge() && // check current_best_edge_.edge()
      (current_best_edge_.edge() != last_outputted_info_edge_ ||
       last_outputted_uci_info_.depth !=
           static_cast<int>(cum_depth_ /
                            (total_playouts_ ? total_playouts_ : 1)) ||
       last_outputted_uci_info_.seldepth != max_depth_ ||
       last_outputted_uci_info_.time + kUciInfoMinimumFrequencyMs <
           GetTimeSinceStart())) {
    SendUciInfo();
    if (params_.GetLogLiveStats()) {
      SendMovesStats();
    }
    if (stop_.load(std::memory_order_acquire) && !ok_to_respond_bestmove_) {
      std::vector<ThinkingInfo> info(1);
      info.back().comment =
          "WARNING: Search has reached limit and does not make any progress.";
      uci_responder_->OutputThinkingInfo(&info);
    }
  }
}

int64_t Search::GetTimeSinceStart() const {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now() - start_time_)
      .count();
}

int64_t Search::GetTimeSinceFirstBatch() const REQUIRES(counters_mutex_) {
  if (!nps_start_time_) return 0;
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now() - *nps_start_time_)
      .count();
}

// Root is depth 0, i.e. even depth.
float Search::GetDrawScore(bool is_odd_depth) const {
  return (is_odd_depth == played_history_.IsBlackToMove()
              ? params_.GetDrawScore()
              : -params_.GetDrawScore());
}


namespace { // Anonymous namespace from search (13).cc (PUCT factor computation)
          // This section will be largely unused or simplified for Thompson Sampling.
          // PUCT components like ComputeCpuctFactor, ComputeExploreFactor, etc.,
          // are not directly part of Thompson Sampling's selection criteria.
          // FPU is still used for unvisited nodes.

// For Thompson Sampling, we primarily need FPU.
// The original GetFpu from search (13).cc:
inline float GetFpuForThompson(const SearchParams& params, Node* node, bool is_root_node,
                               float draw_score, float visited_pol_sum) {
  // For Thompson Sampling, FPU is typically a value added to the policy prior of unvisited nodes
  // or a fixed optimistic value. It's simpler than PUCT's FPU.
  // The 'value' from params.GetFpuValue is the direct additive FPU.
  float fpu_val = params.GetFpuValue(is_root_node);

  if (params.GetFpuAbsolute(is_root_node)) {
      return fpu_val; // Returns the value that will be added, e.g. to policy prob
  } else {
      // Relative FPU for Thompson sampling is less common.
      // If FPU is relative to parent's Q: Q_child_unvisited = -Q_parent - fpu_reduction
      // The fpu_val from GetFpuValue can be this reduction or added term.
      // For simplicity, if not absolute, it might mean a reduction from a neutral prior (0.5 for prob).
      // This part needs clear definition for Thompson Sampling.
      // Let's assume GetFpuValue() returns the direct value to be used/added.
      // The fpu value in `SelectChildThompsonSampling` is directly used.
      // So this GetFpu function might just return params.GetFpuValue(is_root_node).
      // However, the original GetFpu depends on parent_q and visited_policy.
      // Let's keep it simple: return the UCI FPU value.
      // The selection logic `SelectChildThompsonSampling` will use it as needed.
      return fpu_val;
  }
}


// ComputePolicyDecayFactor and ComputePolicyDecay might still be relevant if policy priors are decayed over time.
// Thompson Sampling can use raw or decayed policy for FPU or prior initialization.
inline float ComputePolicyDecayFactor(const SearchParams& params, uint32_t N_node_visits) {
  const float exponent = params.GetPolicyDecayExponent();
  const float proportionality_factor = params.GetPolicyDecayFactor();
  return (exponent == 0.0f || proportionality_factor == 0.0f)
             ? 1.0f
             : FastExp(-FastLog(1.0f + proportionality_factor * N_node_visits) * exponent);
}
inline float ComputePolicyDecay(const float factor, const float policy_prior) {
  return factor == 1.0f ? policy_prior : policy_prior / (policy_prior + (1.0f - policy_prior) * factor);
}

}  // namespace

std::vector<std::string> Search::GetVerboseStats(Node* node) const REQUIRES_SHARED(nodes_mutex_) {
  const bool is_root = (node == root_node_);
  // const bool is_odd_depth = !is_root; // Depth relative to current node for draw score
  const float node_perspective_draw_score = GetDrawScore(played_history_.IsBlackToMove() != is_root); // Draw score for *this* node's eval

  // For Thompson Sampling, FPU is used in selection. For display, show raw stats.
  // float fpu_for_display = GetFpuForThompson(params_, node, is_root, node_perspective_draw_score, node->GetVisitedPolicy());

  std::vector<EdgeAndNode> edges_and_nodes_vec; // Renamed from edges
  for (const auto& edge_and_node_item : node->Edges()) edges_and_nodes_vec.push_back(edge_and_node_item);

  // Sort by a relevant metric for display, e.g., visits or Beta mean.
  std::sort(
      edges_and_nodes_vec.begin(), edges_and_nodes_vec.end(),
      [](const EdgeAndNode& a, const EdgeAndNode& b) {
        if (!a.edge() || !b.edge()) return !a.edge(); // Null edges last
        // Primary sort: visits descending
        if (a.edge()->GetBetaStats().GetVisits() != b.edge()->GetBetaStats().GetVisits()) {
          return a.edge()->GetBetaStats().GetVisits() > b.edge()->GetBetaStats().GetVisits();
        }
        // Secondary sort: BetaMeanValue descending (higher is better for the player whose turn it is at child)
        return a.edge()->GetBetaStats().GetBetaMeanValue() > b.edge()->GetBetaStats().GetBetaMeanValue();
      });

  auto print_val = [](auto* oss, auto pre, auto v, auto post, auto w, int p = 0) { // Renamed from print
    *oss << pre << std::setw(w) << std::setprecision(p) << std::fixed << v << post;
  };
  auto print_header = [&](auto* oss, auto label, int i, auto n_visits_edge, auto n_inflight_edge, auto policy_p_edge, auto check_char) { // Renamed from print_head
    *oss << std::fixed;
    print_val(oss, "", label, " ", 5);
    // print_val(oss, "(", i, ") ", 4); // NN index might not be readily available on EdgeAndNode
    *oss << std::right;
    print_val(oss, "N: ", n_visits_edge, " ", 7);
    print_val(oss, "(+", n_inflight_edge, ") ", 2);
    print_val(oss, "(P: ", policy_p_edge * 100, "%) ", 5, policy_p_edge >= 0.99995f ? 1 : 2);
    // print_val(oss, "C: ", check_char, " ", 4); // Check status might not be on EdgeAndNode
  };
  auto print_edge_stats = [&](auto* oss, const Edge* edge_ptr, const Node* child_node_ptr) { // Renamed from print_stats
    // Values from child's perspective
    if (edge_ptr) {
        const auto& beta_stats = edge_ptr->GetBetaStats();
        print_val(oss, "(V:", beta_stats.GetVisits(), ") ", 7);
        print_val(oss, "(PolP:", edge_ptr->GetP(), ") ", 8, 5);
        print_val(oss, "(BetaMeanV:", beta_stats.GetBetaMeanValue(), ") ", 12, 5); // Value in [-1,1]
        print_val(oss, "(BetaMeanP:", beta_stats.GetBetaMeanProb(), ") ", 12, 5); // Prob in [0,1]
        print_val(oss, "(TradQ:", beta_stats.GetMeanValue(), ") ", 8, 5);
        print_val(oss, "(Uncert:", beta_stats.GetUncertainty(), ")", 10, 7);
        if (child_node_ptr) {
            // print_val(oss, "(ChildN_WL:", child_node_ptr->GetWL(),") ", 10,5); // If Node has GetWL
            // print_val(oss, "(ChildN_D:", child_node_ptr->GetD(), ") ", 8,5);   // If Node has GetD
            // print_val(oss, "(ChildN_M:", child_node_ptr->GetM(), ") ", 8,1);   // If Node has GetM
        }
    } else {
      *oss << "(V:0) (PolP: -.-----) (BetaMeanV:  -.-----) ...";
    }
  };
  // print_tail remains similar

  std::vector<std::string> infos;
  // const auto m_evaluator = network_->GetCapabilities().has_mlh() ? MEvaluator(params_, node) : MEvaluator(); // if combining with MLH

  for (const auto& edge_and_node_item : edges_and_nodes_vec) {
    const Edge* edge_ptr = edge_and_node_item.edge();
    const Node* child_node_ptr = edge_and_node_item.node();
    if (!edge_ptr) continue;

    // For Thompson Sampling, the "score" used for selection is the sample.
    // For display, we show BetaMean or Visits.
    // float selection_score_display = edge_ptr->GetBetaStats().GetBetaMeanValue(); // Example
    // float m_utility_display = m_evaluator.GetMUtility(edge_and_node_item, selection_score_display); // If MLH combined

    std::ostringstream oss;
    oss << std::left;
    print_header(&oss, edge_ptr->GetMove(played_history_.IsBlackToMove()).as_string(),
               0, // Placeholder for NN index if needed
               edge_ptr->GetBetaStats().GetVisits(),
               child_node_ptr ? child_node_ptr->GetNInFlight() : 0, // Assumes Node::GetNInFlight
               edge_ptr->GetP(),
               ' '); // Placeholder for check status
    print_edge_stats(&oss, edge_ptr, child_node_ptr);
    // print_val(&oss, "(DisplayScore: ", selection_score_display + m_utility_display, ") ", 8, 5); // If showing combined score
    // print_tail(&oss, child_node_ptr); // print_tail expects Node*
    infos.emplace_back(oss.str());
  }

  // Include stats about the node itself
  std::ostringstream oss_node_stats; // Renamed from oss
  // print_header(&oss_node_stats, "node ", node->GetNumEdges(), node->GetN(), node->GetNInFlight(), node->GetVisitedPolicy(), false);
  // print_stats for node itself: Here, node's Q,WL,D,M are primary. Edges are children.
  // print_tail(&oss_node_stats, node);
  oss_node_stats << std::endl << "Low nodes: " << total_low_nodes_
       << " NN queries: " << total_nn_queries_
       << " Playouts: " << total_playouts_ + initial_visits_ << std::endl;
  infos.emplace_back(oss_node_stats.str());
  return infos;
}


void Search::SendMovesStats() const REQUIRES(nodes_mutex_) REQUIRES(counters_mutex_) {
  auto move_stats = GetVerboseStats(root_node_);

  if (params_.GetVerboseStats()) {
    std::vector<ThinkingInfo> infos;
    std::transform(move_stats.begin(), move_stats.end(),
                   std::back_inserter(infos), [](const std::string& line) {
                     ThinkingInfo info;
                     info.comment = line;
                     return info;
                   });
    uci_responder_->OutputThinkingInfo(&infos);
  } else {
    LOGFILE << "=== Move stats:";
    for (const auto& line : move_stats) LOGFILE << line;
  }
  // This part for opponent moves might need adjustment depending on how GetVerboseStats is structured
  // And if final_bestmove_ is from EdgeAndNode or just Move.
  if (final_bestmove_.is_valid()) { // Check if final_bestmove_ is valid
      for (auto& edge_and_node_item : root_node_->Edges()) { // Iterate over EdgeAndNode
        if (!edge_and_node_item.edge()) continue;
        if (edge_and_node_item.edge()->GetMove(played_history_.IsBlackToMove()) == final_bestmove_) {
          if (edge_and_node_item.node()) { // Check if node exists
            LOGFILE << "--- Opponent moves after: " << final_bestmove_.as_string();
            for (const auto& line : GetVerboseStats(edge_and_node_item.node())) {
              LOGFILE << line;
            }
          }
          break; 
        }
      }
  }
}


NNCacheLock Search::GetCachedNNEval(const PositionHistory& history) const {
  const auto hash_val = dag_->GetHistoryHash(history); // Renamed from hash
  NNCacheLock nneval(cache_, hash_val);
  return nneval;
}

void Search::MaybeTriggerStop(const IterationStats& stats,
                              StoppersHints* hints) {
  hints->Reset();
  if (params_.GetNpsLimit() > 0) {
    hints->UpdateEstimatedNps(params_.GetNpsLimit());
  }
  SharedMutex::Lock nodes_lock(nodes_mutex_); // Changed to full lock if EnsureBestMoveKnown modifies
  Mutex::Lock lock(counters_mutex_);
  // Already responded bestmove, nothing to do here.
  if (bestmove_is_sent_) return;
  // Don't stop when the root node is not yet expanded.
  if (total_playouts_ + initial_visits_ == 0 && root_node_->GetN() == 0) return; // Check root_node_->GetN() too

  if (!stop_.load(std::memory_order_acquire)) {
    if (stopper_->ShouldStop(stats, hints)) FireStopInternal();
  }

  // If we are the first to see that stop is needed.
  if (stop_.load(std::memory_order_acquire) && ok_to_respond_bestmove_ &&
      !bestmove_is_sent_) {
    // SendUciInfo requires nodes_mutex_ and counters_mutex_
    // EnsureBestMoveKnown requires nodes_mutex_ and counters_mutex_
    // SendMovesStats requires nodes_mutex_ and counters_mutex_
    // These locks are already held.
    SendUciInfo();
    EnsureBestMoveKnown(); // This now also needs nodes_mutex_
    SendMovesStats();
    BestMoveInfo info(final_bestmove_, final_pondermove_);
    uci_responder_->OutputBestMove(&info);
    stopper_->OnSearchDone(stats);
    bestmove_is_sent_ = true;
    current_best_edge_ = EdgeAndNode();
  }
}

// Return the evaluation of the actual best child, regardless of temperature
// settings. This differs from GetBestMove, which does obey any temperature
// settings. So, somethimes, they may return results of different moves.
Eval Search::GetBestEval(Move* move_ptr, bool* is_terminal_ptr) const { // Renamed params
  SharedMutex::SharedLock lock(nodes_mutex_); // Keep as SharedLock if GetBestChildNoTemperature is const and thread-safe for reads
  Mutex::Lock counters_lock(counters_mutex_); // EnsureBestMoveKnown might need full lock on nodes_mutex_ if it modifies
  
  // Eval from parent's perspective.
  // Node::GetWL, GetD, GetM are from the node's own perspective.
  // If root_node_ is for player P, its WL is P's win likelihood.
  // If we pick a child C, C's WL is for player Opponent(P).
  // So, best_edge.GetWL() should be interpreted correctly.
  // Let's assume EdgeAndNode::GetWL returns value from child's perspective.
  // Then parent's eval is -child_eval.

  float parent_wl = root_node_->GetN() > 0 ? root_node_->GetWL() : 0.0f; // Root's own WL (perspective of root player)
  float parent_d = root_node_->GetN() > 0 ? root_node_->GetD() : 0.0f;
  float parent_m = root_node_->GetN() > 0 ? root_node_->GetM() : 0.0f;

  if (!root_node_->HasChildren()) return {parent_wl, parent_d, parent_m}; // Return root's eval if no children

  EdgeAndNode best_edge_and_node = GetBestChildNoTemperature(root_node_, 0); // Renamed from best_edge
  if (!best_edge_and_node.edge()) return {parent_wl, parent_d, parent_m}; // No valid best child

  if (move_ptr) *move_ptr = best_edge_and_node.edge()->GetMove(played_history_.IsBlackToMove());
  if (is_terminal_ptr) *is_terminal_ptr = best_edge_and_node.IsTerminal();

  // Get WL, D, M from the child's perspective (i.e., value of the resulting state)
  // For Thompson Sampling, this would be the BetaMeanValue of the edge.
  float child_wl = best_edge_and_node.edge()->GetBetaStats().GetBetaMeanValue();
  float child_d = best_edge_and_node.node() ? best_edge_and_node.node()->GetD() : default_d_child_perspective; // Default if no node
  float child_m = best_edge_and_node.node() ? best_edge_and_node.node()->GetM() : 0.0f;
  
  // The returned Eval should be from the parent's (root_node's) perspective.
  // If child_wl is good for child (e.g., +0.8), it's bad for parent (-0.8).
  // So, eval for parent is -child_wl.
  return {-child_wl, child_d, child_m + 1.0f}; // M is plies to mate from child, so parent is child_m+1
}


std::pair<Move, Move> Search::GetBestMove() {
  SharedMutex::Lock lock(nodes_mutex_); // EnsureBestMoveKnown modifies, so full lock
  Mutex::Lock counters_lock(counters_mutex_);
  EnsureBestMoveKnown();
  return {final_bestmove_, final_pondermove_};
}

std::int64_t Search::GetTotalPlayouts() const {
  SharedMutex::SharedLock lock(nodes_mutex_); // OK if only reading total_playouts_
  return total_playouts_;
}

void Search::ResetBestMove() {
  SharedMutex::Lock nodes_lock(nodes_mutex_); // EnsureBestMoveKnown modifies
  Mutex::Lock lock(counters_mutex_);
  bool old_sent = bestmove_is_sent_;
  bestmove_is_sent_ = false;
  EnsureBestMoveKnown();
  bestmove_is_sent_ = old_sent;
}

// Computes the best move, maybe with temperature (according to the settings).
void Search::EnsureBestMoveKnown() REQUIRES(nodes_mutex_) REQUIRES(counters_mutex_) {
  if (bestmove_is_sent_) return;
  if (root_node_->GetN() == 0 && total_playouts_ == 0) return; // Also check total_playouts_
  if (!root_node_->HasChildren()) return;

  float temperature = params_.GetTemperature();
  const int cutoff_move = params_.GetTemperatureCutoffMove();
  const int decay_delay_moves = params_.GetTempDecayDelayMoves();
  const int decay_moves = params_.GetTempDecayMoves();
  const int moves_played_count = played_history_.Last().GetGamePly() / 2; // Renamed from moves

  if (cutoff_move && (moves_played_count + 1) >= cutoff_move) {
    temperature = params_.GetTemperatureEndgame();
  } else if (temperature != 0.0f && decay_moves > 0) { // Check temperature != 0
    if (moves_played_count >= decay_delay_moves + decay_moves) {
      temperature = 0.0;
    } else if (moves_played_count >= decay_delay_moves) {
      temperature *=
          static_cast<float>(decay_delay_moves + decay_moves - moves_played_count) /
          decay_moves;
    }
    // don't allow temperature to decay below endgame temperature
    if (temperature < params_.GetTemperatureEndgame()) {
      temperature = params_.GetTemperatureEndgame();
    }
  }

  EdgeAndNode bestmove_edge_and_node = temperature != 0.0f // Check temperature != 0
                           ? GetBestRootChildWithTemperature(temperature)
                           : GetBestChildNoTemperature(root_node_, 0);
  
  if (!bestmove_edge_and_node.edge()) { // Handle case where no valid edge is found
      final_bestmove_ = Move::invalid(); // Or some other indicator of no move
      final_pondermove_ = Move::invalid();
      return;
  }

  final_bestmove_ = bestmove_edge_and_node.edge()->GetMove(played_history_.IsBlackToMove());

  // Ponder move selection
  if (bestmove_edge_and_node.GetN() > 0 && bestmove_edge_and_node.node() && bestmove_edge_and_node.node()->HasChildren()) {
    EdgeAndNode ponder_edge_and_node = GetBestChildNoTemperature(bestmove_edge_and_node.node(), 1);
    if (ponder_edge_and_node.edge()) {
        final_pondermove_ = ponder_edge_and_node.edge()->GetMove(!played_history_.IsBlackToMove());
    } else {
        final_pondermove_ = Move::invalid();
    }
  } else {
      final_pondermove_ = Move::invalid();
  }
}


// Returns @count children with most visits.
// For Thompson Sampling, "best" usually means most visited for final move selection,
// or highest BetaMean for evaluation. This function should sort by visits.
std::vector<EdgeAndNode> Search::GetBestChildrenNoTemperature(Node* parent,
                                                              int count,
                                                              int depth) const REQUIRES_SHARED(nodes_mutex_) {
  if (parent->GetN() == 0 && total_playouts_ == 0 && parent == root_node_) return {}; // Check total_playouts for root
  if (!parent->HasChildren()) return {}; // Added check for HasChildren

  const bool is_odd_depth = (depth % 2) == 1;
  const float current_node_draw_score = GetDrawScore(is_odd_depth); // Draw score from current parent's perspective

  std::vector<EdgeAndNode> edges_and_nodes_list; // Renamed from edges
  for (auto& edge_and_node_item : parent->Edges()) { // Iterate EdgeAndNode
    if (!edge_and_node_item.edge()) continue; // Skip if no edge data
    if (parent == root_node_ && !root_move_filter_.empty() &&
        std::find(root_move_filter_.begin(), root_move_filter_.end(),
                  edge_and_node_item.edge()->GetMove()) == root_move_filter_.end()) {
      continue;
    }
    edges_and_nodes_list.push_back(edge_and_node_item);
  }

  const auto middle = (static_cast<int>(edges_and_nodes_list.size()) > count)
                          ? edges_and_nodes_list.begin() + count
                          : edges_and_nodes_list.end();
  
  std::partial_sort(
      edges_and_nodes_list.begin(), middle, edges_and_nodes_list.end(),
      [current_node_draw_score](const auto& a, const auto& b) { // draw_score renamed
        // The function returns "true" when a is preferred to b.
        if (!a.edge()) return false; // b is preferred if a has no edge
        if (!b.edge()) return true;  // a is preferred if b has no edge

        // Terminal evaluation is from child's perspective.
        // EdgeAndNode::GetWL is from child's perspective.
        // EdgeAndNode::GetM is plies to mate from child.

        enum EdgeRank {
          kTerminalLoss,    // Child loses (Parent wins)
          kTablebaseLoss,   // Child loses via TB
          kNonTerminal,     // Non terminal or terminal draw.
          kTablebaseWin,    // Child wins via TB
          kTerminalWin,     // Child wins (Parent loses)
        };

        auto GetEdgeRank = [](const EdgeAndNode& ean) { // ean for edge_and_node
          if (ean.GetN() == 0 || !ean.IsTerminal()) { // ean.GetN(), ean.IsTerminal()
            return kNonTerminal;
          }
          const float child_wl = ean.GetWL(0.0f); // WL from child's perspective
          if (child_wl == 0.0f) return kNonTerminal; // Draw

          if (ean.IsTbTerminal()) { // ean.IsTbTerminal()
            return child_wl < 0.0 ? kTablebaseWin : kTablebaseLoss; // Child loses => Parent wins (TablebaseLoss for child)
                                                                    // Child wins => Parent loses (TablebaseWin for child)
          }
          return child_wl < 0.0 ? kTerminalWin : kTerminalLoss; // Child loses => Parent wins (TerminalLoss for child)
        };

        const auto a_rank = GetEdgeRank(a);
        const auto b_rank = GetEdgeRank(b);
        if (a_rank != b_rank) return a_rank > b_rank; // Higher rank is better for parent

        // If both are terminal draws (kNonTerminal but IsTerminal() is true and WL is 0)
        if (a_rank == kNonTerminal && a.GetN() != 0 && b.GetN() != 0 &&
            a.IsTerminal() && b.IsTerminal()) {
          if (a.IsTbTerminal() != b.IsTbTerminal()) {
            return a.IsTbTerminal() < b.IsTbTerminal(); // Prefer non-TB draws
          }
          return a.GetM(0.0f) < b.GetM(0.0f); // Prefer shorter draws (M is plies from child)
        }

        // Standard rule: visits, then BetaMean (from parent's perspective), then policy.
        if (a_rank == kNonTerminal) {
          if (a.edge()->GetBetaStats().GetVisits() != b.edge()->GetBetaStats().GetVisits())
            return a.edge()->GetBetaStats().GetVisits() > b.edge()->GetBetaStats().GetVisits();
          
          // BetaMeanValue is from child's perspective. We need -BetaMeanValue for parent's perspective.
          float a_beta_mean_parent = -a.edge()->GetBetaStats().GetBetaMeanValue();
          float b_beta_mean_parent = -b.edge()->GetBetaStats().GetBetaMeanValue();

          if (std::abs(a_beta_mean_parent - b_beta_mean_parent) > 1e-6) { // Compare floats carefully
            return a_beta_mean_parent > b_beta_mean_parent;
          }
          return a.edge()->GetP() > b.edge()->GetP();
        }

        // Both are winning for parent (child is losing)
        if (a_rank > kNonTerminal) { // e.g. kTablebaseWin, kTerminalWin (meaning child loses)
          return a.GetM(0.0f) < b.GetM(0.0f); // Prefer shorter wins for parent (child has fewer moves to lose)
        }

        // Both are losing for parent (child is winning)
        // Ranks are kTerminalLoss, kTablebaseLoss (meaning child wins)
        return a.GetM(0.0f) > b.GetM(0.0f); // Prefer longer losses for parent (child has more moves to win)
      });

  if (count < static_cast<int>(edges_and_nodes_list.size())) {
    edges_and_nodes_list.resize(count);
  }
  return edges_and_nodes_list;
}


// Returns a child with most visits.
EdgeAndNode Search::GetBestChildNoTemperature(Node* parent, int depth) const REQUIRES_SHARED(nodes_mutex_) {
  auto res = GetBestChildrenNoTemperature(parent, 1, depth);
  return res.empty() ? EdgeAndNode() : res.front();
}

// Returns a child of a root chosen according to weighted-by-temperature visit
// count. This is used for exploration during play if temperature > 0.
EdgeAndNode Search::GetBestRootChildWithTemperature(float temperature) const REQUIRES_SHARED(nodes_mutex_) {
  const float draw_score_root = GetDrawScore(/* is_odd_depth= */ false);

  std::vector<std::pair<const Edge*, float>> edge_weights; // Store edge and its weight for sampling
  float sum_pow_visits = 0.0;
  
  // Find max visits among eligible moves for normalization (optional, but can help stabilize)
  uint32_t max_visits_for_temp = 0;
  for (auto& edge_and_node : root_node_->Edges()) {
      if (!edge_and_node.edge()) continue;
      if (!root_move_filter_.empty() &&
          std::find(root_move_filter_.begin(), root_move_filter_.end(),
                    edge_and_node.edge()->GetMove()) == root_move_filter_.end()) {
          continue;
      }
      max_visits_for_temp = std::max(max_visits_for_temp, edge_and_node.edge()->GetBetaStats().GetVisits());
  }


  for (auto& edge_and_node : root_node_->Edges()) {
    if (!edge_and_node.edge()) continue;
    if (!root_move_filter_.empty() &&
        std::find(root_move_filter_.begin(), root_move_filter_.end(),
                  edge_and_node.edge()->GetMove()) == root_move_filter_.end()) {
      continue;
    }

    // Temperature applies to visit counts.
    // N_i ^ (1/temp)
    // Using raw visits from BetaStats.
    uint32_t visits = edge_and_node.edge()->GetBetaStats().GetVisits();
    if (visits == 0 && params_.GetTemperatureVisitOffset() == 0) continue; // Skip 0-visit moves if no offset

    float weight = std::pow(static_cast<float>(visits) + params_.GetTemperatureVisitOffset(), 1.0f / temperature);
    
    // Optional: Filter by eval if TemperatureWinpctCutoff is used.
    // This requires getting Q value (e.g. BetaMean from child's perspective)
    if (params_.GetTemperatureWinpctCutoff() > 0 && root_node_->GetN() > 0) { // Check root_node has visits
        // This part is complex as it compares to max_eval.
        // For simplicity, skipping eval-based filtering for temp selection here.
        // Original code has complex logic with max_weight and max_eval.
    }

    edge_weights.push_back({edge_and_node.edge(), weight});
    sum_pow_visits += weight;
  }

  if (edge_weights.empty() || sum_pow_visits == 0.0f) {
      // Fallback to best child by visits if no suitable candidates or sum is zero
      return GetBestChildNoTemperature(root_node_, 0);
  }

  // Select proportionally to weights
  float toss = Random::Get().GetFloat(sum_pow_visits);
  float current_sum = 0.0f;
  for (const auto& pair : edge_weights) {
    current_sum += pair.second;
    if (toss <= current_sum) {
      // Find the EdgeAndNode corresponding to pair.first (Edge*)
      for (auto& ean_lookup : root_node_->Edges()) {
          if (ean_lookup.edge() == pair.first) return ean_lookup;
      }
      break; // Should have found it
    }
  }
  // Fallback if something went wrong (e.g. float precision issues)
  return GetBestChildNoTemperature(root_node_, 0);
}


void Search::StartThreads(size_t how_many) {
  thread_count_.store(how_many, std::memory_order_release);
  Mutex::Lock lock(threads_mutex_);
  // First thread is a watchdog thread.
  if (threads_.empty()) { // Check threads_.size() == 0
    threads_.emplace_back([this]() { WatchdogThread(); });
  }
  // Start working threads.
  for (size_t i = 0; i < how_many; i++) {
    threads_.emplace_back([this, i]() {
      SearchWorker worker(this, params_, i);
      worker.RunBlocking();
    });
  }
  LOGFILE << "Search started. "
          << std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::steady_clock::now() - start_time_)
                 .count()
          << "ms already passed.";
}

void Search::RunBlocking(size_t threads_to_run) { // Renamed param
  StartThreads(threads_to_run);
  Wait();
}

bool Search::IsSearchActive() const {
  return !stop_.load(std::memory_order_acquire);
}

// Static method to populate UCI parameters.
// Added from leela_search_cc (1).txt / search (6).h
void Search::PopulateUciParams(OptionsParser* options) {
  // Beta-Bernoulli Thompson Sampling specific options
  // FPU value is often already present. If not, add it like this:
  // options->Add<FloatOption>("FpuValue", kDefaultFpuValueTS, 0.0f, 1.0f)
  //     << "First Play Urgency value for Thompson Sampling";
  // Assuming FPU structure in SearchParams is more complex (e.g. FpuRoot, FpuLoss, etc.)
  // these will be handled by SearchParams internal UCI option registration.

  options->Add<FloatOption>("PolicyTemperature", 1.0f, 0.01f, 100.0f) // Default 1.0f
      << "Temperature for policy prior integration into Beta distribution (higher = stronger prior)";

  options->Add<BoolOption>("UsePolicyPriors", true) // Default true
      << "Whether to use neural network policy to initialize Beta priors for Thompson Sampling";

  // Keep other existing UCI options from Leela's Search::PopulateUciParams if this method
  // is intended to replace an existing one. If it's additive, this is fine.
  // Example: Threads, MultiPV, etc. are usually here.
   options->Add<IntOption>("Threads", 4, 1, 1024) // Example defaults
      << "Number of search threads";
   options->Add<IntOption>("MultiPV", 1, 1, 500)
       << "Number of principal variations to output";
}


void Search::PopulateCommonIterationStats(IterationStats* stats) {
  stats->time_since_movestart = GetTimeSinceStart();

  SharedMutex::SharedLock nodes_lock(nodes_mutex_); // Keep as shared if only reading
  {
    Mutex::Lock counters_lock(counters_mutex_);
    stats->time_since_first_batch = GetTimeSinceFirstBatch();
    if (!nps_start_time_ && total_playouts_ > 0) {
      nps_start_time_ = std::chrono::steady_clock::now();
    }
  }
  stats->total_visits = total_playouts_ + initial_visits_;
  stats->total_allocated_nodes = dag_->AllocatedNodeCount();
  stats->nodes_since_movestart = total_playouts_;
  stats->batches_since_movestart = total_batches_;
  stats->average_depth = (total_playouts_ > 0) ? (cum_depth_ / total_playouts_) : 0; // Avoid div by zero
  stats->edge_n.clear();
  stats->win_found = false;
  stats->may_resign = true;
  stats->num_losing_edges = 0;
  stats->time_usage_hint_ = IterationStats::TimeUsageHint::kNormal;
  stats->mate_depth = std::numeric_limits<int>::max();

  // If root node hasn't finished first visit, none of this code is safe.
  if (root_node_->GetN() > 0 && root_node_->HasChildren()) { // Check HasChildren
    // Draw score from child's perspective (depth 1 is odd)
    const float child_perspective_draw_score = GetDrawScore(true); 
    // FPU for root's children, from child's perspective
    // float fpu_for_children = GetFpuForThompson(params_, root_node_, true, child_perspective_draw_score, root_node_->GetVisitedPolicy());
    
    float max_q_plus_m = -std::numeric_limits<float>::infinity(); // Init to very small
    uint64_t max_n_visits = 0; // Renamed from max_n
    bool max_n_has_max_q_plus_m = true;
    const auto m_evaluator = network_->GetCapabilities().has_mlh()
                                 ? MEvaluator(params_, root_node_)
                                 : MEvaluator();
    for (const auto& edge_and_node : root_node_->Edges()) {
      if (!edge_and_node.edge()) continue;
      const auto& edge_data = *edge_and_node.edge(); // Use edge_data consistently
      const auto n_visits_edge = edge_data.GetBetaStats().GetVisits(); // Visits from BetaStats
      
      // Q value from parent's (root) perspective for this edge
      // BetaMeanValue is from child's perspective. So -BetaMeanValue is for parent.
      const float q_parent_perspective = -edge_data.GetBetaStats().GetBetaMeanValue();
      
      // const float m_utility = m_evaluator.GetMUtility(edge_and_node, q_parent_perspective); // q is parent's perspective
      // The m_evaluator.GetMUtility expects q from parent's perspective
      const float m_utility = m_evaluator.GetMUtility(edge_and_node, q_parent_perspective);


      const float q_plus_m = q_parent_perspective + m_utility;

      stats->edge_n.push_back(n_visits_edge);
      if (n_visits_edge > 0 && edge_and_node.IsTerminal()) {
          float child_wl = edge_and_node.GetWL(0.0f); // WL from child's perspective
          if (child_wl < 0.0f) { // Child loses -> Parent wins
              stats->win_found = true;
          }
          if (child_wl > 0.0f) { // Child wins -> Parent loses
              stats->num_losing_edges += 1;
          }
          if (child_wl < -0.999f && !edge_and_node.IsTbTerminal()) { // Parent wins by checkmate
            stats->mate_depth =
                std::min(stats->mate_depth,
                         static_cast<int>(std::round(edge_and_node.GetM(0.0f))) / 2 + 1); // M is plies from child
          }
      }


      // Resign logic: if root's Q for any reasonable move is not too bad
      if (n_visits_edge > 0 && q_parent_perspective > -0.98f) {
        stats->may_resign = false;
      }
      if (max_n_visits < n_visits_edge) {
        max_n_visits = n_visits_edge;
        max_n_has_max_q_plus_m = (max_q_plus_m <= q_plus_m); // If new max_n, check if its q_plus_m is also max
      } else if (max_n_visits == n_visits_edge) { // If visits are same, ensure current max_q_plus_m is indeed the max
         if (q_plus_m > max_q_plus_m) max_n_has_max_q_plus_m = false; // A same-visit move has better Q
      }

      if (max_q_plus_m <= q_plus_m) { // Standard update for max_q_plus_m
        if (max_q_plus_m < q_plus_m || n_visits_edge > max_n_visits) { // Strictly better or better due to visits
             max_n_has_max_q_plus_m = (max_n_visits == n_visits_edge);
        }
        max_q_plus_m = q_plus_m;
      }
    }
    if (!max_n_has_max_q_plus_m && max_n_visits > 0) { // Ensure max_n_visits is positive before hint
      stats->time_usage_hint_ = IterationStats::TimeUsageHint::kNeedMoreTime;
    }
  }
}


void Search::WatchdogThread() {
  LOGFILE << "Start a watchdog thread.";
  StoppersHints hints;
  IterationStats iter_stats; // Renamed from stats
  while (true) {
    PopulateCommonIterationStats(&iter_stats);
    MaybeTriggerStop(iter_stats, &hints); // Pass iter_stats
    MaybeOutputInfo();

    constexpr auto kMaxWaitTimeMs = 100;
    constexpr auto kMinWaitTimeMs = 1;

    Mutex::Lock lock(counters_mutex_);
    // Only exit when bestmove is responded. It may happen that search threads
    // already all exited, and we need at least one thread that can do that.
    if (bestmove_is_sent_) break;

    auto remaining_time_ms = hints.GetEstimatedRemainingTimeMs(); // Renamed from remaining_time
    if (remaining_time_ms > kMaxWaitTimeMs) remaining_time_ms = kMaxWaitTimeMs;
    if (remaining_time_ms < kMinWaitTimeMs) remaining_time_ms = kMinWaitTimeMs;

    watchdog_cv_.wait_for(
        lock.get_raw(), std::chrono::milliseconds(remaining_time_ms),
        [this]() REQUIRES(counters_mutex_) { return stop_.load(std::memory_order_acquire) || bestmove_is_sent_; }); // Check bestmove_is_sent_ also
  }
  LOGFILE << "End a watchdog thread.";
}

void Search::FireStopInternal() {
  stop_.store(true, std::memory_order_release);
  watchdog_cv_.notify_all();
}

void Search::Stop() {
  Mutex::Lock lock(counters_mutex_);
  ok_to_respond_bestmove_ = true;
  FireStopInternal();
  LOGFILE << "Stopping search due to `stop` uci command.";
}

void Search::Abort() {
  Mutex::Lock lock(counters_mutex_);
  if (!stop_.load(std::memory_order_acquire) ||
      (!bestmove_is_sent_ && !ok_to_respond_bestmove_)) {
    bestmove_is_sent_ = true; // Force bestmove_is_sent_ to allow watchdog to exit
    FireStopInternal();
  }
  LOGFILE << "Aborting search, if it is still active.";
}

void Search::Wait() {
  Mutex::Lock lock(threads_mutex_);
  while (!threads_.empty()) {
    if (threads_.back().joinable()) { // Check if joinable
        threads_.back().join();
    }
    threads_.pop_back();
  }
}

void Search::CancelSharedCollisions() REQUIRES(nodes_mutex_) {
  for (auto& entry : shared_collisions_) {
    auto& path_ref = entry.first; // Use reference
    // Check if path is not empty before trying to access elements.
    if (path_ref.empty()) continue;

    for (auto it = ++(path_ref.crbegin()); it != path_ref.crend(); ++it) {
      Node* parent_node = std::get<0>(*it);
      if (parent_node) { // Check for nullptr
          // This part of original code `CancelScoreUpdate` is not directly applicable to BetaBernoulliStats
          // DecrementNInFlight is the main action for cancelling a visit.
          // The second element of path_ref.crbegin() (i.e. path_ref.back()) is the child.
          // The parent is *(path_ref.crbegin() + 1)
          // The edge is from parent to child. Child needs to call DecrementNInFlight on itself.
          // The value entry.second is the multivisit count.
          // This logic needs to be clear: which node's NInFlight is decremented?
          // Original code: std::get<0>(*it)->CancelScoreUpdate(entry.second);
          // std::get<0>(*it) is the parent in this loop.
          // This implies CancelScoreUpdate was a method on Node.
          // For Beta-Bernoulli, if a visit is cancelled, NInFlight on the *child* of that edge needs reduction.
          // And the BetaStats should not have been updated. This is complex.
          // Simplest for now: assume DecrementNInFlight is the primary mechanism.
          // If path_ref.back() is the leaf node that was a collision:
          Node* leaf_node_of_collision = std::get<0>(path_ref.back());
          if(leaf_node_of_collision) leaf_node_of_collision->DecrementNInFlight(entry.second);
          // And its parent (if any) might also need adjustment if NInFlight was on parent.
          // The original logic for NInFlight needs to be respected.
          // NInFlight is usually incremented on the child node when selected.
      }
    }
  }
  shared_collisions_.clear();
}

Search::~Search() {
  Abort();
  Wait();
  {
    SharedMutex::Lock lock(nodes_mutex_);
    CancelSharedCollisions();

#ifndef NDEBUG
    // assert(root_node_->ZeroNInFlight()); // ZeroNInFlight might not exist, check root_node_->GetNInFlight() == 0
#endif
  }

  // Free previously released nodes that were not reused during this search.
  dag_->TTMaintenance();
  dag_->TTMaintenance();

  LOGFILE << "Search destroyed.";
}

//////////////////////////////////////////////////////////////////////////////
// SearchWorker
//////////////////////////////////////////////////////////////////////////////

void SearchWorker::RunTasks(int tid) {
  while (true) {
    PickTask* task_ptr = nullptr; // Renamed from task
    int task_id = 0; // Renamed from id
    {
      int spins = 0;
      while (true) {
        int nta = tasks_taken_.load(std::memory_order_acquire);
        int tc = task_count_.load(std::memory_order_acquire);
        if (nta < tc && tc != -1) { // Added tc != -1 check
          int current_val = 0; // Renamed from val
          if (task_taking_started_.compare_exchange_weak(
                  current_val, 1, std::memory_order_acq_rel,
                  std::memory_order_relaxed)) {
            nta = tasks_taken_.load(std::memory_order_acquire); // Re-read after getting lock
            tc = task_count_.load(std::memory_order_acquire);  // Re-read after getting lock
            if (nta < tc && tc != -1) { // Double check with fresh values
              task_id = tasks_taken_.fetch_add(1, std::memory_order_acq_rel);
              if (task_id < tc) { // Ensure task_id is within bounds if tc changed
                 task_ptr = &picking_tasks_[task_id];
              } else { // Roll back if task_id is out of bounds (another thread might have changed tc)
                 tasks_taken_.fetch_sub(1, std::memory_order_relaxed); // Best effort rollback
                 task_id = -1; // Mark as no task taken
              }
              task_taking_started_.store(0, std::memory_order_release);
              if (task_ptr) break; // Break if task successfully acquired
            } else {
              task_taking_started_.store(0, std::memory_order_release); // Release spin lock if no task
            }
          }
          SpinloopPause(); // Moved here, always pause if CAS fails or conditions not met
          spins = 0; // Reset spins as we made a CAS attempt or found work potentially
          continue;
        } else if (tc != -1) { // Still tasks to do, but this thread didn't get one
          spins++;
          if (spins >= 512) {
            std::this_thread::yield();
            spins = 0;
          } else {
            SpinloopPause();
          }
          continue;
        }
        // tc == -1 (no tasks currently set) or (nta >= tc and tc != -1 which means all tasks taken)
        spins = 0;
        Mutex::Lock lock(picking_tasks_mutex_);
        nta = tasks_taken_.load(std::memory_order_acquire); // Re-check under mutex
        tc = task_count_.load(std::memory_order_acquire);
        
        if (exiting_ && (tc == -1 || nta >= tc)) return; // Exit condition
        if (tc != -1 && nta < tc) continue; // Tasks might have been added while acquiring mutex

        task_added_.wait(lock.get_raw());
        // Re-check after wake-up
        nta = tasks_taken_.load(std::memory_order_acquire);
        tc = task_count_.load(std::memory_order_acquire);
        if (exiting_ && (tc == -1 || nta >= tc)) return;
      }
    }
    if (task_ptr != nullptr) { // Check task_ptr
      switch (task_ptr->task_type) {
        case PickTask::kGathering: {
          PickNodesToExtendTask(task_ptr->start_path, task_ptr->collision_limit,
                                task_ptr->history, &(task_ptr->results),
                                &(task_workspaces_[tid]));
          break;
        }
        case PickTask::kProcessing: {
          ProcessPickedTask(task_ptr->start_idx, task_ptr->end_idx);
          break;
        }
      }
      // Ensure task_id is valid before accessing picking_tasks_
      if (task_id >=0 && static_cast<size_t>(task_id) < picking_tasks_.size()) {
          picking_tasks_[task_id].complete = true;
      }
      completed_tasks_.fetch_add(1, std::memory_order_acq_rel);
    }
  }
}


void SearchWorker::ExecuteOneIteration() {
  // 1. Initialize internal structures.
  InitializeIteration(search_->network_->NewComputation());

  if (params_.GetMaxConcurrentSearchers() != 0) {
    std::unique_ptr<SpinHelper> spin_helper_ptr; // Renamed from spin_helper
    if (params_.GetSearchSpinBackoff()) {
      spin_helper_ptr = std::make_unique<ExponentialBackoffSpinHelper>();
    } else {
      spin_helper_ptr = std::make_unique<SpinHelper>();
    }

    while (true) {
      if (search_->stop_.load(std::memory_order_acquire) &&
          (search_->GetTotalPlayouts() + search_->initial_visits_ > 0 || root_node_->GetN() > 0) ) { // Check root_node N too
        return;
      }

      int available_searchers = search_->pending_searchers_.load(std::memory_order_acquire); // Renamed
      if (available_searchers == 0) {
        spin_helper_ptr->Wait();
        continue;
      }

      if (search_->pending_searchers_.compare_exchange_weak(
              available_searchers, available_searchers - 1, std::memory_order_acq_rel, std::memory_order_relaxed)) { // Added relaxed failure order
        break;
      } else {
        spin_helper_ptr->Backoff();
      }
    }
  }

  // 2. Gather minibatch.
  GatherMinibatch();
  task_count_.store(-1, std::memory_order_release); // Mark tasks as finished for task runners
  search_->backend_waiting_counter_.fetch_add(1, std::memory_order_relaxed);

  // 2b. Collect collisions.
  CollectCollisions();

  if (params_.GetMaxConcurrentSearchers() != 0) {
    search_->pending_searchers_.fetch_add(1, std::memory_order_acq_rel);
  }

  // 4. Run NN computation.
  RunNNComputation();
  search_->backend_waiting_counter_.fetch_add(-1, std::memory_order_relaxed);

  // 5. Retrieve NN computations (and terminal values) into nodes.
  FetchMinibatchResults();

  // 6. Propagate the new nodes' information to all their parents in the tree.
  DoBackupUpdate();

  // 7. Update the Search's status and progress information.
  UpdateCounters();

  // If required, waste time to limit nps.
  if (params_.GetNpsLimit() > 0) {
    while (search_->IsSearchActive()) {
      int64_t time_since_first_batch_ms = 0;
      {
        Mutex::Lock lock(search_->counters_mutex_);
        time_since_first_batch_ms = search_->GetTimeSinceFirstBatch();
      }
      if (time_since_first_batch_ms <= 0) {
        time_since_first_batch_ms = search_->GetTimeSinceStart();
      }
      if (time_since_first_batch_ms <=0) break; // Avoid division by zero if time is still zero

      auto current_nps = (search_->GetTotalPlayouts() + search_->initial_visits_) * 1000.0f / time_since_first_batch_ms; // Use float for division
      if (current_nps > params_.GetNpsLimit()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      } else {
        break;
      }
    }
  }
}

// 1. Initialize internal structures.
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void SearchWorker::InitializeIteration(
    std::unique_ptr<NetworkComputation> net_computation) { // Renamed from computation
  computation_ = std::make_unique<CachingComputation>(
      std::move(net_computation), search_->network_->GetCapabilities().input_format,
      params_.GetHistoryFill(), search_->cache_);
  computation_->Reserve(params_.GetMiniBatchSize());
  minibatch_.clear();
  minibatch_.reserve(2 * params_.GetMiniBatchSize()); // Reserve double for potential collisions etc.
}

// 2. Gather minibatch.
// ~~~~~~~~~~~~~~~~~~~~
namespace { // Anonymous namespace from search (13).cc
int MixInt(int high, int low, float ratio) { // Renamed from Mix
  return static_cast<int>(std::round(static_cast<float>(low) +
                                     static_cast<float>(high - low) * ratio));
}

int CalculateCollisionsLeftForGather(int64_t current_node_visits, const SearchParams& params) { // Renamed
  // End checked first
  if (current_node_visits >= params.GetMaxCollisionVisitsScalingEnd()) {
    return params.GetMaxCollisionVisits();
  }
  if (current_node_visits <= params.GetMaxCollisionVisitsScalingStart()) {
    return 1;
  }
  // Ensure denominator is not zero.
  float denominator = static_cast<float>(params.GetMaxCollisionVisitsScalingEnd() -
                                         params.GetMaxCollisionVisitsScalingStart());
  if (denominator == 0.0f) return 1; // Avoid division by zero, return min collisions

  return MixInt(params.GetMaxCollisionVisits(), 1,
             std::pow((static_cast<float>(current_node_visits) -
                       params.GetMaxCollisionVisitsScalingStart()) /
                          denominator,
                      params.GetMaxCollisionVisitsScalingPower()));
}
}  // namespace

void SearchWorker::GatherMinibatch() {
  uint32_t current_minibatch_size = 0; // Renamed from minibatch_size
  int current_root_visits = 0; // Renamed from cur_n
  {
    SharedMutex::SharedLock lock(search_->nodes_mutex_); // Use shared lock if only reading GetN
    current_root_visits = search_->root_node_->GetN();
  }
  int64_t remaining_playouts_estimate = latest_time_manager_hints_.GetEstimatedRemainingPlayouts(); // Renamed
  uint32_t collisions_available = CalculateCollisionsLeftForGather( // Renamed
      std::min(static_cast<int64_t>(current_root_visits), remaining_playouts_estimate), params_);

  number_out_of_order_ = 0;
  int current_thread_count = search_->thread_count_.load(std::memory_order_acquire); // Renamed

  while (current_minibatch_size < params_.GetMiniBatchSize() &&
         number_out_of_order_ < params_.GetMaxOutOfOrderEvals()) {
    if (current_minibatch_size > 0 && computation_->GetCacheMisses() == 0 && !minibatch_.empty()) return; // Added !minibatch_.empty()

    if (current_thread_count > 1 && current_minibatch_size > 0 && !minibatch_.empty() && // Added !minibatch_.empty()
        computation_->GetCacheMisses() > params_.GetIdlingMinimumWork() &&
        current_thread_count -
                search_->backend_waiting_counter_.load(
                    std::memory_order_relaxed) >
            params_.GetThreadIdlingThreshold()) {
      return;
    }

    int new_pickup_start_idx = static_cast<int>(minibatch_.size()); // Renamed

    PickNodesToExtend(
        std::min({collisions_available, params_.GetMiniBatchSize() - current_minibatch_size,
                  params_.GetMaxOutOfOrderEvals() - number_out_of_order_}));

    int non_collision_count = 0; // Renamed
    for (int i = new_pickup_start_idx; i < static_cast<int>(minibatch_.size()); i++) {
      auto& picked_node_ref = minibatch_[i]; // Renamed
      if (picked_node_ref.IsCollision()) {
        continue;
      }
      ++non_collision_count;
      ++current_minibatch_size;
    }

    {
      SharedMutex::Lock lock(search_->nodes_mutex_); // Full lock for ProcessPickedTask modifications

      bool needs_task_wait = false; // Renamed
      int process_task_start_idx = new_pickup_start_idx; // Renamed
      if (params_.GetTaskWorkersPerSearchWorker() > 0 &&
          non_collision_count >= params_.GetMinimumWorkSizeForProcessing()) {
        const int num_processing_tasks = std::clamp( // Renamed
            non_collision_count / params_.GetMinimumWorkPerTaskForProcessing(), 2,
            params_.GetTaskWorkersPerSearchWorker() + 1);
        int items_per_worker = non_collision_count / num_processing_tasks; // Renamed
        needs_task_wait = true;
        ResetTasks(); // Ensure this is safe to call here
        int items_found_for_task = 0; // Renamed
        for (int i = new_pickup_start_idx; i < static_cast<int>(minibatch_.size()); i++) {
          auto& picked_node_ref = minibatch_[i]; // Renamed
          if (picked_node_ref.IsCollision()) {
            continue;
          }
          ++items_found_for_task;
          if (items_found_for_task == items_per_worker && picking_tasks_.size() < static_cast<size_t>(num_processing_tasks - 1)) { // Check picking_tasks_ size
            picking_tasks_.emplace_back(process_task_start_idx, i + 1);
            task_count_.fetch_add(1, std::memory_order_acq_rel);
            process_task_start_idx = i + 1;
            items_found_for_task = 0;
            // if (picking_tasks_.size() == static_cast<size_t>(num_processing_tasks - 1)) {
            //   break; // This break was inside the loop, should be outside or part of condition
            // }
          }
        }
        // Add the last task if any items remain for the main worker or a final task
        if (process_task_start_idx < static_cast<int>(minibatch_.size())) {
             // If items remain that were not assigned to a task because loop ended or items_found_for_task < items_per_worker
             // These will be processed by the main thread in ProcessPickedTask if no task is created for them.
             // Or, create one last task if items_per_worker > 0
             if (items_per_worker > 0 && (static_cast<int>(minibatch_.size()) - process_task_start_idx > 0) ) { // Ensure there are items for the last task
                 bool non_collision_in_remainder = false;
                 for(int k=process_task_start_idx; k < static_cast<int>(minibatch_.size()); ++k) {
                     if (!minibatch_[k].IsCollision()) { non_collision_in_remainder = true; break;}
                 }
                 if (non_collision_in_remainder) {
                    picking_tasks_.emplace_back(process_task_start_idx, static_cast<int>(minibatch_.size()));
                    task_count_.fetch_add(1, std::memory_order_acq_rel);
                    process_task_start_idx = static_cast<int>(minibatch_.size()); // Mark all as tasked
                 }
             }
        }


      }
      ProcessPickedTask(process_task_start_idx, static_cast<int>(minibatch_.size())); // Process remaining items
      if (needs_task_wait) {
        WaitForTasks();
      }
    }
    bool any_ooo_completed = false; // Renamed
    for (int i = static_cast<int>(minibatch_.size()) - 1; i >= new_pickup_start_idx; i--) {
      if (minibatch_[i].ooo_completed) {
        any_ooo_completed = true;
        break;
      }
    }
    if (any_ooo_completed) {
      SharedMutex::Lock lock(search_->nodes_mutex_); // Full lock for modifications
      for (int i = static_cast<int>(minibatch_.size()) - 1; i >= new_pickup_start_idx; i--) {
        if (minibatch_[i].IsCollision()) {
          // Simplified: just decrement NInFlight on the leaf of the collision path
          Node* collision_leaf_node = std::get<0>(minibatch_[i].path.back());
          if (collision_leaf_node) collision_leaf_node->DecrementNInFlight(minibatch_[i].multivisit);
          minibatch_.erase(minibatch_.begin() + i);
        } else if (minibatch_[i].ooo_completed) {
          FetchSingleNodeResult(&minibatch_[i], minibatch_[i], 0); // minibatch_[i] acts as computation for cache hit
          DoBackupUpdateSingleNode(minibatch_[i]);
          minibatch_.erase(minibatch_.begin() + i);
          --current_minibatch_size;
          ++number_out_of_order_;
        }
      }
    }
    // Add to NN computation batch
    for (size_t i = new_pickup_start_idx; i < minibatch_.size(); i++) { // Use size_t for loop
      if (!minibatch_[i].ShouldAddToInput()) continue;
      if (minibatch_[i].is_cache_hit) {
        computation_->AddInputByHash(minibatch_[i].hash,
                                     std::move(minibatch_[i].lock));
      } else {
        computation_->AddInput(minibatch_[i].hash, minibatch_[i].history);
      }
    }

    for (size_t i = new_pickup_start_idx; i < minibatch_.size(); i++) { // Use size_t
      auto& picked_node_ref = minibatch_[i]; // Renamed
      if (picked_node_ref.IsCollision()) {
        if (picked_node_ref.maxvisit > 0 &&
            collisions_available > picked_node_ref.multivisit) {
          SharedMutex::Lock lock(search_->nodes_mutex_); // Full lock
          int extra_visits = std::min(picked_node_ref.maxvisit, collisions_available) -
                           picked_node_ref.multivisit;
          if (extra_visits > 0) { // Only if positive
              picked_node_ref.multivisit += extra_visits;
              // Increment NInFlight on the leaf node of the collision path
              Node* collision_leaf_node = std::get<0>(picked_node_ref.path.back());
              if (collision_leaf_node) collision_leaf_node->IncrementNInFlight(extra_visits);
          }
        }
        if ((collisions_available -= picked_node_ref.multivisit) <= 0) return; // collisions_available can be 0
        if (search_->stop_.load(std::memory_order_acquire)) return;
      }
    }
  }
}


void SearchWorker::ProcessPickedTask(int start_idx, int end_idx) {
  for (int i = start_idx; i < end_idx; i++) {
    if (i >= static_cast<int>(minibatch_.size())) break; // Boundary check
    auto& picked_node_ref = minibatch_[i]; // Renamed
    if (picked_node_ref.IsCollision()) continue;
    if (picked_node_ref.IsExtendable()) {
      SharedMutex::Lock lock(search_->nodes_mutex_); // ExtendNode modifies shared tree structure
      ExtendNode(picked_node_ref);
    }
    picked_node_ref.ooo_completed =
        params_.GetOutOfOrderEval() && picked_node_ref.CanEvalOutOfOrder();
  }
}

#define MAX_PICKING_TASKS 100 // Renamed from MAX_TASKS

void SearchWorker::ResetTasks() {
  task_count_.store(0, std::memory_order_release);
  tasks_taken_.store(0, std::memory_order_release);
  completed_tasks_.store(0, std::memory_order_release);
  // picking_tasks_ should be locked if accessed by other threads, but here it's this thread's setup phase
  Mutex::Lock lock(picking_tasks_mutex_); // Lock for modifying picking_tasks_
  picking_tasks_.clear();
  picking_tasks_.reserve(MAX_PICKING_TASKS);
}


int SearchWorker::WaitForTasks() {
  while (true) {
    int tasks_completed_count = completed_tasks_.load(std::memory_order_acquire); // Renamed
    int tasks_to_do_count = task_count_.load(std::memory_order_acquire); // Renamed
    if (tasks_to_do_count == -1 || tasks_completed_count >= tasks_to_do_count) return tasks_completed_count; // Check for tc == -1
    SpinloopPause();
  }
}

void SearchWorker::PickNodesToExtend(int collision_limit_val) { // Renamed param
  ResetTasks();
  {
    Mutex::Lock lock(picking_tasks_mutex_); // Lock for notify
    task_added_.notify_all();
  }
  // std::vector<Move> empty_movelist; // Not used in this function scope
  SharedMutex::Lock lock(search_->nodes_mutex_); // This lock is critical and held for a long time.
                                                 // PickNodesToExtendTask itself does not acquire/release this lock.
  history_.Trim(search_->played_history_.GetLength());
  PickNodesToExtendTask({std::make_tuple(search_->root_node_, 0, 0)}, // Initial path
                        collision_limit_val, history_, &minibatch_, // Pass worker's minibatch_
                        &main_workspace_);

  int tasks_completed_count = WaitForTasks(); // Renamed
  // Collect results from tasks
  Mutex::Lock task_lock(picking_tasks_mutex_); // Lock for accessing picking_tasks_
  for (int i = 0; i < tasks_completed_count; i++) { // Iterate up to completed tasks
    if (static_cast<size_t>(i) < picking_tasks_.size() && picking_tasks_[i].task_type == PickTask::kGathering) { // Check type
        for (size_t j = 0; j < picking_tasks_[i].results.size(); j++) { // Use size_t
          minibatch_.emplace_back(std::move(picking_tasks_[i].results[j]));
        }
    }
  }
}


// Depth starts with 0 at root, so number of plies in PV equals depth.
std::pair<int, int> SearchWorker::GetRepetitions(int depth,
                                                 const Position& position) {
  const auto current_repetitions = position.GetRepetitions(); // Renamed

  if (current_repetitions == 0) return {0, 0};
  if (current_repetitions >= 2) return {current_repetitions, 0}; // Threefold or more

  // Twofold repetition check (current_repetitions == 1)
  const auto plies_since_rep = position.GetPliesSincePrevRepetition(); // Renamed
  // Original Lc0 condition: depth >= 4 && depth >= plies_since_rep
  // This means the repeated position is "close" in the search path.
  if (params_.GetTwoFoldDraws() && depth >= 4 && depth >= plies_since_rep) {
    return {1, plies_since_rep};
  }

  return {0, 0}; // Not a draw by repetition according to these rules
}

// Check if PickNodesToExtendTask should stop picking at this @node.
bool SearchWorker::ShouldStopPickingHere(Node* node_to_check, bool is_root_node, // Renamed
                                         int repetition_count) REQUIRES_SHARED(search_->nodes_mutex_) { // Renamed
  constexpr double kWLDifferenceLimit = 0.01; // Renamed
  constexpr float kDrawProbDifferenceLimit = 0.01f; // Renamed
  constexpr float kMovesLeftDifferenceLimit = 2.0f; // Renamed

  if (!node_to_check || node_to_check->GetN() == 0 || node_to_check->IsTerminal()) return true; // Added !node_to_check

  assert(!is_root_node || node_to_check == search_->root_node_);
  if (is_root_node) return false;

  if (repetition_count >= 2) return true; // Draw by 3-fold repetition

  auto low_node_ptr = node_to_check->GetLowNode(); // Renamed
  if (!low_node_ptr) return false; // No LowNode, so cannot compare; extend normally

  // Only known transpositions can differ. This check might be too restrictive.
  // If LowNode has different terminal status or eval due to deeper search, we might stop.
  // if (!low_node_ptr->IsTransposition()) return false; // Original check, reconsider if needed

  if (low_node_ptr->IsTerminal() && !node_to_check->IsTerminal()) return true; // LowNode terminal, Node not

  // Bounds comparison needs Node::GetBounds()
  // auto [low_node_lower, low_node_upper] = low_node_ptr->GetBounds();
  // auto [node_lower, node_upper] = node_to_check->GetBounds();
  // if (low_node_lower != -node_upper || low_node_upper != -node_lower) return true;

  // WL, D, M comparisons (ensure these methods exist on LowNode and Node)
  // float wl_diff = std::abs(low_node_ptr->GetWL() + node_to_check->GetWL());
  // if (wl_diff >= kWLDifferenceLimit) return true;
  // float d_diff = std::abs(low_node_ptr->GetD() - node_to_check->GetD());
  // if (d_diff >= kDrawProbDifferenceLimit) return true;
  // float m_diff = std::abs(low_node_ptr->GetM() + 1 - node_to_check->GetM());
  // if (m_diff >= kMovesLeftDifferenceLimit) return true;

  return false; // Default to not stopping here if no conditions met
}


void SearchWorker::PickNodesToExtendTask(
    const BackupPath& initial_path, int collision_limit_val, PositionHistory& current_history, // Renamed params
    std::vector<NodeToProcess>* result_receiver, // Renamed
    TaskWorkspace* workspace_ptr) NO_THREAD_SAFETY_ANALYSIS { // Renamed
  
  assert(!initial_path.empty());

  auto& vtp_buffer_ref = workspace_ptr->vtp_buffer; // Renamed
  auto& visits_to_perform_list = workspace_ptr->visits_to_perform; // Renamed
  visits_to_perform_list.clear();
  auto& vtp_last_filled_indices = workspace_ptr->vtp_last_filled; // Renamed
  vtp_last_filled_indices.clear();
  auto& current_selection_path_indices = workspace_ptr->current_path; // Renamed
  current_selection_path_indices.clear();
  auto& full_backup_path = workspace_ptr->full_path; // Renamed
  full_backup_path = initial_path;
  
  Node* current_node_ptr; // Renamed
  int current_repetition_count; // Renamed
  int current_moves_left; // Renamed (not used in original selection much)
  std::tie(current_node_ptr, current_repetition_count, current_moves_left) = full_backup_path.back();

  if (result_receiver->capacity() < 30) {
    result_receiver->reserve(result_receiver->size() + 30);
  }

  std::array<float, 256> current_edge_selection_scores; // Renamed and purpose clarified
  // std::array<bool, 256> visited_edge_flags; // Renamed, might not be needed for TS if FPU handles unvisited

  // For Thompson Sampling, top utils for policy boosting aren't standard.
  // If MLH is combined, store Q + MLH. PUCT's U is replaced by Thompson sample.
  // std::array<float, num_top> top_utils; // Remove if not using policy boosting like PUCT's

  auto& iterators_cache = workspace_ptr->cur_iters; // Renamed

  const int64_t current_best_root_edge_visits = search_->current_best_edge_.GetN(); // Renamed

  int tasks_passed_off_count = 0; // Renamed
  int visits_completed_this_task = 0; // Renamed

  bool is_on_root_node = (current_node_ptr == search_->root_node_); // Renamed
  const float even_depth_draw_score = search_->GetDrawScore(false); // Renamed
  const float odd_depth_draw_score = search_->GetDrawScore(true); // Renamed
  const auto& root_moves_filter_ref = search_->root_move_filter_; // Renamed
  auto m_evaluator_instance = moves_left_support_ ? MEvaluator(params_) : MEvaluator(); // Renamed

  int dynamic_visit_limit = std::numeric_limits<int>::max(); // Renamed

  current_selection_path_indices.push_back(-1); // Start with a sentinel for root/current node processing

  while (!current_selection_path_indices.empty()) {
    assert(full_backup_path.size() >= initial_path.size());
    if (current_selection_path_indices.back() == -1) { // Process current_node_ptr
      int visits_to_allocate = collision_limit_val;
      if (current_selection_path_indices.size() > 1) { // Not the initial node of this task
        visits_to_allocate = (*visits_to_perform_list.back())[current_selection_path_indices[current_selection_path_indices.size() - 2]];
      }

      if (ShouldStopPickingHere(current_node_ptr, is_on_root_node, current_repetition_count)) {
        if (is_on_root_node && current_node_ptr->TryStartScoreUpdate()) { // Special handling for root
             visits_to_allocate -=1; // One visit processed directly
             // Create NodeToProcess for the root itself if it's a terminal/unexpandable
             result_receiver->push_back(NodeToProcess::Visit(full_backup_path, current_history));
             visits_completed_this_task++;
        }
        if (visits_to_allocate > 0) {
          int max_collision_count = 0;
          if (visits_to_allocate == collision_limit_val && initial_path.size() == 1 && dynamic_visit_limit > visits_to_allocate) {
             max_collision_count = dynamic_visit_limit;
          }
          result_receiver->push_back(NodeToProcess::Collision(full_backup_path, visits_to_allocate, max_collision_count));
          visits_completed_this_task += visits_to_allocate;
        }
        // Pop from paths
        current_history.Pop();
        full_backup_path.pop_back();
        if (!full_backup_path.empty()) {
          std::tie(current_node_ptr, current_repetition_count, current_moves_left) = full_backup_path.back();
        } else {
          current_node_ptr = nullptr; // Should not happen if initial_path was not empty
        }
        current_selection_path_indices.pop_back();
        continue;
      }
      if (is_on_root_node) { // Root NInFlight handled by TryStartScoreUpdate or similar
          current_node_ptr->IncrementNInFlight(visits_to_allocate); 
      }


      // Prepare for selecting children from current_node_ptr
      if (!vtp_buffer_ref.empty()) {
        visits_to_perform_list.push_back(std::move(vtp_buffer_ref.back()));
        vtp_buffer_ref.pop_back();
      } else {
        visits_to_perform_list.push_back(std::make_unique<std::array<int, 256>>());
      }
      vtp_last_filled_indices.push_back(-1);
      std::fill(visits_to_perform_list.back()->begin(), visits_to_perform_list.back()->end(), 0);


      // Determine draw score from current_node_ptr's perspective
      const float current_node_draw_score_val = (full_backup_path.size() % 2 != 0) ? even_depth_draw_score : odd_depth_draw_score; // Adjusted logic based on path size for depth parity
      m_evaluator_instance.SetParent(current_node_ptr);
      
      // Get FPU value. Note: Leela's FPU is complex.
      // For TS, FPU is simpler. `SelectChildThompsonSampling` will handle FPU internally.
      // We need parent_q for relative FPU if used by SelectChildThompsonSampling.
      float parent_q_for_fpu_calc = current_node_ptr->GetQ(current_node_draw_score_val);


      // Policy decay (if used with Thompson Sampling for priors/FPU)
      // const float decay_factor = ComputePolicyDecayFactor(params_, current_node_ptr->GetWeight()); // GetWeight might not be on Node


      // Loop to allocate `visits_to_allocate` among children
      int temp_visits_to_allocate = visits_to_allocate; // Use a temp variable
      while (temp_visits_to_allocate > 0) {
        uint16_t selected_child_idx = current_node_ptr->SelectChildThompsonSampling(rng_, params_, parent_q_for_fpu_calc);
        
        // Original PUCT had logic for `estimated_visits_to_change_best` to cap visits per selection.
        // For Thompson Sampling, this is less direct. Each sample is one "virtual playout".
        // We can either do one visit per selection, or try to estimate multiple.
        // Simplest: 1 visit per selection.
        int visits_for_this_child = 1; 
        // Could add logic to assign more visits if a child is overwhelmingly better or to match PUCT's multivisit.
        // For now, stick to 1 visit.

        visits_for_this_child = std::min(visits_for_this_child, temp_visits_to_allocate);
        if (selected_child_idx >= current_node_ptr->GetNumEdges()) { // Safety check
            if (current_node_ptr->GetNumEdges() > 0) selected_child_idx = 0; // Default to first if error
            else break; // No edges
        }


        (*visits_to_perform_list.back())[selected_child_idx] += visits_for_this_child;
        temp_visits_to_allocate -= visits_for_this_child;

        // This part is tricky: PUCT updates `current_weightstarted` and `current_score` to reflect the new visit.
        // For Thompson Sampling, the Beta distribution of the selected child *would* change.
        // Re-sampling immediately might be too slow or complex here.
        // The common approach is to select, add to batch, and update stats after NN eval.
        // So, we don't update the selection scores within this loop for TS.
        // We just mark that this edge will get `visits_for_this_child`.
        
        // Update vtp_last_filled_indices
        if (selected_child_idx > vtp_last_filled_indices.back()) {
             vtp_last_filled_indices.back() = selected_child_idx;
        }
      }
      is_on_root_node = false; // Subsequent nodes are not root

      // Task splitting logic (original PUCT had this here)
      // For Thompson Sampling, splitting tasks is still valid.
      for (int i = 0; i <= vtp_last_filled_indices.back(); ++i) {
        int child_visits_total = (*visits_to_perform_list.back())[i];
        if (child_visits_total == 0) continue;

        if (params_.GetTaskWorkersPerSearchWorker() > 0 &&
            child_visits_total > params_.GetMinimumWorkSizeForPicking() &&
            child_visits_total < ((collision_limit_val - tasks_passed_off_count - visits_completed_this_task) * 2 / 3) && // Check against remaining original limit
            child_visits_total + tasks_passed_off_count + visits_completed_this_task <
                collision_limit_val - params_.GetMinimumRemainingWorkSizeForPicking()) {

          Edge& child_edge = current_node_ptr->GetEdge(i); // Assumes Node::GetEdge(idx)
          Node* child_node_obj = child_edge.GetOrSpawnNode(current_node_ptr, i); // Pass parent and index

          current_history.Append(child_edge.GetMove());
          auto [child_reps, child_ml] = GetRepetitions(full_backup_path.size(), current_history.Last());
          BackupPath child_path = full_backup_path; // Copy
          child_path.push_back({child_node_obj, child_reps, child_ml});

          if (!ShouldStopPickingHere(child_node_obj, false, child_reps)) {
            bool task_passed = false;
            {
              Mutex::Lock task_creation_lock(picking_tasks_mutex_); // Renamed
              if (picking_tasks_.size() < MAX_PICKING_TASKS) {
                picking_tasks_.emplace_back(child_path, current_history, child_visits_total);
                task_count_.fetch_add(1, std::memory_order_acq_rel);
                task_added_.notify_all(); // Notify task threads
                task_passed = true;
                tasks_passed_off_count += child_visits_total;
              }
            }
            if (task_passed) {
              (*visits_to_perform_list.back())[i] = 0; // These visits are now handled by a sub-task
            }
          }
          current_history.Pop(); // Backtrack history after processing child
        }
      }
      // Fall through to select the first child for this worker to continue with.
    } // End of if (current_selection_path_indices.back() == -1)

    // Select next child for this worker to descend into
    int last_processed_child_idx = current_selection_path_indices.back(); // Renamed
    bool found_next_child_for_worker = false; // Renamed
    if (vtp_last_filled_indices.back() > last_processed_child_idx) {
      for (int i = last_processed_child_idx + 1; i <= vtp_last_filled_indices.back(); ++i) {
        if ((*visits_to_perform_list.back())[i] > 0) {
          current_selection_path_indices.back() = i; // Update current level's processed index
          current_selection_path_indices.push_back(-1); // Add sentinel for new child level

          Edge& child_edge_to_descend = current_node_ptr->GetEdge(i); // Assumes Node::GetEdge
          current_node_ptr = child_edge_to_descend.GetOrSpawnNode(current_node_ptr, i); // Update current_node_ptr
          current_history.Append(child_edge_to_descend.GetMove());
          std::tie(current_repetition_count, current_moves_left) =
              GetRepetitions(full_backup_path.size(), current_history.Last());
          full_backup_path.push_back({current_node_ptr, current_repetition_count, current_moves_left});
          
          found_next_child_for_worker = true;
          break;
        }
      }
    }

    if (!found_next_child_for_worker) { // Backtrack
      current_history.Pop();
      full_backup_path.pop_back();
      if (!full_backup_path.empty()) {
        std::tie(current_node_ptr, current_repetition_count, current_moves_left) = full_backup_path.back();
      } else {
        current_node_ptr = nullptr;
      }
      current_selection_path_indices.pop_back();
      
      if (!visits_to_perform_list.empty()) { // Add to buffer only if list is not empty
          vtp_buffer_ref.push_back(std::move(visits_to_perform_list.back()));
          visits_to_perform_list.pop_back();
      }
      if (!vtp_last_filled_indices.empty()) { // Pop only if not empty
          vtp_last_filled_indices.pop_back();
      }
    }
  }
}


void SearchWorker::ExtendNode(NodeToProcess& picked_node_ref) REQUIRES(search_->nodes_mutex_) { // Renamed
  const auto& backup_path_ref = picked_node_ref.path; // Renamed
  Node* node_to_extend = picked_node_ref.node; // Renamed
  // Ensure node_to_extend->GetLowNode() is null for a new extension.
  // If it's not null, it implies it was already processed or is a TT hit,
  // which should be handled by IsExtendable or other checks.
  assert(!node_to_extend->GetLowNode() || node_to_extend->GetLowNode()->GetN() == 0);


  const PositionHistory& node_history = picked_node_ref.history; // Renamed

  const auto& board_state = node_history.Last().GetBoard(); // Renamed
  std::vector<Move> legal_moves_list = board_state.GenerateLegalMoves(); // Renamed

  if (legal_moves_list.empty()) {
    if (board_state.IsUnderCheck()) {
      node_to_extend->MakeTerminal(GameResult::WHITE_WON); // Or BLACK_WON depending on turn
    } else {
      node_to_extend->MakeTerminal(GameResult::DRAW); // Stalemate
    }
    return; // Node is terminal, no further extension or NN query needed for policy
  }

  // Create edges for the node_to_extend
  // Pass current position to CreateEdges if it needs it for move details
  node_to_extend->CreateEdges(legal_moves_list, node_history.Last()); 
  // node_to_extend now has edges, but P values are 0 and BetaStats are default.

  // Standard terminal checks (50-move, insufficient material, etc.)
  if (node_to_extend != search_->root_node_) { // Don't make root terminal by rule here
    if (!board_state.HasMatingMaterial()) {
      node_to_extend->MakeTerminal(GameResult::DRAW); return;
    }
    if (node_history.Last().GetRule50Ply() >= 100) {
      node_to_extend->MakeTerminal(GameResult::DRAW); return;
    }
    if (picked_node_ref.repetitions >= 2) { // 3-fold repetition is a draw
      // This is a pseudo-terminal for search, but NN eval might still be useful.
      // For Thompson Sampling, if it's a draw, backups should reflect that.
      // node_to_extend->MakeTerminal(GameResult::DRAW); // Or handle as pseudo-terminal.
      // Let's assume it's not made hard-terminal here, but eval will be draw.
    }
    // Syzygy TB probe
    else if (search_->syzygy_tb_ && !search_->root_is_in_dtz_ &&
             board_state.castlings().no_legal_castle() &&
             node_history.Last().GetRule50Ply() == 0 && // Only if 50-move counter is 0
             (board_state.ours() | board_state.theirs()).count() <=
                 search_->syzygy_tb_->max_cardinality()) {
      ProbeState probe_state; // Renamed
      const WDLScore wdl_score = search_->syzygy_tb_->probe_wdl(node_history.Last(), &probe_state); // Renamed
      if (probe_state != FAIL) {
        float m_val = 0.0f; // Renamed
        if (backup_path_ref.size() > 1) {
          auto parent_node_ptr = std::get<0>(backup_path_ref[backup_path_ref.size() - 2]); // Renamed
          if (parent_node_ptr) m_val = std::max(0.0f, parent_node_ptr->GetM() - 1.0f); // Assumes Node::GetM()
        }
        GameResult game_res; // Renamed
        if (wdl_score == WDL_WIN) game_res = GameResult::BLACK_WON; // Check perspective
        else if (wdl_score == WDL_LOSS) game_res = GameResult::WHITE_WON;
        else game_res = GameResult::DRAW;
        node_to_extend->MakeTerminal(game_res, m_val /*, Terminal::Tablebase */); // Terminal type if Node supports
        search_->tb_hits_.fetch_add(1, std::memory_order_acq_rel);
        return; // Terminal by TB
      }
    }
  }

  picked_node_ref.nn_queried = true; // Will proceed to NN query or cache/TT lookup

  picked_node_ref.hash = search_->dag_->GetHistoryHash(node_history);
  picked_node_ref.ch_hash = search_->dag_->GetCHHash(node_history); // If CH is used

  auto tt_low_node_ptr = search_->dag_->TTFind(picked_node_ref.hash); // Renamed
  if (tt_low_node_ptr != nullptr) {
    picked_node_ref.tt_low_node = tt_low_node_ptr;
    picked_node_ref.is_tt_hit = true;
    // If TT hit, policy priors for Beta init should come from the TT entry's stored policy,
    // or this node should not be re-initialized if it's already mature.
    // The SetLowNode call later will handle this.
  } else {
    // Twin logic (if enabled by params_.GetMoveRuleBucketing())
    // ... (original twin logic from search (13).cc) ...
    picked_node_ref.lock = NNCacheLock(search_->cache_, picked_node_ref.hash);
    picked_node_ref.is_cache_hit = static_cast<bool>(picked_node_ref.lock); // Check lock validity
  }
  // After this, FetchMinibatchResults will get NN eval (if miss) or use TT/cache.
  // Then, node_to_extend->InitializeBetaPriors should be called with the policy from NN/TT/cache.
}


// 2b. Copy collisions into shared collisions.
void SearchWorker::CollectCollisions() {
  SharedMutex::Lock lock(search_->nodes_mutex_); // Full lock for modifying shared_collisions_

  for (const NodeToProcess& node_proc : minibatch_) { // Renamed
    if (node_proc.IsCollision()) {
      search_->shared_collisions_.emplace_back(node_proc.path,
                                               node_proc.multivisit);
    }
  }
}

// 4. Run NN computation.
// ~~~~~~~~~~~~~~~~~~~~~~
void SearchWorker::RunNNComputation() {
  if (computation_->GetCacheMisses() > 0 || computation_->GetCacheHits() > 0) { // Only if there's work
    computation_->ComputeBlocking(params_.GetPolicySoftmaxTemp());
  }
}

// 5. Retrieve NN computations (and terminal values) into nodes.
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void SearchWorker::FetchMinibatchResults() {
  SharedMutex::Lock nodes_lock(search_->nodes_mutex_); // Full lock as SetLowNode/InitializeBetaPriors modifies nodes
  int computation_idx = 0; // Renamed
  for (auto& node_proc : minibatch_) { // Renamed, use ref
    FetchSingleNodeResult(&node_proc, *computation_, computation_idx);
    if (node_proc.ShouldAddToInput()) ++computation_idx;

    // *** IMPORTANT: Initialize Beta Priors here ***
    // After NN eval is fetched (or TT/cache used) and LowNode is set,
    // and if the node was actually expanded (has edges and policy from NN).
    if (node_proc.nn_queried && node_proc.node && node_proc.node->GetLowNode() && !node_proc.node->IsTerminal()) {
        // Extract policy from LowNode's NNEval
        auto nn_eval_ptr = node_proc.node->GetLowNode()->nn_eval_; // Assuming nn_eval_ is public or via getter
        if (nn_eval_ptr && !nn_eval_ptr->policy.empty()) {
            // Collate policy probabilities for the edges of node_proc.node
            // This requires mapping nn_eval_ptr->policy (often full policy map)
            // to the actual legal moves/edges present in node_proc.node.
            std::vector<float> policy_for_edges(node_proc.node->GetNumEdges());
            // This mapping logic depends on how NNEval stores policy and how Node stores edges.
            // Assuming policy in NNEval is ordered corresponding to legal moves used to create edges.
            // Or that Edge has a method to get its policy_idx.
            // For now, direct mapping (Placeholder logic)
            const auto& nn_policy_map = nn_eval_ptr->policy; // This is map<Move, float> in NNEval
            for(uint16_t i=0; i<node_proc.node->GetNumEdges(); ++i) {
                Move edge_move = node_proc.node->GetEdge(i).GetMove();
                auto policy_it = nn_policy_map.find(edge_move);
                if (policy_it != nn_policy_map.end()) {
                    policy_for_edges[i] = policy_it->second;
                } else {
                    policy_for_edges[i] = 0.0f; // Should not happen if policy covers all legal moves
                }
            }
            node_proc.node->InitializeBetaPriors(policy_for_edges, params_);
        }
    }
  }
}


template <typename ComputationType> // Renamed from Computation
void SearchWorker::FetchSingleNodeResult(NodeToProcess* node_proc_ptr, // Renamed
                                         const ComputationType& computation_obj, // Renamed
                                         int idx_in_computation_obj) // Renamed
    REQUIRES(search_->nodes_mutex_) {
  if (!node_proc_ptr->nn_queried) return;

  Node* current_processing_node = node_proc_ptr->node; // Renamed

  if (!node_proc_ptr->is_tt_hit) {
    if (node_proc_ptr->is_twin_hit) {
      // Twin logic assumes twin_low_node is valid and contains data of another position.
      // We create a new LowNode for current hash, copying data from twin, then mark as twin.
      LowNode twin_data_copy = *(node_proc_ptr->twin_low_node); // Copy data
      auto [new_tt_low_node, is_miss] = search_->dag_->TTGetOrCreate(twin_data_copy, node_proc_ptr->hash); // Create with copied data
      assert(new_tt_low_node != nullptr);
      new_tt_low_node->MakeTwin(); // Mark as twin
      node_proc_ptr->tt_low_node = new_tt_low_node;
    } else {
      auto [new_tt_low_node, is_miss] = search_->dag_->TTGetOrCreate(node_proc_ptr->hash); // Renamed
      assert(new_tt_low_node != nullptr);
      node_proc_ptr->tt_low_node = new_tt_low_node;

      if (is_miss) { // True TT miss, need to set NN eval
        auto nn_eval_result = computation_obj.GetNNEval(idx_in_computation_obj); // Renamed
        if (!nn_eval_result) { /* Handle error or empty result */ return; }
        // WDL Rescaling (original logic from search (13).cc)
        if (params_.GetWDLRescaleRatio() != 1.0f ||
            (params_.GetWDLRescaleDiff() != 0.0f &&
             search_->contempt_mode_ != ContemptMode::NONE)) {
          bool root_stm_flag = (search_->contempt_mode_ == ContemptMode::WHITE); // Renamed
          auto sign_val = (root_stm_flag ^ node_proc_ptr->history.IsBlackToMove()) ? 1.0f : -1.0f; // Renamed
          float v_val = nn_eval_result->q; // Renamed
          float d_val = nn_eval_result->d; // Renamed
          WDLRescale(v_val, d_val, params_.GetWDLRescaleRatio(),
                     search_->contempt_mode_ == ContemptMode::NONE ? 0 : params_.GetWDLRescaleDiff(),
                     sign_val, false);
          nn_eval_result->q = v_val;
          nn_eval_result->d = d_val;
        }
        node_proc_ptr->tt_low_node->SetNNEval(nn_eval_result);
        node_proc_ptr->tt_low_node->SetCHHash(node_proc_ptr->ch_hash); // If CH is used
      }
    }
  }
  // At this point, node_proc_ptr->tt_low_node should be valid.
  current_processing_node->SetLowNode(node_proc_ptr->tt_low_node);

  // Apply Dirichlet noise if at root and enabled.
  // This should affect policy priors (P values) on edges before Beta init.
  if (params_.GetNoiseEpsilon() > 0.0f && current_processing_node == search_->root_node_ && current_processing_node->GetN() == 0) { // Apply only on first expansion
    // ApplyDirichletNoise modifies edge P values.
    ApplyDirichletNoise(current_processing_node, params_.GetNoiseEpsilon(), params_.GetNoiseAlpha(), params_);
    // After noise, edges need to be sorted if subsequent logic depends on P-sorted edges.
    // current_processing_node->SortEdgesByPolicy(); // If Node has such a method.
  }
  // Beta prior initialization happens after this function, in the loop in FetchMinibatchResults.
}


// 6. Propagate the new nodes' information to all their parents in the tree.
// ~~~~~~~~~~~~~~
void SearchWorker::DoBackupUpdate() {
  SharedMutex::Lock lock(search_->nodes_mutex_); // Full lock for modifications

  bool any_work_done = (number_out_of_order_ > 0); // Renamed
  for (const NodeToProcess& node_proc : minibatch_) { // Renamed
    if (!node_proc.IsCollision()) { // Only backup non-collisions here
        DoBackupUpdateSingleNode(node_proc);
        any_work_done = true;
    }
  }
  if (!any_work_done && search_->shared_collisions_.empty()) return; // Skip if no actual updates or collisions

  // Handle shared collisions (original logic seems to decrement NInFlight)
  // For Thompson Sampling, this means some paths chosen for exploration did not result in new NN eval.
  // Their Beta stats should not be updated with an NN eval.
  // The NInFlight decrement is important.
  search_->CancelSharedCollisions(); // This handles NInFlight for collisions.
  search_->total_batches_ += 1;
}


// This function needs significant changes for Thompson Sampling backup.
// The core idea is to update BetaBernoulliStats on the EDGES.
// The node's own Q, W, L values might still be updated for other purposes (e.g., resign, mate scores).
void SearchWorker::DoBackupUpdateSingleNode(
    const NodeToProcess& node_proc) REQUIRES(search_->nodes_mutex_) { // Renamed
  
  if (node_proc.IsCollision()) return; // Collisions handled by CancelSharedCollisions for NInFlight

  auto& backup_path_ref = node_proc.path; // Renamed
  if (backup_path_ref.empty()) return;

  Node* leaf_node = std::get<0>(backup_path_ref.back());
  if (!leaf_node) return; // Should not happen

  float value_to_backup; // This is Q-like value in [-1, 1] from leaf's perspective

  // Determine the value from the leaf node
  if (leaf_node->IsTerminal()) {
    // Convert GameResult to value: WIN = 1, LOSS = -1, DRAW = 0
    // This depends on whose perspective GameResult is. Assume it's absolute (WhiteWon, BlackWon).
    // And needs to be converted to current player at leaf_node's perspective.
    // For simplicity, assume GetWL() on terminal node gives value from its perspective.
    value_to_backup = leaf_node->GetWL(); // Assumes Node::GetWL returns value for terminal node
  } else if (leaf_node->GetLowNode()) {
    // Value from NN (LowNode). LowNode's Q is from its perspective.
    value_to_backup = leaf_node->GetLowNode()->GetWL();
  } else {
    // Should not happen for a node that was processed. Maybe an error or uninitialized.
    return; 
  }

  // *** Core Thompson Sampling Backup ***
  // This call will iterate up the path, flip signs, and update BetaStats on edges.
  leaf_node->BackupValueToAncestors(value_to_backup, params_);
  // BackupValueToAncestors also increments node visit counts (n_).

  // The rest of the original DoBackupUpdateSingleNode from search (13).cc deals with:
  // 1. Updating Node's own Q, D, M values (wl_, d_, m_).
  // 2. Handling repetitions.
  // 3. Propagating bounds for sticky endgames.
  // 4. Adjusting for transpositions/terminals found deeper.
  // This logic can largely remain if these Node-level stats are still desired for other heuristics
  // (resign, mate scores, display). Thompson Sampling itself primarily uses edge BetaStats for selection.

  // --- Start of adapted original backup logic for Node Q,D,M and bounds ---
  // (This part is complex and highly dependent on Leela's specific Node/LowNode interactions)
  // For brevity and focus on TS, this part is simplified. A full port would need careful integration.
  // The key is that BetaStats on edges are updated by BackupValueToAncestors.
  // What follows is for updating node's own (non-TS) stats like Q, D, M, bounds.

  bool update_parent_bounds_flag = params_.GetStickyEndgames() && leaf_node->IsTerminal() && leaf_node->GetN() == 1; // Only on first visit to new terminal
  LowNode* current_low_node = leaf_node->GetLowNode(); // Renamed

  float v_prop = value_to_backup; // Value from current node's perspective
  float d_prop = current_low_node ? current_low_node->GetD() : 0.0f;
  float m_prop = current_low_node ? current_low_node->GetM() : 0.0f;
  if (leaf_node->IsTerminal()) m_prop = leaf_node->GetM(); // Use node's M if terminal

  // Backup Node's Q,D,M values (not Beta stats) up the tree
  Node* n_iter = leaf_node;
  int current_depth_in_path = backup_path_ref.size() -1;

  while(n_iter) {
      // Update n_iter's Q, D, M based on v_prop, d_prop, m_prop
      // n_iter->SetWL(v_prop); n_iter->SetD(d_prop); n_iter->SetM(m_prop); // Example
      // Original code has complex FinalizeScoreUpdate and AdjustForTerminal logic.
      // This is a placeholder for that logic.

      // If repetitions make this node a draw
      int path_reps = std::get<1>(backup_path_ref[current_depth_in_path]);
      if (path_reps >=2 && !n_iter->IsTerminal()) { // 3-fold repetition
          // Treat as draw for propagation upwards
          v_prop = 0.0f; d_prop = 1.0f;
          // m_prop = some small value for draw by rep
      }

      // Bound setting logic
      if (update_parent_bounds_flag && n_iter->GetParent() && n_iter->GetParent() != search_->root_node_) {
          // update_parent_bounds_flag = MaybeSetBounds(n_iter->GetParent(), m_prop_for_parent, ...);
          // This needs careful porting of MaybeSetBounds and its params.
      } else {
          update_parent_bounds_flag = false;
      }


      if (!n_iter->GetParent()) break; // Reached root of path segment
      
      // Prepare for parent
      v_prop = -v_prop; // Flip value for parent
      m_prop += 1.0f;   // Increment M for parent
      // d_prop usually doesn't flip.

      n_iter = n_iter->GetParent();
      current_depth_in_path--;
      if (current_depth_in_path < 0) break; // Should not happen if parent exists
  }
  // --- End of adapted original backup logic ---


  // Update global search stats (already done by BackupValueToAncestors for visits)
  search_->total_playouts_ += node_proc.multivisit; // If BackupValueToAncestors doesn't handle multivisit for this
  search_->cum_depth_ += node_proc.path.size() * node_proc.multivisit;
  search_->max_depth_ = std::max(search_->max_depth_, static_cast<uint16_t>(node_proc.path.size()));
  if (!node_proc.is_tt_hit && !node_proc.is_twin_hit) { // If it was a true NN query source
    search_->total_low_nodes_++; // Count new LowNodes created from NN
  }
  if (node_proc.ShouldAddToInput()) { // Count actual NN queries
    search_->total_nn_queries_++;
  }

  // Update current_best_edge_ if root's children changed significantly
  // This logic is complex and depends on how "best" is defined with TS (visits or BetaMean)
  // Original logic:
  if (leaf_node && leaf_node->GetParent() == search_->root_node_) { // If leaf_node is a child of root
      // Check if this update could change the best move from root
      // (e.g. if its visit count or BetaMean now makes it best)
      // This needs GetBestChildNoTemperature to be efficient or called sparingly.
      // search_->current_best_edge_ = search_->GetBestChildNoTemperature(search_->root_node_, 0);
      // For performance, this might be deferred or updated based on simpler heuristics.
  }
}


bool SearchWorker::MaybeSetBounds(Node* parent_node, float m_val_for_parent, uint32_t* n_to_fix, // Renamed
                                  float* weight_to_fix, float* v_delta, float* d_delta,
                                  float* m_delta, float* vs_delta) const REQUIRES(search_->nodes_mutex_) {
  // This function is highly specific to Leela's bounds propagation and terminal handling.
  // A full port requires understanding Node::GetBounds(), Edge::GetBounds(), MakeTerminal(), SetBounds().
  // For Thompson Sampling, the main effect is determining if a node becomes provably terminal,
  // which then affects the values backed up for BetaStats.
  // Placeholder:
  return false;
}

// 7. Update the Search's status and progress information.
//~~~~~~~~~~~~~~~~~~~~
void SearchWorker::UpdateCounters() {
  search_->PopulateCommonIterationStats(&iteration_stats_);
  search_->MaybeTriggerStop(iteration_stats_, &latest_time_manager_hints_);
  search_->MaybeOutputInfo(); // This might try to acquire nodes_mutex again, ensure calling context is okay

  bool any_work_done_this_iter = (number_out_of_order_ > 0); // Renamed
  if (!any_work_done_this_iter) {
    for (NodeToProcess& node_proc : minibatch_) { // Renamed
      if (!node_proc.IsCollision()) {
        any_work_done_this_iter = true;
        break;
      }
    }
  }
  if (!any_work_done_this_iter && minibatch_.empty() && search_->shared_collisions_.empty()) { // Check minibatch and collisions
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

}  // namespace lczero
