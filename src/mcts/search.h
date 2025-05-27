/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2019 The LCZero Team

  Leela Chess Zero is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess Zero is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess Zero.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <memory>
#include <random>
#include <shared_mutex>
#include <atomic>
#include <vector>
#include <thread>
#include <mutex>
#include <chrono>
#include "neural/cache.h"
#include "neural/network.h"
#include "utils/optionsdict.h"
#include "utils/optionsparser.h"
#include "chess/bitboard.h"
#include "chess/board.h"
#include "src/chess/callbacks.h"
#include "src/mcts/node.h"

namespace lczero {

// Forward declarations
class Node;
class NodeTree;
class SearchBehaviorParams;
class SyzygyTablebase;

// Supporting structures
struct SearchScopeIterationStats {
  uint64_t total_nodes = 0;
  uint64_t nodes_since_last = 0;
  uint32_t batches_pending = 0;
  float average_depth = 0.0f;
  int64_t time_since_movestart = 0;
  int64_t time_since_first_batch = 0;
};

struct SearchStats {
  uint64_t total_nodes = 0;
  uint64_t nodes_since_last = 0;
  uint32_t batches_pending = 0;
  float average_depth = 0.0f;
  int64_t time_since_movestart = 0;
  int64_t time_since_first_batch = 0;
};

/*
struct ThinkingInfo {
  int multipv = 1;
  int depth = 0;
  int seldepth = 0;
  int64_t time = 0;
  uint64_t nodes = 0;
  int score = 0;
  std::vector<Move> pv;
};
*/

/*
struct BestMoveInfo {
  Move bestmove;
  Move ponder;
  using Callback = std::function<void(const BestMoveInfo&)>;
};
*/

enum class StoppageReason {
  VISITS_LIMIT,
  TIME_LIMIT,
  MOVETIME_LIMIT,
  HIGH_CONFIDENCE,
  USER_STOP
};

enum class GameResult {
  UNDECIDED,
  WHITE_WON,
  BLACK_WON,
  DRAW
};

// Simple SearchParams class for configuration
class SearchBehaviorParams {
public:
  SearchBehaviorParams() = default;
  
  void SetFpuValue(float value) { fpu_value_ = value; }
  float GetFpuValue() const { return fpu_value_; }
  
  void SetPolicyTemperature(float temp) { policy_temperature_ = temp; }
  float GetPolicyTemperature() const { return policy_temperature_; }
  
  void SetUsePolicyPriors(bool use) { use_policy_priors_ = use; }
  bool GetUsePolicyPriors() const { return use_policy_priors_; }

private:
  float fpu_value_ = 0.0f;
  float policy_temperature_ = 1.0f;
  bool use_policy_priors_ = true;
};

// Forward declare SearchWorker
class SearchWorker;

// Beta-Bernoulli Thompson Sampling data for each edge
struct BetaBernoulliStats {
  // Beta distribution parameters
  std::atomic<double> alpha{1.0};
  std::atomic<double> beta{1.0};
  
  // Traditional stats for compatibility
  std::atomic<int> visits{0};
  std::atomic<double> value_sum{0.0};
  
  BetaBernoulliStats() = default;
  
  // Copy constructor for atomic members
  BetaBernoulliStats(const BetaBernoulliStats& other) 
    : alpha(other.alpha.load())
    , beta(other.beta.load())
    , visits(other.visits.load())
    , value_sum(other.value_sum.load()) {}
  
  // Update with game outcome value in [-1, 1]
  void Update(double value) {
    visits.fetch_add(1, std::memory_order_relaxed);
    double old_value_sum = value_sum.load(std::memory_order_relaxed);
    while (!value_sum.compare_exchange_weak(old_value_sum, old_value_sum + value,
                                           std::memory_order_relaxed, std::memory_order_relaxed)) {}
    
    // Convert value to probability [0, 1]
    double prob = (value + 1.0) / 2.0;
    
    // Update Beta parameters atomically
    double old_alpha = alpha.load(std::memory_order_relaxed);
    while (!alpha.compare_exchange_weak(old_alpha, old_alpha + prob, 
                                       std::memory_order_relaxed)) {}
    
    double old_beta = beta.load(std::memory_order_relaxed);
    while (!beta.compare_exchange_weak(old_beta, old_beta + (1.0 - prob),
                                      std::memory_order_relaxed)) {}
  }
  
  // Sample from Beta distribution for Thompson Sampling
  double SampleBeta(std::mt19937& rng) const {
    double a = alpha.load(std::memory_order_relaxed);
    double b = beta.load(std::memory_order_relaxed);
    
    if (visits.load(std::memory_order_relaxed) == 0) {
      return 0.5; // Uniform prior for unvisited nodes
    }
    
    // Use Gamma sampling for Beta distribution
    std::gamma_distribution<double> gamma_a(a, 1.0);
    std::gamma_distribution<double> gamma_b(b, 1.0);
    
    double x = gamma_a(rng);
    double y = gamma_b(rng);
    
    if (x + y == 0.0) return 0.5;
    return x / (x + y);
  }
  
  // Get Beta distribution mean
  double GetBetaMean() const {
    double a = alpha.load(std::memory_order_relaxed);
    double b = beta.load(std::memory_order_relaxed);
    return a / (a + b);
  }
  
  // Get uncertainty (Beta variance)
  double GetUncertainty() const {
    double a = alpha.load(std::memory_order_relaxed);
    double b = beta.load(std::memory_order_relaxed);
    double total = a + b;
    return (a * b) / (total * total * (total + 1));
  }
  
  // Get traditional stats for compatibility
  double GetMeanValue() const {
    int v = visits.load(std::memory_order_relaxed);
    if (v == 0) return 0.0;
    return value_sum.load(std::memory_order_relaxed) / v;
  }
  
  int GetVisits() const {
    return visits.load(std::memory_order_relaxed);
  }
};

/*
class Edge {
 public:
  // Move
  Move GetMove(bool as_opponent = false) const {
    return as_opponent ? move_.flipped() : move_;  
  }
  void SetMove(Move move) { move_ = move; }

  // Policy
  float GetP() const { return p_; }
  void SetP(float val) { p_ = val; }

  // Beta-Bernoulli statistics
  BetaBernoulliStats& GetBetaStats() { return beta_stats_; }
  const BetaBernoulliStats& GetBetaStats() const { return beta_stats_; }

  // Legacy methods for compatibility
  float GetQ(float default_q) const { 
    return beta_stats_.GetMeanValue(); 
  }
  
  uint32_t GetN() const { 
    return beta_stats_.GetVisits(); 
  }
  
  float GetU(float cpuct, float numerator) const {
    // For compatibility, return Thompson sample instead of UCB
    thread_local std::mt19937 rng{std::random_device{}()};
    return beta_stats_.SampleBeta(rng);
  }

  // Node
  Node* GetNode() const { return node_.get(); }
  Node* GetOrSpawnNode(Node* parent);
  void SetNode(std::unique_ptr<Node> node) { node_ = std::move(node); }

  bool HasNode() const { return node_ != nullptr; }
  bool IsTerminal() const;

  std::string DebugString() const;

 private:
  // Move corresponding to this node. From the perspective of player to move at
  // the parent node.
  Move move_;
  
  // Probability that this move should be played, from the policy head of the
  // neural network (but after adding Dirichlet noise and possibly other transformations).
  float p_ = 0.0f;
  
  // Beta-Bernoulli Thompson Sampling statistics
  BetaBernoulliStats beta_stats_;
  
  // Node that this edge points to.
  std::unique_ptr<Node> node_;
};
*/

// typedef std::vector<Edge> EdgeList; // EdgeList is defined in node.h

/*
class Node {
 public:
  enum class Terminal : uint8_t { NonTerminal, EndOfGame, Tablebase };

  Node(Node* parent, uint16_t index)
      : parent_(parent), index_(index) {}

  void MakeTerminal(GameResult result, float plies_left = 0.0f,
                    Terminal type = Terminal::EndOfGame);

  bool IsTerminal() const { return terminal_type_ != Terminal::NonTerminal; }
  uint16_t GetNumEdges() const { return edges_.size(); }
  float GetVisitedPolicy() const;

  // Edge related.
  Edge* GetEdges() { return edges_.data(); }
  const Edge* GetEdges() const { return edges_.data(); }
  Edge& GetEdge(uint16_t id) { return edges_[id]; }
  const Edge& GetEdge(uint16_t id) const { return edges_[id]; }

  // Children related.
  uint32_t GetN() const { return n_; }
  uint32_t GetChildrenVisits() const { return n_in_flight_ + n_; }
  uint32_t GetInFlightVisits() const { return n_in_flight_; }
  void IncrementNInFlight() { ++n_in_flight_; }
  void DecrementNInFlight() { --n_in_flight_; }

  // Thompson Sampling edge selection
  uint16_t SelectChildThompsonSampling(std::mt19937& rng, float fpu_value = 0.0f) const;
  
  // Legacy UCB selection for compatibility
  uint16_t SelectChildToExtend(float cpuct, float fpu_value, 
                               bool fpu_absolute) const;

  // Backup a value and update Beta-Bernoulli statistics
  void BackupValue(float value);
  void BackupValueToAncestors(float value);

  void MakeNotTerminal() { terminal_type_ = Terminal::NonTerminal; }
  void CreateEdges(const std::vector<Move>& moves);

  // Initialize Beta parameters with policy priors
  void InitializeBetaPriors(const std::vector<float>& policy_probs);
  void SortEdges();
  std::string DebugString() const;
  bool HasChildren() const { return !edges_.empty(); }

  // Getters.
  float GetQ(float default_q) const { return wl_; }
  float GetD() const { return d_; }
  float GetM() const { return m_; }
  uint32_t GetIndex() const { return index_; }
  Node* GetParent() const { return parent_; }

  // Value head (from neural network evaluation)
  void SetWL(float wl) { wl_ = wl; }
  void SetD(float d) { d_ = d; }
  void SetM(float m) { m_ = m; }

  GameResult GetResult() const { return result_; }
  Terminal GetTerminalType() const { return terminal_type_; }
  float GetPliesLeft() const { return plies_left_; }

  void ReleaseChildren();
  void ReleaseChildrenExceptOne(Node* subtree_root);

  float GetRawQ(float default_q) const;

 private:
  // For nodes without visits, use parent's values
  void SetBounds();

  // Parent and tree structure
  Node* parent_ = nullptr;
  uint16_t index_;

  // Edges to child nodes
  EdgeList edges_;

  // Visits and in-flight visits
  uint32_t n_ = 0;
  uint32_t n_in_flight_ = 0;

  // Values from neural network evaluation
  float wl_ = 0.0f;  // Win/Loss probability difference
  float d_ = 0.0f;   // Draw probability  
  float m_ = 0.0f;   // Moves left

  // Terminal status
  Terminal terminal_type_ = Terminal::NonTerminal;
  GameResult result_ = GameResult::UNDECIDED;
  float plies_left_ = 0.0f;

  mutable std::shared_mutex mutex_;
};
*/

class NodeTree {
 public:
  ~NodeTree() { DeallocateTree(); }
  
  // Returns whether node was created.
  std::pair<Node*, bool> GetOrCreateChildAt(Node* node, int child_idx);
  void MakeMove(Move move);
  void TrimTreeAtHead();
  Node* GetCurrentHead() const { return current_head_; }
  Node* GetGameBeginNode() const { return gamebegin_node_.get(); }
  void DeallocateTree();
  void ResetToPosition(const std::string& starting_fen,
                       const std::vector<Move>& moves);

 private:
  std::unique_ptr<Node> gamebegin_node_;
  Node* current_head_ = nullptr;
  std::vector<Move> history_;
};

struct SearchLimits {
  std::int64_t visits = -1;
  std::int64_t time_ms = -1;
  std::int64_t movetime = -1;
  int depth = -1;
  std::vector<Move> searchmoves;
  bool infinite = false;
  bool ponder = false;
};

class Search {
 public:
  Search(const NodeTree& tree, Network* network,
         BestMoveInfo::Callback best_move_callback,
         ThinkingInfo::Callback info_callback, const SearchLimits& limits,
         const OptionsDict& options, NNCache* cache,
         SyzygyTablebase* syzygy_tb);

  ~Search();

  // Populates UciOptions with search parameters.
  static void PopulateUciParams(OptionsParser* options);

  // Starts worker threads and returns immediately.
  void StartThreads(size_t how_many);

  // Stops search. Will return as soon as possible.
  void Stop();
  
  // Stops search and waits for threads to stop.
  void Abort();
  
  // Wait for search to stop naturally (game end/limits reached).
  void Wait();
  
  // Returns whether search is active.
  bool IsSearchActive() const;

  // Returns best move, from the point of view of the player to move.
  std::pair<Move, Move> GetBestMove() const;
  
  // Returns the evaluation of the best move, NOT from the point of view of the
  // player to move (i.e. reversed).
  float GetBestEval() const;
  
  // Returns multiPV info for current search state.
  std::vector<ThinkingInfo> GetMultiPvInfo() const;

  // Returns search statistics.
  SearchStats GetStats() const;

 private:
  // Can run multiple times.
  void RunBlocking();
  void SendUciInfo();  
  int64_t GetTimeSinceStart() const;
  int64_t GetTimeSinceFirstBatch() const;
  void MaybeTriggerStop(const SearchScopeIterationStats& stats, StoppageReason* reason);
  void MaybeOutputInfo();
  
  // Sets up search parameters from UCI options  
  void SetupSearchParams(const OptionsDict& options);

  const NodeTree& played_history_;
  Network* const network_;
  const SearchLimits limits_;
  const std::chrono::steady_clock::time_point start_time_;
  const BestMoveInfo::Callback best_move_callback_;
  const ThinkingInfo::Callback info_callback_;

  mutable std::mutex threads_mutex_;
  std::vector<std::thread> threads_;
  std::atomic<bool> stop_{false};

  Node* root_node_;
  
  // Search parameters
  std::unique_ptr<SearchBehaviorParams> params_;
  NNCache* cache_;
  SyzygyTablebase* syzygy_tb_;

  // Random number generator for Thompson Sampling
  mutable std::mt19937 rng_;
  
  // Friend class for worker access
  friend class SearchWorker;
};

}  // namespace lczero
