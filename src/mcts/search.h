/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors

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

#pragma once

#include <array>
#include <atomic> // Required for BetaBernoulliStats
#include <condition_variable>
#include <functional>
#include <memory> // Required for std::unique_ptr
#include <optional>
#include <random> // Required for std::mt19937 and distributions
#include <shared_mutex>
#include <thread>
#include <tuple>
#include <vector>

#include "chess/callbacks.h"
#include "chess/uciloop.h"
#include "mcts/node.h" // Assumed to define base Node, Edge, Move, GameResult etc. or that we define them below.
#include "mcts/params.h"
#include "mcts/stoppers/timemgr.h"
#include "neural/cache.h"
#include "neural/network.h" // Required for Network NNEval
#include "syzygy/syzygy.h"
#include "utils/logging.h"
#include "utils/mutex.h"
#include "utils/optionsdict.h" // Required for OptionsDict
#include "utils/optionsparser.h" // Required for OptionsParser

namespace lczero {

// Forward declarations if not fully defined by mcts/node.h
// class Node; // Already forward declared in original search (6).h via mcts/node.h
// class Edge; // Not in original search (6).h, assume it's in mcts/node.h
// struct Move; // Assume defined
// enum class GameResult; // Assume defined

// **************************************************************************
// Definitions for Beta-Bernoulli Thompson Sampling
// These would typically go into mcts/node.h or a similar core MCTS header.
// Based on leela_search_h (1).txt
// **************************************************************************

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
      : alpha(other.alpha.load()),
        beta(other.beta.load()),
        visits(other.visits.load()),
        value_sum(other.value_sum.load()) {}
  
  BetaBernoulliStats& operator=(const BetaBernoulliStats& other) {
    if (this != &other) {
      alpha.store(other.alpha.load());
      beta.store(other.beta.load());
      visits.store(other.visits.load());
      value_sum.store(other.value_sum.load());
    }
    return *this;
  }

  // Update with game outcome value in [-1, 1]
  void Update(double value) {
    visits.fetch_add(1, std::memory_order_relaxed);
    value_sum.fetch_add(value, std::memory_order_relaxed);

    // Convert value to probability [0, 1]
    double prob = (value + 1.0) / 2.0;

    // Update Beta parameters atomically
    double old_alpha = alpha.load(std::memory_order_relaxed);
    while (!alpha.compare_exchange_weak(old_alpha, old_alpha + prob,
                                        std::memory_order_acq_rel, std::memory_order_relaxed)) {
    }

    double old_beta = beta.load(std::memory_order_relaxed);
    while (!beta.compare_exchange_weak(old_beta, old_beta + (1.0 - prob),
                                       std::memory_order_acq_rel, std::memory_order_relaxed)) {
    }
  }

  // Sample from Beta distribution for Thompson Sampling
  double SampleBeta(std::mt19937& rng) const {
    double a = alpha.load(std::memory_order_relaxed);
    double b = beta.load(std::memory_order_relaxed);

    if (visits.load(std::memory_order_relaxed) == 0) {
      // For unvisited nodes, a common practice is to return a prior,
      // or rely on FPU. Here, a 0.5 can be a placeholder if FPU handles it.
      // Alternatively, could use a more informed prior if available.
      return 0.5; // Uniform prior for unvisited nodes for pure sampling
    }

    // Use Gamma sampling for Beta distribution
    std::gamma_distribution<double> gamma_a(a, 1.0);
    std::gamma_distribution<double> gamma_b(b, 1.0);

    double x = gamma_a(rng);
    double y = gamma_b(rng);

    if (x + y == 0.0) return 0.5; // Avoid division by zero
    return x / (x + y);
  }

  // Get Beta distribution mean (expected win rate from this edge's perspective)
  // Result is in [0, 1]
  double GetBetaMeanProb() const {
    double a = alpha.load(std::memory_order_relaxed);
    double b = beta.load(std::memory_order_relaxed);
    if (a + b == 0.0) return 0.5; // Should not happen with alpha, beta >= 1
    return a / (a + b);
  }
  
  // Get Beta mean converted to [-1, 1] range (Q-value like)
  double GetBetaMeanValue() const {
      return 2.0 * GetBetaMeanProb() - 1.0;
  }

  // Get uncertainty (Beta variance)
  double GetUncertainty() const {
    double a = alpha.load(std::memory_order_relaxed);
    double b = beta.load(std::memory_order_relaxed);
    double total = a + b;
    if (total == 0.0 || total + 1.0 == 0.0) return 1.0; // High uncertainty
    return (a * b) / (total * total * (total + 1.0));
  }

  // Get traditional stats for compatibility
  double GetMeanValue() const { // Traditional Q in [-1, 1]
    int v = visits.load(std::memory_order_relaxed);
    if (v == 0) return 0.0; // Or parent's Q / FPU
    return value_sum.load(std::memory_order_relaxed) / v;
  }

  int GetVisits() const {
    return visits.load(std::memory_order_relaxed);
  }
};

// Assuming Edge is defined in mcts/node.h and looks something like this.
// We need to ensure it contains BetaBernoulliStats.
// If Edge is not defined externally, this definition can be used.
#ifndef EDGE_DEFINED_EXTERNALLY // Guard in case mcts/node.h defines Edge
#define EDGE_DEFINED_EXTERNALLY
class Edge {
 public:
  // Default constructor
  Edge() : p_(0.0f), node_(nullptr) {}
  // Constructor with move
  Edge(Move move) : move_(move), p_(0.0f), node_(nullptr) {}


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

  // Legacy methods for compatibility or specific uses
  float GetQ(float default_q) const {
    if (beta_stats_.GetVisits() == 0) return default_q;
    return beta_stats_.GetMeanValue(); // Traditional Q
  }

  uint32_t GetN() const {
    return beta_stats_.GetVisits();
  }

  // Node
  Node* GetNode() const { return node_.get(); }
  Node* GetOrSpawnNode(Node* parent, uint16_t index_in_parent); // Modified to pass index
  void SetNode(std::unique_ptr<Node> node) { node_ = std::move(node); }

  bool HasNode() const { return node_ != nullptr; }
  // bool IsTerminal() const; // Original Edge in search (13).cc doesn't have this. Node does.

  std::string DebugString() const; // Implementation in .cc

 private:
  Move move_;
  float p_ = 0.0f; // Policy prior
  BetaBernoulliStats beta_stats_;
  std::unique_ptr<Node> node_; // Child node
};
#endif // EDGE_DEFINED_EXTERNALLY

// Add to Node class (assuming it's defined in mcts/node.h or here)
// Methods related to Thompson Sampling.
// If Node is not defined externally, this definition can be used.
#ifndef NODE_DEFINED_EXTERNALLY // Guard in case mcts/node.h defines Node
#define NODE_DEFINED_EXTERNALLY

typedef std::vector<Edge> EdgeList;

class Node {
 public:
  // enum class Terminal : uint8_t { NonTerminal, EndOfGame, Tablebase }; // From leela_search_h (1)

  Node(Node* parent, uint16_t index_in_parent)
      : parent_(parent), index_in_parent_(index_in_parent) {}


  // Thompson Sampling edge selection
  // Needs SearchParams for FPU handling and policy prior usage flag
  uint16_t SelectChildThompsonSampling(std::mt19937& rng, const SearchParams& params, float parent_q_for_fpu) const;

  // Backup a value and update Beta-Bernoulli statistics on the edge leading to this node
  void BackupValueToAncestors(float value, const SearchParams& params);

  // Create edges for legal moves
  void CreateEdges(const std::vector<Move>& moves, const Position& pos);

  // Initialize Beta parameters with policy priors
  void InitializeBetaPriors(const std::vector<float>& policy_probs, const SearchParams& params);
  
  Edge& GetEdge(uint16_t id) { return edges_[id]; }
  const Edge& GetEdge(uint16_t id) const { return edges_[id]; }
  uint16_t GetNumEdges() const { return static_cast<uint16_t>(edges_.size()); }
  EdgeList& Edges() { return edges_; } // For iteration
  const EdgeList& Edges() const { return edges_; }


  // --- Existing/Assumed Node members from Leela ---
  // These would be part of the full Node class definition in mcts/node.h
  Node* parent_ = nullptr;
  uint16_t index_in_parent_ = 0; // Index of this node in parent's edge list
  EdgeList edges_;
  std::atomic<uint32_t> n_{0}; // Visit count for this node state
  // Other members like Q-value (wl_), D-value (d_), M-value (m_), terminal status, etc.
  // float wl_ = 0.0f, d_ = 0.0f, m_ = 0.0f;
  // bool is_terminal_ = false; GameResult result_;
  // LowNode* low_node_ = nullptr;
  // void SetLowNode(LowNode* n); LowNode* GetLowNode();
  // bool IsTerminal() const; void MakeTerminal(...);
  // uint32_t GetN() const { return n_.load(); }
  // float GetQ(float default_q) const;
  // void IncrementNInFlight(int count = 1); void DecrementNInFlight(int count = 1);
  // ... other methods from Leela's Node class
  // --- End Existing/Assumed Node members ---


  // Minimalistic example for compilation, actual Node is more complex
  bool IsTerminal() const { return false; /* Placeholder */ }
  void MakeTerminal(GameResult result, float plies_left = 0.0f) { /* Placeholder */ }
  uint32_t GetN() const { return n_.load(std::memory_order_relaxed); }
  float GetQ(float /*default_q*/) const { return 0.0f; /* Placeholder for node's own Q */ }
  Node* GetParent() const { return parent_; }
  uint16_t GetIndexInParent() const { return index_in_parent_; }
  void SortEdgesByPolicy(); // For efficient iteration or specific strategies
  float GetVisitedPolicy() const; // Sum of P of visited children
  LowNode* GetLowNode() const { return low_node_; } // Add LowNode member
  void SetLowNode(LowNode* ln) { low_node_ = ln; }   // Add LowNode member
  void IncrementNInFlight(int count = 1) { /* Placeholder */ }
  void DecrementNInFlight(int count = 1) { /* Placeholder */ }


 private:
  LowNode* low_node_ = nullptr; // Example member, Leela's Node is complex
  // Other members like wl_, d_, m_ etc.
};
#endif // NODE_DEFINED_EXTERNALLY
// **************************************************************************
// End of Beta-Bernoulli Thompson Sampling Definitions
// **************************************************************************


typedef std::vector<std::tuple<Node*, int, int>> BackupPath;

class Search {
 public:
  Search(NodeTree* dag, Network* network,
         std::unique_ptr<UciResponder> uci_responder,
         const MoveList& searchmoves,
         std::chrono::steady_clock::time_point start_time,
         std::unique_ptr<SearchStopper> stopper, bool infinite, bool ponder,
         const OptionsDict& options, NNCache* cache,
         SyzygyTablebase* syzygy_tb);

  ~Search();

  // Starts worker threads and returns immediately.
  void StartThreads(size_t how_many);

  // Starts search with k threads and wait until it finishes.
  void RunBlocking(size_t threads);

  // Stops search. At the end bestmove will be returned. The function is not
  // blocking, so it returns before search is actually done.
  void Stop();
  // Stops search, but does not return bestmove. The function is not blocking.
  void Abort();
  // Blocks until all worker thread finish.
  void Wait();
  // Returns whether search is active. Workers check that to see whether another
  // search iteration is needed.
  bool IsSearchActive() const;

  // Returns best move, from the point of view of white player. And also ponder.
  // May or may not use temperature, according to the settings.
  std::pair<Move, Move> GetBestMove();

  // Returns the evaluation of the best move, WITHOUT temperature. This differs
  // from the above function; with temperature enabled, these two functions may
  // return results from different possible moves. If @move and @is_terminal are
  // not nullptr they are set to the best move and whether it leads to a
  // terminal node respectively.
  Eval GetBestEval(Move* move = nullptr, bool* is_terminal = nullptr) const;
  // Returns the total number of playouts in the search.
  std::int64_t GetTotalPlayouts() const;
  // Returns the search parameters.
  const SearchParams& GetParams() const { return params_; }

  // If called after GetBestMove, another call to GetBestMove will have results
  // from temperature having been applied again.
  void ResetBestMove();

  // Returns NN eval for a given node from cache, if that node is cached.
  NNCacheLock GetCachedNNEval(const PositionHistory& history) const;

  // Static method to populate UCI parameters.
  // Moved here from leela_search_cc (1).txt as it's a static method of Search
  static void PopulateUciParams(OptionsParser* options);


 private:
  // Computes the best move, maybe with temperature (according to the settings).
  void EnsureBestMoveKnown() REQUIRES(nodes_mutex_) REQUIRES(counters_mutex_);

  // Returns a child with most visits, with or without temperature.
  // NoTemperature is safe to use on non-extended nodes, while WithTemperature
  // accepts only nodes with at least 1 visited child.
  EdgeAndNode GetBestChildNoTemperature(Node* parent, int depth) const REQUIRES_SHARED(nodes_mutex_);
  std::vector<EdgeAndNode> GetBestChildrenNoTemperature(Node* parent, int count,
                                                        int depth) const REQUIRES_SHARED(nodes_mutex_);
  EdgeAndNode GetBestRootChildWithTemperature(float temperature) const REQUIRES_SHARED(nodes_mutex_);

  int64_t GetTimeSinceStart() const;
  int64_t GetTimeSinceFirstBatch() const REQUIRES(counters_mutex_);
  void MaybeTriggerStop(const IterationStats& stats, StoppersHints* hints);
  void MaybeOutputInfo();
  void SendUciInfo() REQUIRES(nodes_mutex_) REQUIRES(counters_mutex_);
  // Sets stop to true and notifies watchdog thread.
  void FireStopInternal();

  void SendMovesStats() const REQUIRES(nodes_mutex_) REQUIRES(counters_mutex_);
  // Function which runs in a separate thread and watches for time and
  // uci `stop` command;
  void WatchdogThread();

  // Fills IterationStats with global (rather than per-thread) portion of search
  // statistics. Currently all stats there (in IterationStats) are global
  // though.
  void PopulateCommonIterationStats(IterationStats* stats);

  // Returns verbose information about given node, as vector of strings.
  // Node can only be root or ponder (depth 1).
  std::vector<std::string> GetVerboseStats(Node* node) const REQUIRES_SHARED(nodes_mutex_);

  // Returns the draw score at the root of the search. At odd depth pass true to
  // the value of @is_odd_depth to change the sign of the draw score.
  // Depth of a root node is 0 (even number).
  float GetDrawScore(bool is_odd_depth) const;

  // Ensure that all shared collisions are cancelled and clear them out.
  void CancelSharedCollisions() REQUIRES(nodes_mutex_);

  mutable Mutex counters_mutex_ ACQUIRED_AFTER(nodes_mutex_);
  // Tells all threads to stop.
  std::atomic<bool> stop_{false};
  // Condition variable used to watch stop_ variable.
  std::condition_variable watchdog_cv_;
  // Tells whether it's ok to respond bestmove when limits are reached.
  // If false (e.g. during ponder or `go infinite`) the search stops but nothing
  // is responded until `stop` uci command.
  bool ok_to_respond_bestmove_ GUARDED_BY(counters_mutex_) = true;
  // There is already one thread that responded bestmove, other threads
  // should not do that.
  bool bestmove_is_sent_ GUARDED_BY(counters_mutex_) = false;
  // Stored so that in the case of non-zero temperature GetBestMove() returns
  // consistent results.
  Move final_bestmove_ GUARDED_BY(counters_mutex_);
  Move final_pondermove_ GUARDED_BY(counters_mutex_);
  std::unique_ptr<SearchStopper> stopper_ GUARDED_BY(counters_mutex_);

  Mutex threads_mutex_;
  std::vector<std::thread> threads_ GUARDED_BY(threads_mutex_);

  Node* root_node_;
  NNCache* cache_;
  NodeTree* dag_;
  SyzygyTablebase* syzygy_tb_;
  // Fixed positions which happened before the search.
  const PositionHistory& played_history_;

  Network* const network_;
  SearchParams params_; // This should be initialized from OptionsDict
  const MoveList searchmoves_;
  const std::chrono::steady_clock::time_point start_time_;
  int64_t initial_visits_;
  // root_is_in_dtz_ must be initialized before root_move_filter_.
  bool root_is_in_dtz_ = false;
  // tb_hits_ must be initialized before root_move_filter_.
  std::atomic<int> tb_hits_{0};
  const MoveList root_move_filter_;

  mutable SharedMutex nodes_mutex_;
  EdgeAndNode current_best_edge_ GUARDED_BY(nodes_mutex_);
  const Edge* last_outputted_info_edge_ GUARDED_BY(nodes_mutex_) = nullptr; // Made const Edge*
  ThinkingInfo last_outputted_uci_info_ GUARDED_BY(nodes_mutex_);
  int64_t total_playouts_ GUARDED_BY(nodes_mutex_) = 0;
  int64_t total_low_nodes_ GUARDED_BY(nodes_mutex_) = 0;
  int64_t total_nn_queries_ GUARDED_BY(nodes_mutex_) = 0;
  int64_t total_batches_ GUARDED_BY(nodes_mutex_) = 0;
  // Maximum search depth = length of longest path taken in PickNodetoExtend.
  uint16_t max_depth_ GUARDED_BY(nodes_mutex_) = 0;
  // Cumulative depth of all paths taken in PickNodetoExtend.
  uint64_t cum_depth_ GUARDED_BY(nodes_mutex_) = 0;

  std::optional<std::chrono::steady_clock::time_point> nps_start_time_
      GUARDED_BY(counters_mutex_);

  std::atomic<int> pending_searchers_{0};
  std::atomic<int> backend_waiting_counter_{0};
  std::atomic<int> thread_count_{0};

  std::vector<std::pair<BackupPath, int>> shared_collisions_
      GUARDED_BY(nodes_mutex_);

  std::unique_ptr<UciResponder> uci_responder_;
  ContemptMode contempt_mode_;

  // Random number generator for Thompson Sampling, if Search class itself needs it
  // (e.g. for UCI info generation if it samples, or if root selection uses it directly)
  // SearchWorker will have its own.
  mutable std::mt19937 rng_{std::random_device{}()};


  friend class SearchWorker;
};

// Single thread worker of the search engine.
// That used to be just a function Search::Worker(), but to parallelize it
// within one thread, have to split into stages.
class SearchWorker {
 public:
  SearchWorker(Search* search, const SearchParams& params, int id)
      : search_(search),
        history_(search_->played_history_),
        params_(params),
        moves_left_support_(search_->network_->GetCapabilities().moves_left !=
                            pblczero::NetworkFormat::MOVES_LEFT_NONE),
        rng_(std::random_device{}() // Initialize RNG for this worker
             // Potentially seed with id for better diversity: std::random_device{}(), id
            ) {
    search_->network_->InitThread(id);
    for (int i = 0; i < params.GetTaskWorkersPerSearchWorker(); i++) {
      task_workspaces_.emplace_back();
      // Pass unique seed to task worker RNGs if they also need one.
      // For now, tasks use the SearchWorker's RNG if selection happens there.
      task_threads_.emplace_back([this, i]() { this->RunTasks(i); });
    }
  }

  ~SearchWorker() {
    {
      task_count_.store(-1, std::memory_order_release);
      Mutex::Lock lock(picking_tasks_mutex_);
      exiting_ = true;
      task_added_.notify_all();
    }
    for (size_t i = 0; i < task_threads_.size(); i++) {
      task_threads_[i].join();
    }
  }

  // Runs iterations while needed.
  void RunBlocking() {
    LOGFILE << "Started search thread.";
    try {
      // A very early stop may arrive before this point, so the test is at the
      // end to ensure at least one iteration runs before exiting.
      do {
        ExecuteOneIteration();
      } while (search_->IsSearchActive());
    } catch (std::exception& e) {
      std::cerr << "Unhandled exception in worker thread: " << e.what()
                << std::endl;
      abort();
    }
  }

  // Does one full iteration of MCTS search:
  // 1. Initialize internal structures.
  // 2. Gather minibatch.
  // 3.
  // 4. Run NN computation.
  // 5. Retrieve NN computations (and terminal values) into nodes.
  // 6. Propagate the new nodes' information to all their parents in the tree.
  // 7. Update the Search's status and progress information.
  void ExecuteOneIteration();

  // The same operations one by one:
  // 1. Initialize internal structures.
  // @computation is the computation to use on this iteration.
  void InitializeIteration(std::unique_ptr<NetworkComputation> computation);

  // 2. Gather minibatch.
  void GatherMinibatch();

  // 2b. Copy collisions into shared_collisions_.
  void CollectCollisions();

  // 4. Run NN computation.
  void RunNNComputation();

  // 5. Retrieve NN computations (and terminal values) into nodes.
  void FetchMinibatchResults();

  // 6. Propagate the new nodes' information to all their parents in the tree.
  void DoBackupUpdate();

  // 7. Update the Search's status and progress information.
  void UpdateCounters();

 private:
  struct NodeToProcess {
    bool IsExtendable() const {
      return !is_collision && !node->IsTerminal() && !node->GetLowNode();
    }
    bool IsCollision() const { return is_collision; }
    bool CanEvalOutOfOrder() const {
      return is_tt_hit || is_cache_hit || node->IsTerminal() ||
             node->GetLowNode();
    }
    bool ShouldAddToInput() const {
      return nn_queried && !is_tt_hit && !is_twin_hit;
    }
    int GetRule50Ply() const { return history.Last().GetRule50Ply(); }

    // The path to the node to extend.
    BackupPath path;
    // The node to extend.
    Node* node;
    uint32_t multivisit = 0;
    // If greater than multivisit, and other parameters don't imply a lower
    // limit, multivist could be increased to this value without additional
    // change in outcome of next selection.
    uint32_t maxvisit = 0;
    float error = 0.0f;
    bool nn_queried = false;
    bool is_tt_hit = false;
    bool is_twin_hit = false;

    bool is_cache_hit = false;
    bool is_collision = false;

    // values for improving r50 estimates, filled in as we go
    float twin_error;

    // Details that are filled in as we go.
    uint64_t hash;
    uint64_t ch_hash;

    LowNode* tt_low_node;
    LowNode* twin_low_node;

    NNCacheLock lock;
    PositionHistory history;
    bool ooo_completed = false;

    // Repetition draws.
    int repetitions = 0;

    static NodeToProcess Collision(const BackupPath& path, int collision_count,
                                   int max_count) {
      return NodeToProcess(path, collision_count, max_count);
    }
    static NodeToProcess Visit(const BackupPath& path,
                               const PositionHistory& history) {
      return NodeToProcess(path, history);
    }

    void SetR50Bounds(NodeTree* dag) {}

    // Method to allow NodeToProcess to conform as a 'Computation'. Only safe
    // to call if is_cache_hit is true in the multigather path.
    std::shared_ptr<NNEval> GetNNEval(int) const { return lock->eval; }

    std::string DebugString() const {
      std::ostringstream oss;
      oss << "<NodeToProcess> This:" << this << " Depth:" << path.size()
          << " Node:" << node << " Multivisit:" << multivisit
          << " Maxvisit:" << maxvisit << " NNQueried:" << nn_queried
          << " TTHit:" << is_tt_hit << " CacheHit:" << is_cache_hit
          << " Collision:" << is_collision << " OOO:" << ooo_completed
          << " Repetitions:" << repetitions << " Path:";
      for (auto it = path.cbegin(); it != path.cend(); ++it) {
        if (it != path.cbegin()) oss << "->";
        auto n = std::get<0>(*it);
        auto nl = n->GetLowNode();
        oss << n << ":" << n->GetNInFlight();
        if (nl) {
          oss << "(" << nl << ")";
        }
      }
      oss << " --- " << std::get<0>(path.back())->DebugString(); // Assumes Node::DebugString exists
      if (node->GetLowNode())
        oss << " --- " << node->GetLowNode()->DebugString();

      return oss.str();
    }

   private:
    NodeToProcess(const BackupPath& path, uint32_t multivisit,
                  uint32_t max_count)
        : path(path),
          node(std::get<0>(path.back())),
          multivisit(multivisit),
          maxvisit(max_count),
          is_collision(true),
          history(search_->played_history_), // Initialize history properly
          repetitions(0) {}
    NodeToProcess(const BackupPath& path, const PositionHistory& in_history)
        : path(path),
          node(std::get<0>(path.back())),
          multivisit(1),
          maxvisit(0),
          is_collision(false),
          history(in_history),
          repetitions(std::get<1>(path.back())) {}
  };

  // Holds per task worker scratch data
  struct TaskWorkspace {
    // Node::Iterator in original was ArrayView<EdgeAndNode>::iterator
    // If EdgeList contains Edge directly, this needs to be EdgeList::iterator
    // Assuming EdgeAndNode is still the iterated type from node->Edges()
    std::array<Node::EdgeIterator, 256> cur_iters; // Changed to Node::EdgeIterator
    std::vector<std::unique_ptr<std::array<int, 256>>> vtp_buffer;
    std::vector<std::unique_ptr<std::array<int, 256>>> visits_to_perform;
    std::vector<int> vtp_last_filled;
    std::vector<int> current_path;
    BackupPath full_path;
    TaskWorkspace() {
      vtp_buffer.reserve(30);
      visits_to_perform.reserve(30);
      vtp_last_filled.reserve(30);
      current_path.reserve(30);
      full_path.reserve(30);
    }
  };

  struct PickTask {
    enum PickTaskType { kGathering, kProcessing };
    PickTaskType task_type;

    // For task type gathering.
    BackupPath start_path;
    Node* start_node; // Changed from start to avoid conflict
    int collision_limit;
    PositionHistory history;
    std::vector<NodeToProcess> results;

    // Task type post gather processing.
    int start_idx;
    int end_idx;

    bool complete = false;

    PickTask(const BackupPath& sp, const PositionHistory& in_history,
             int cl) // sp for start_path, cl for collision_limit
        : task_type(kGathering),
          start_path(sp),
          start_node(std::get<0>(sp.back())),
          collision_limit(cl),
          history(in_history) {}
    PickTask(int si, int ei) // si for start_idx, ei for end_idx
        : task_type(kProcessing), start_idx(si), end_idx(ei) {}
  };

  // NodeToProcess PickNodeToExtend(int collision_limit); // This was original single path picker
  // Adjust parameters for updating node @n and its parent low node if node is
  // terminal or its child low node is a transposition. Also update bounds and
  // terminal status of node @n using information from its child low node.
  // Return true if adjustment happened.
  bool MaybeAdjustForTerminalOrTransposition(
      Node* n, const LowNode* nl, float& v, float& d, float& m, float& vs,
      uint32_t& n_to_fix, float& weight_to_fix, float& v_delta, float& d_delta,
      float& m_delta, float& vs_delta, bool& update_parent_bounds) const;
  void DoBackupUpdateSingleNode(const NodeToProcess& node_to_process) REQUIRES(search_->nodes_mutex_);
  // Returns whether a node's bounds were set based on its children.
  bool MaybeSetBounds(Node* p, float m, uint32_t* n_to_fix,
                      float* weight_to_fix, float* v_delta, float* d_delta,
                      float* m_delta, float* vs_delta) const REQUIRES(search_->nodes_mutex_);
  void PickNodesToExtend(int collision_limit); // Main gather orchestrator
  void PickNodesToExtendTask(const BackupPath& path, int collision_limit,
                             PositionHistory& history,
                             std::vector<NodeToProcess>* receiver,
                             TaskWorkspace* workspace) NO_THREAD_SAFETY_ANALYSIS; // Worker fn

  // Check if the situation described by @depth under root and @position is a
  // safe two-fold or a draw by repetition and return the number of safe
  // repetitions and moves_left.
  std::pair<int, int> GetRepetitions(int depth, const Position& position);
  // Check if there is a reason to stop picking and pick @node.
  bool ShouldStopPickingHere(Node* node, bool is_root_node, int repetitions) REQUIRES_SHARED(search_->nodes_mutex_);
  void ProcessPickedTask(int batch_start, int batch_end);
  void ExtendNode(NodeToProcess& picked_node) REQUIRES(search_->nodes_mutex_);
  template <typename Computation>
  void FetchSingleNodeResult(NodeToProcess* node_to_process,
                             const Computation& computation,
                             int idx_in_computation) REQUIRES(search_->nodes_mutex_);
  void RunTasks(int tid);
  void ResetTasks();
  // Returns how many tasks there were.
  int WaitForTasks();

  Search* const search_;
  // List of nodes to process.
  std::vector<NodeToProcess> minibatch_;
  std::unique_ptr<CachingComputation> computation_;
  // History is reset and extended by PickNodeToExtend().
  PositionHistory history_;
  uint32_t number_out_of_order_ = 0;
  const SearchParams& params_;
  // std::unique_ptr<Node> precached_node_; // precached_node_ not used in search (13).cc
  const bool moves_left_support_;
  IterationStats iteration_stats_;
  StoppersHints latest_time_manager_hints_;

  // Multigather task related fields.

  Mutex picking_tasks_mutex_;
  std::vector<PickTask> picking_tasks_ GUARDED_BY(picking_tasks_mutex_);
  std::atomic<int> task_count_ = -1;
  std::atomic<int> task_taking_started_ = 0;
  std::atomic<int> tasks_taken_ = 0;
  std::atomic<int> completed_tasks_ = 0;
  std::condition_variable task_added_ GUARDED_BY(picking_tasks_mutex_);
  std::vector<std::thread> task_threads_;
  std::vector<TaskWorkspace> task_workspaces_;
  TaskWorkspace main_workspace_;
  bool exiting_ = false;

  std::mt19937 rng_; // Random number generator for this worker
};

}  // namespace lczero
