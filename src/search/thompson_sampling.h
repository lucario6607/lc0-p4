#ifndef LCZERO_THOMPSON_SAMPLING_H_
#define LCZERO_THOMPSON_SAMPLING_H_

#include <random>
#include <vector>
#include <memory>
#include <mutex>
#include <atomic>
#include <chrono> // Added for std::chrono
#include <limits> // Added for std::numeric_limits
#include <string> // Added for std::string
#include <cmath>  // Added for std::sqrt, std::log
#include <functional> // Added for std::function
#include "mcts/search.h"
#include "mcts/node.h"
#include "mcts/params.h"
#include "neural/cache.h"
#include "utils/optionsdict.h"
#include "chess/callbacks.h" // For lczero::BestMoveInfo, lczero::ThinkingInfo

namespace lczero {

// Thompson Sampling parameters
struct ThompsonSamplingParams {
  // Prior parameters for Beta distribution
  float alpha_prior = 1.0f;
  float beta_prior = 1.0f;
  
  // Exploration parameters
  float exploration_factor = 1.0f;
  bool use_value_head = true;
  bool use_policy_head = true;
  
  // Sample count for Thompson sampling
  int sample_count = 100;
  
  // UCB fallback parameters
  float ucb_factor = 1.414f;
  bool enable_ucb_fallback = true;
  int min_visits_for_ts = 10;
};

class ThompsonSamplingSearch : public Search {
 public:
  ThompsonSamplingSearch(lczero::NodeTree& tree, 
                        lczero::Network* network,
                        std::function<void(const lczero::BestMoveInfo&)> best_move_callback,
                        std::function<void(const std::vector<lczero::ThinkingInfo>&)> info_callback,
                        const lczero::OptionsDict& options,
                        lczero::NNCache* cache,
                        lczero::SyzygyTablebase* syzygy_tb);

  ~ThompsonSamplingSearch() = default; // Removed override for diagnostics

  // Main search methods
  void StartThreads(size_t how_many); // Removed override
  void RunBlocking(); // Removed override
  void Stop(); // Removed override
  void Abort(); // Removed override
  void Wait(); // Removed override

  // Get search statistics
  const std::vector<std::string>& GetVerboseStats() const; // Removed override
  
 private:
  // Core Thompson Sampling implementation
  void Worker();
  void SendUciInfo();
  
  // Thompson Sampling specific methods
  Node* SelectChildThompsonSampling(Node* node);
  float SampleFromBeta(float alpha, float beta);
  float CalculateThompsonValue(Node* child);
  
  // Fallback to UCB when insufficient data
  Node* SelectChildUCB(Node* node);
  float CalculateUCB(Node* parent, Node* child);
  
  // Node evaluation and expansion
  void ExpandNode(Node* node);
  void BackupValue(Node* node, float value);
  
  // Search tree navigation
  std::vector<Node*> GetPath(Node* node);
  bool ShouldStop() const;
  
  // Thread management
  std::vector<std::thread> threads_;
  std::atomic<bool> stop_flag_{false};
  std::atomic<bool> abort_flag_{false};
  std::mutex search_mutex_;
  
  // Search parameters
  ThompsonSamplingParams params_;
  
  // Random number generation
  thread_local static std::mt19937 rng_;
  // thread_local static std::beta_distribution<float> beta_dist_; // Commented for diagnostics
  
  // Search statistics
  std::atomic<uint64_t> nodes_searched_{0};
  std::atomic<uint64_t> tb_hits_{0};
  std::chrono::steady_clock::time_point start_time_;
  
  // Verbose statistics
  mutable std::vector<std::string> verbose_stats_;
  
  // Network and cache
  Network* network_;
  NNCache* cache_;
  SyzygyTablebase* syzygy_tb_;
  
  // Search limits and callbacks (limits_ removed)
  std::function<void(const lczero::BestMoveInfo&)> best_move_callback_;
  std::function<void(const std::vector<lczero::ThinkingInfo>&)> info_callback_;
  
  // Node tree
  lczero::NodeTree& tree_; // Changed to non-const ref
};

// Thread-local random number generator initialization
thread_local std::mt19937 ThompsonSamplingSearch::rng_(std::random_device{}());
// thread_local std::beta_distribution<float> ThompsonSamplingSearch::beta_dist_; // Commented for diagnostics

ThompsonSamplingSearch::ThompsonSamplingSearch(
    lczero::NodeTree& tree, 
    lczero::Network* network,
    std::function<void(const lczero::BestMoveInfo&)> best_move_callback,
    std::function<void(const std::vector<lczero::ThinkingInfo>&)> info_callback,
    const lczero::OptionsDict& options,
    lczero::NNCache* cache,
    lczero::SyzygyTablebase* syzygy_tb)
    : Search(&tree, // Pass NodeTree*
             network,
             std::make_unique<lczero::CallbackUciResponder>(best_move_callback, info_callback),
             lczero::MoveList{}, // Empty searchmoves for now
             std::chrono::steady_clock::now(), // start_time for base
             nullptr, // SearchStopper - passing null for now
             options.GetOrDefault<bool>("infinite", false),
             options.GetOrDefault<bool>("ponder", false),
             options,
             cache,
             syzygy_tb),
      tree_(tree), // Initialize own tree_ member
      best_move_callback_(best_move_callback),
      info_callback_(info_callback),
      network_(network), // Initialize own network_ member 
      cache_(cache),       // Initialize own cache_ member 
      syzygy_tb_(syzygy_tb), // Initialize own syzygy_tb_ member
      start_time_(std::chrono::steady_clock::now()) { // Initialize own start_time_
  
  // Initialize Thompson Sampling parameters from options
  params_.alpha_prior = options.GetOrDefault<float>("ts-alpha-prior", 1.0f);
  params_.beta_prior = options.GetOrDefault<float>("ts-beta-prior", 1.0f);
  params_.exploration_factor = options.GetOrDefault<float>("ts-exploration", 1.0f);
  params_.sample_count = options.GetOrDefault<int>("ts-sample-count", 100);
  params_.use_value_head = options.GetOrDefault<bool>("ts-use-value", true);
  params_.use_policy_head = options.GetOrDefault<bool>("ts-use-policy", true);
  params_.ucb_factor = options.GetOrDefault<float>("ts-ucb-factor", 1.414f);
  params_.enable_ucb_fallback = options.GetOrDefault<bool>("ts-ucb-fallback", true);
  params_.min_visits_for_ts = options.GetOrDefault<int>("ts-min-visits", 10);
}

void ThompsonSamplingSearch::StartThreads(size_t how_many) {
  stop_flag_ = false;
  abort_flag_ = false;
  
  for (size_t i = 0; i < how_many; ++i) {
    threads_.emplace_back([this]() { Worker(); });
  }
}

void ThompsonSamplingSearch::RunBlocking() {
  if (threads_.empty()) {
    Worker();
  } else {
    Wait();
  }
}

void ThompsonSamplingSearch::Stop() {
  stop_flag_ = true;
}

void ThompsonSamplingSearch::Abort() {
  abort_flag_ = true;
  stop_flag_ = true;
}

void ThompsonSamplingSearch::Wait() {
  for (auto& thread : threads_) {
    if (thread.joinable()) {
      thread.join();
    }
  }
  threads_.clear();
}

void ThompsonSamplingSearch::Worker() {
  while (!ShouldStop()) {
    Node* root = tree_.GetCurrentHead();
    if (!root) break;
    
    // Select path using Thompson Sampling
    std::vector<Node*> path;
    Node* current = root;
    
    // Navigate down the tree
    while (current && !current->IsTerminal() && !ShouldStop()) {
      path.push_back(current);
      
      if (!current->HasChildren()) {
        // Expand node if not expanded
        ExpandNode(current);
        if (!current->HasChildren()) break;
      }
      
      // Select child using Thompson Sampling or UCB fallback
      Node* selected_child = nullptr;
      if (current->GetN() >= params_.min_visits_for_ts && params_.enable_ucb_fallback) {
        selected_child = SelectChildThompsonSampling(current);
      } else {
        selected_child = SelectChildUCB(current);
      }
      
      if (!selected_child) break;
      current = selected_child;
    }
    
    if (current && !ShouldStop()) {
      path.push_back(current);
      
      // Evaluate position
      float value = 0.0f;
      if (current->IsTerminal()) {
        value = current->GetWL();
      } else {
        // Get neural network evaluation
        // This would need to be implemented based on Lc0's NN interface
        // For now, using a placeholder
        value = 0.0f; // Placeholder for NN evaluation
      }
      
      // Backup the value
      BackupValue(current, value);
      
      // Update node count
      nodes_searched_++;
      
      // Send UCI info periodically
      if (nodes_searched_ % 1000 == 0) {
        SendUciInfo();
      }
    }
  }
}

Node* ThompsonSamplingSearch::SelectChildThompsonSampling(Node* node) {
  if (!node->HasChildren()) return nullptr;
  
  Node* best_child = nullptr;
  float best_sample = -std::numeric_limits<float>::infinity();

  for (lczero::Node::Iterator it = node->Edges().begin(); it != node->Edges().end(); ++it) {
    lczero::Node* child = it.node(); // Changed it->node() to it.node()
    if (!child) {
        continue; 
    }
    float sampled_value = CalculateThompsonValue(child);
    
    if (sampled_value > best_sample) {
      best_sample = sampled_value;
      best_child = child;
    }
  }
  
  return best_child;
}

float ThompsonSamplingSearch::CalculateThompsonValue(Node* child) {
  uint32_t visits = child->GetN();
  if (visits == 0) {
    // For unvisited nodes, sample from prior
    return SampleFromBeta(params_.alpha_prior, params_.beta_prior);
  }
  
  // Calculate wins and losses for Beta distribution
  float wins = (child->GetWL() + 1.0f) / 2.0f * visits; // Convert from [-1,1] to [0,1]
  float losses = visits - wins;
  
  // Add prior
  float alpha = wins + params_.alpha_prior;
  float beta = losses + params_.beta_prior;
  
  // Sample from Beta distribution
  float sampled_winrate = SampleFromBeta(alpha, beta);
  
  // Apply exploration factor
  return sampled_winrate * params_.exploration_factor;
}

float ThompsonSamplingSearch::SampleFromBeta(float alpha, float beta) {
  if (alpha <= 0.0f || beta <= 0.0f) {
    return 0.5f; // Fallback for invalid parameters
  }
  
  // Use thread-local beta distribution
  // beta_dist_.param(std::beta_distribution<float>::param_type(alpha, beta)); // Commented for diagnostics
  // return beta_dist_(rng_); // Commented for diagnostics
  return 0.5f; // Temporary fallback
}

Node* ThompsonSamplingSearch::SelectChildUCB(Node* node) {
  if (!node->HasChildren()) return nullptr;
  
  Node* best_child = nullptr;
  float best_ucb = -std::numeric_limits<float>::infinity();
  
  uint32_t parent_visits = node->GetN();
  if (parent_visits == 0) parent_visits = 1; // Avoid log(0)

  for (lczero::Node::Iterator it = node->Edges().begin(); it != node->Edges().end(); ++it) {
    lczero::Node* child = it.node(); // Changed it->node() to it.node()
    if (!child) {
        continue;
    }
    float ucb_value = CalculateUCB(node, child);
    
    if (ucb_value > best_ucb) {
      best_ucb = ucb_value;
      best_child = child;
    }
  }
  
  return best_child;
}

float ThompsonSamplingSearch::CalculateUCB(Node* parent, Node* child) {
  uint32_t child_visits = child->GetN();
  uint32_t parent_visits = parent->GetN();
  
  if (child_visits == 0) {
    return std::numeric_limits<float>::infinity(); // Prioritize unvisited nodes
  }
  
  float exploitation = child->GetQ(Search::GetParams().GetDrawScore());
  float exploration = params_.ucb_factor * 
                     std::sqrt(std::log(static_cast<float>(parent_visits)) / child_visits); // Added static_cast for log
  
  return exploitation + exploration;
}

void ThompsonSamplingSearch::ExpandNode(Node* node) {
  // This would need to interface with Lc0's node expansion logic
  // Placeholder implementation
  std::lock_guard<std::mutex> lock(search_mutex_);
  // Actual expansion would happen here using Lc0's mechanisms
}

void ThompsonSamplingSearch::BackupValue(Node* node, float value) {
  // This would need to interface with Lc0's backup logic
  // Placeholder implementation
  std::lock_guard<std::mutex> lock(search_mutex_);
  // Actual backup would happen here using Lc0's mechanisms
}

bool ThompsonSamplingSearch::ShouldStop() const {
  if (abort_flag_.load() || stop_flag_.load()) return true; // Use .load() for atomics
  // // Check time limits (Original code referring to non-existent limits_)
  // if (limits_.time_ms > 0) {
  //   auto elapsed = std::chrono::steady_clock::now() - start_time_;
  //   if (std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() 
  //       >= limits_.time_ms) {
  //     return true;
  //   }
  // }
  // // Check node limits (Original code referring to non-existent limits_)
  // if (limits_.nodes > 0 && nodes_searched_ >= limits_.nodes) {
  //   return true;
  // }
  
  // TODO: Re-implement ShouldStop. It should likely consult the SearchStopper 
  // from the base class or use Search::IsSearchActive().
  // For now, rely on abort_flag_ and stop_flag_ which are controlled externally.
  // Or, could check against options directly if they define limits.
  // This needs to align with how the base Search class expects stopping to be managed.
  // A simple approach for now:
  // return Search::IsSearchActive(); // If base class manages this well.
  // However, Search::IsSearchActive() is not const.
  // The base Search::stop_ is atomic and can be checked.
  // The base class Search itself has a stopper_ member.
  // For now, only abort_flag_ and stop_flag_ (ThompsonSamplingSearch's own flags).
  return false; // Placeholder - this makes it never stop on its own based on time/nodes.
}

void ThompsonSamplingSearch::SendUciInfo() {
  if (!info_callback_) return;
  
  auto elapsed = std::chrono::steady_clock::now() - start_time_;
  auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
  
  // Create a vector for ThinkingInfo as expected by ThinkingCallback
  std::vector<lczero::ThinkingInfo> infos;
  lczero::ThinkingInfo info; // Use lczero::ThinkingInfo
  info.depth = 0; 
  info.seldepth = 0; 
  info.time = elapsed_ms;
  info.nodes = nodes_searched_.load();
  info.nps = elapsed_ms > 0 ? (nodes_searched_.load() * 1000) / elapsed_ms : 0;
  info.tb_hits = tb_hits_.load();
  
  // Add Thompson Sampling specific info
  info.comment = "Thompson Sampling";
  
  infos.push_back(info);
  info_callback_(infos); // Pass vector of ThinkingInfo
}

const std::vector<std::string>& ThompsonSamplingSearch::GetVerboseStats() const {
  verbose_stats_.clear();
  verbose_stats_.push_back("Thompson Sampling Search Statistics:");
  verbose_stats_.push_back(std::string("Nodes searched: ") + std::to_string(nodes_searched_.load()));
  verbose_stats_.push_back(std::string("Alpha prior: ") + std::to_string(params_.alpha_prior));
  verbose_stats_.push_back(std::string("Beta prior: ") + std::to_string(params_.beta_prior));
  verbose_stats_.push_back(std::string("Exploration factor: ") + std::to_string(params_.exploration_factor));
  verbose_stats_.push_back(std::string("Sample count: ") + std::to_string(params_.sample_count));
  verbose_stats_.push_back(std::string("UCB fallback enabled: ") + 
                          (params_.enable_ucb_fallback ? "true" : "false"));
  verbose_stats_.push_back(std::string("Min visits for TS: ") + std::to_string(params_.min_visits_for_ts));
  
  return verbose_stats_;
}

// Factory function to create Thompson Sampling search
// Note: SearchLimits removed from parameters.
// Callbacks changed to explicit std::function types.
// NodeTree changed to lczero::NodeTree&.
std::unique_ptr<Search> MakeThompsonSamplingSearch(
    lczero::NodeTree& tree, 
    lczero::Network* network,
    std::function<void(const lczero::BestMoveInfo&)> best_move_callback,
    std::function<void(const std::vector<lczero::ThinkingInfo>&)> info_callback,
    const lczero::OptionsDict& options,
    lczero::NNCache* cache,
    lczero::SyzygyTablebase* syzygy_tb) {
  
  return std::make_unique<ThompsonSamplingSearch>(
      tree, network, best_move_callback, info_callback,
      options, cache, syzygy_tb);
}

} // namespace lczero

#endif // LCZERO_THOMPSON_SAMPLING_H_
