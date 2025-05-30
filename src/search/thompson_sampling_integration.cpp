// thompson_sampling_integration.cpp
// Integration file to add Thompson Sampling to Leela Chess Zero

#include "thompson_sampling.h"
#include "engine.h"
#include "utils/optionsdict.h"
#include "search/search.h" // Required for MakeDefaultSearch declaration

// Add this include if MakeDefaultSearch is not found by the line above.
// #include "search_factory.h" 

#include <iostream> // Required for std::cout, std::endl
#include <string> // Required for std::string related operations
#include <vector> // Required for std::vector
#include <memory> // Required for std::unique_ptr
#include <cmath> // Required for std::log, std::sqrt
#include <limits> // Required for std::numeric_limits

// Forward declaration for GetEngineOptions, assuming it's defined elsewhere
// and returns a pointer to OptionsDict. If it's in a specific header,
// that header should be included.
// For example, if GetEngineOptions() is in "uci_engine.h":
// #include "uci_engine.h" 
// Or if it's part of a class/namespace:
// extern lczero::OptionsDict* GetEngineOptions(); // Adjust namespace if needed

namespace lczero {

// If GetEngineOptions() is not globally accessible, this might be a placeholder
// for how it's actually obtained, e.g., via a singleton or a global function.
// This is a common pattern, but the actual implementation detail depends on Lc0's codebase.
extern OptionsDict* GetEngineOptions(); // Ensure this declaration matches Lc0's structure

// Declaration for MakeDefaultSearch, assuming it's the standard MCTS search factory.
// This function's signature must match the one in Lc0.
// It's often found in a file like search_factory.cpp or search.cpp.
// Updated to match new callback types and removal of SearchLimits.
std::unique_ptr<Search> MakeDefaultSearch(
    NodeTree& tree, // Changed to non-const
    Network* network,
    lczero::BestMoveCallback best_move_callback, // Changed type
    lczero::ThinkingCallback info_callback,     // Changed type
    const OptionsDict& options,
    NNCache* cache,
    SyzygyTablebase* syzygy_tb);


// Add Thompson Sampling UCI options to the engine
void AddThompsonSamplingOptions(OptionsDict* options) {
  if (!options) return; // Safety check
  // Main Thompson Sampling enable option
  options->Add<BoolOption>("ThompsonSampling", false, 
                          "Enable Thompson Sampling search instead of standard MCTS");
  
  // Thompson Sampling specific parameters
  options->Add<FloatOption>("TS-AlphaPrior", 0.1f, 10.0f, 1.0f,
                           "Alpha parameter for Beta prior distribution in Thompson Sampling");
  
  options->Add<FloatOption>("TS-BetaPrior", 0.1f, 10.0f, 1.0f,
                           "Beta parameter for Beta prior distribution in Thompson Sampling");
  
  options->Add<FloatOption>("TS-Exploration", 0.1f, 5.0f, 1.0f,
                           "Exploration factor for Thompson Sampling");
  
  options->Add<IntOption>("TS-SampleCount", 1, 1000, 100,
                         "Number of samples to draw for Thompson Sampling");
  
  options->Add<BoolOption>("TS-UseValue", true,
                          "Use neural network value head in Thompson Sampling");
  
  options->Add<BoolOption>("TS-UsePolicy", true,
                          "Use neural network policy head in Thompson Sampling");
  
  // UCB fallback options
  options->Add<BoolOption>("TS-UCBFallback", true,
                          "Enable UCB fallback for nodes with insufficient visits");
  
  options->Add<FloatOption>("TS-UCBFactor", 0.1f, 5.0f, 1.414f,
                           "UCB exploration constant for fallback");
  
  options->Add<IntOption>("TS-MinVisits", 1, 1000, 10,
                         "Minimum visits required before using Thompson Sampling");
  
  // Advanced Thompson Sampling options
  options->Add<FloatOption>("TS-ValueWeight", 0.0f, 1.0f, 1.0f,
                           "Weight for value head in Thompson Sampling evaluation");
  
  options->Add<FloatOption>("TS-PolicyWeight", 0.0f, 1.0f, 0.5f,
                           "Weight for policy head in Thompson Sampling evaluation");
  
  options->Add<BoolOption>("TS-AdaptiveExploration", false,
                          "Enable adaptive exploration based on position complexity");
  
  options->Add<FloatOption>("TS-TemperatureDecay", 0.9f, 1.0f, 0.99f,
                           "Temperature decay factor for exploration over time");
  
  // Debugging and analysis options
  options->Add<BoolOption>("TS-Verbose", false,
                          "Enable verbose Thompson Sampling statistics");
  
  options->Add<IntOption>("TS-LogInterval", 100, 10000, 1000,
                         "Interval for logging Thompson Sampling statistics");
}

// Modified search factory to support Thompson Sampling
// Updated signature: NodeTree&, specific callbacks, SearchLimits removed.
std::unique_ptr<Search> MakeSearch(NodeTree& tree, // Changed to non-const
                                  Network* network,
                                  lczero::BestMoveCallback best_move_callback, // Changed type
                                  lczero::ThinkingCallback info_callback,     // Changed type
                                  const OptionsDict& options,
                                  NNCache* cache,
                                  SyzygyTablebase* syzygy_tb) {
  
  // Check if Thompson Sampling is enabled
  if (options.GetOrDefault<bool>("ThompsonSampling", false)) {
    // Validate options before creating Thompson Sampling search
    if (!ValidateThompsonSamplingOptions(options)) {
        // Optionally, log an error or throw an exception
        std::cerr << "Invalid Thompson Sampling options. Falling back to default search." << std::endl;
        // Fallback to default search if validation fails (limits removed)
        return MakeDefaultSearch(tree, network, best_move_callback,
                                 info_callback, options,
                                 cache, syzygy_tb);
    }
    // Call MakeThompsonSamplingSearch (tree is already non-const, limits removed)
    return MakeThompsonSamplingSearch(tree, network, best_move_callback,
                                     info_callback, options,
                                     cache, syzygy_tb);
  }
  
  // Otherwise, use the default search (MCTS) (limits removed)
  return MakeDefaultSearch(tree, network, best_move_callback,
                          info_callback, options,
                          cache, syzygy_tb);
}

// Engine modification to register Thompson Sampling options
void RegisterThompsonSamplingWithEngine() {
  // This function should be called during engine initialization
  // to register the Thompson Sampling options with the UCI interface
  
  auto* engine_options = GetEngineOptions(); // Get engine's options dictionary
  if (engine_options) {
    AddThompsonSamplingOptions(engine_options);
  } else {
    // Log error or handle missing engine_options appropriately
    std::cerr << "Error: Could not get engine options to register Thompson Sampling." << std::endl;
  }
}

// Helper function to validate Thompson Sampling options
bool ValidateThompsonSamplingOptions(const OptionsDict& options) {
  bool ts_enabled = options.GetOrDefault<bool>("ThompsonSampling", false);
  
  if (!ts_enabled) return true; // No validation needed if TS is disabled
  
  // Validate parameter ranges
  float alpha = options.GetOrDefault<float>("TS-AlphaPrior", 1.0f);
  float beta = options.GetOrDefault<float>("TS-BetaPrior", 1.0f);
  
  if (alpha <= 0.0f || beta <= 0.0f) {
    std::cerr << "Error: Thompson Sampling Alpha and Beta priors must be positive." << std::endl;
    return false;
  }
  
  float exploration = options.GetOrDefault<float>("TS-Exploration", 1.0f);
  if (exploration < 0.0f) {
    std::cerr << "Error: Thompson Sampling exploration factor must be non-negative." << std::endl;
    return false;
  }
  
  int sample_count = options.GetOrDefault<int>("TS-SampleCount", 100);
  if (sample_count <= 0) {
    std::cerr << "Error: Thompson Sampling sample count must be positive." << std::endl;
    return false;
  }
  
  int min_visits = options.GetOrDefault<int>("TS-MinVisits", 10);
  if (min_visits < 0) {
    std::cerr << "Error: Thompson Sampling minimum visits must be non-negative." << std::endl;
    return false;
  }
  
  return true;
}

// Thompson Sampling information for UCI
void PrintThompsonSamplingInfo() {
  std::cout << "Thompson Sampling Search for Leela Chess Zero" << std::endl;
  std::cout << "=============================================" << std::endl;
  std::cout << "Thompson Sampling is a Bayesian approach to the multi-armed bandit problem." << std::endl;
  std::cout << "It maintains probability distributions over the value of each move and" << std::endl;
  std::cout << "samples from these distributions to make exploration decisions." << std::endl;
  std::cout << std::endl;
  std::cout << "Key advantages:" << std::endl;
  std::cout << "- Natural exploration-exploitation balance" << std::endl;
  std::cout << "- Bayesian uncertainty quantification" << std::endl;
  std::cout << "- Adaptive to position complexity" << std::endl;
  std::cout << "- Theoretically optimal regret bounds" << std::endl;
  std::cout << std::endl;
  std::cout << "To enable: setoption name ThompsonSampling value true" << std::endl;
  std::cout << "For help with parameters: setoption name TS-Verbose value true" << std::endl;
}

} // namespace lczero
