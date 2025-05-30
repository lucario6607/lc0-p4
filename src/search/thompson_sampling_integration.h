#pragma once // Include guard

// Forward declaration for OptionsDict to avoid full include if not necessary for this header.
// However, "utils/optionsdict.h" is small and commonly included.
// For robustness, let's include it directly.
#include "utils/optionsdict.h" // For lczero::OptionsDict

// Forward declarations for other types used by functions in this header if any were needed.
// For example, if MakeSearch was declared here:
// namespace lczero {
// class NodeTree; class Network; class NNCache; class SyzygyTablebase;
// struct BestMoveInfo; struct ThinkingInfo; struct SearchLimits;
// class Search; // For std::unique_ptr<Search>
// template<typename T> class std::unique_ptr; // For std::unique_ptr
// template<typename T> class std::vector; // For std::vector in ThinkingInfo
// template<typename T> class std::function; // For std::function in callbacks
// }

namespace lczero {

// Declarations for functions defined in thompson_sampling_integration.cpp
// that might be called from elsewhere (though typically only MakeSearch is).

// void AddThompsonSamplingOptions(OptionsDict* options); // Already in .cpp, might not need to be public
// bool ValidateThompsonSamplingOptions(const OptionsDict& options); // Already in .cpp, might not need to be public
// void PrintThompsonSamplingInfo(); // Already in .cpp, might not need to be public

// This is the critical declaration needed by engine.cc
void RegisterThompsonSamplingWithEngine();

// The main search factory declaration, if it's intended to be callable from outside thompson_sampling_integration.cpp
// (e.g. if src/search/search.cc calls this specific MakeSearch).
// The original plan had src/search/search.cc include this file.
// std::unique_ptr<Search> MakeSearch(
//     NodeTree& tree,
//     Network* network,
//     std::function<void(const BestMoveInfo&)> best_move_callback,
//     std::function<void(const std::vector<ThinkingInfo>&)> info_callback,
//     const OptionsDict& options,
//     NNCache* cache,
//     SyzygyTablebase* syzygy_tb);
// For now, let's assume only RegisterThompsonSamplingWithEngine needs to be in this header for engine.cc

} // namespace lczero
