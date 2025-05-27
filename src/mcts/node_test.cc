#include "gtest/gtest.h"
#include "mcts/node.h"
// For SearchParams to set options if needed by Node static members directly.
// SearchParams includes optionsdict.h and optionsparser.h indirectly.
#include "mcts/search_params.h" 
#include "chess/move.h"      // For lczero::Move
#include "utils/optionsdict.h" // For OptionsDict to initialize SearchParams

#include <vector>
#include <iostream>
#include <random>
#include <chrono>
#include <cmath> // For std::sqrt
#include <iomanip> // For std::fixed and std::setprecision

// Assuming Node class and related classes are in the lczero namespace
using namespace lczero;

// Minimal OptionsDict for SearchParams construction if default options are not sufficient
// or if specific TS related params from SearchParams are needed by Node static members.
// Node's static members for TS are set directly in tests.

TEST(NodeTest, ThompsonSamplingBasic) {
    // Set Thompson Sampling related static members in Node class
    Node::SetThompsonSampling(true);
    Node::SetPolicyTemperature(1.0); // Default policy temperature for TS

    // Test a single node's Beta distribution update and sampling
    Node node_for_test(0); // Create a node with index 0

    // Default Beta parameters are alpha=1.0, beta=1.0 (from std::atomic default member init)
    // So, initial mean should be 1.0 / (1.0 + 1.0) = 0.5
    double initial_mean = node_for_test.GetBetaMean();
    ASSERT_NEAR(initial_mean, 0.5, 1e-5);

    // Simulate some game results.
    // UpdateBeta expects a value between 0.0 (loss for current player) and 1.0 (win for current player).
    
    // Initial: alpha=1, beta=1
    node_for_test.UpdateBeta(1.0); // Win: prob=1. alpha -> 1+1=2, beta -> 1+(1-1)=1
    node_for_test.UpdateBeta(1.0); // Win: prob=1. alpha -> 2+1=3, beta -> 1+(1-1)=1
    node_for_test.UpdateBeta(0.0); // Loss: prob=0. alpha -> 3+0=3, beta -> 1+(1-0)=2
                                   // Expected mean = alpha / (alpha+beta) = 3 / (3+2) = 0.6

    double new_mean = node_for_test.GetBetaMean();
    ASSERT_NE(new_mean, initial_mean); 
    ASSERT_NEAR(new_mean, 0.6, 1e-5);

    // Test SampleBeta - it's random, so check many samples or bounds
    // The mean of many samples should be close to the distribution's mean.
    double s_sum = 0;
    int count = 100000; // Increased count for better accuracy for sample mean
    std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count()); 

    for (int i = 0; i < count; ++i) {
       s_sum += node_for_test.SampleBeta(rng); 
    }
    // Check if the sample mean is close to the theoretical mean of the Beta distribution.
    // For a Beta(alpha, beta) distribution, the mean is alpha / (alpha + beta).
    // Here, alpha=3, beta=2, so mean = 3 / (3+2) = 0.6.
    ASSERT_NEAR(s_sum / count, 0.6, 0.01); // Increased tolerance slightly for sampling variance
}


TEST(NodeTest, BenchmarkSelectionSpeed) {
    lczero::Node test_node_bench(0); 
    // Initialize with some history to make Beta distribution non-trivial
    test_node_bench.UpdateBeta(1.0); // alpha=2,beta=1
    test_node_bench.UpdateBeta(0.0); // alpha=2,beta=2
    test_node_bench.UpdateBeta(1.0); // alpha=3,beta=2
    test_node_bench.UpdateBeta(1.0); // alpha=4,beta=2. Mean = 4/(4+2) = 0.666...

    const int NUM_SAMPLES = 1000000;
    volatile double ts_sample_sink = 0; 
    std::mt19937 rng(12345); // Fixed seed for benchmark consistency

    auto start_ts = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_SAMPLES; ++i) {
        ts_sample_sink = test_node_bench.SampleBeta(rng);
    }
    auto end_ts = std::chrono::high_resolution_clock::now();
    auto thompson_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_ts - start_ts).count();

    // Simplified PUCT-like calculation: Q + C * P * sqrt(N_parent) / (1+N_child)
    float Q_child = 0.6f;    
    float P_child = 0.05f;   
    int N_parent = 50;       
    int N_child = 5;         
    float C_puct = 1.25f;    
    volatile double puct_score_sink = 0; 

    auto start_puct = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_SAMPLES; ++i) {
        double U_term = C_puct * P_child * std::sqrt(static_cast<double>(N_parent)) / (1.0 + N_child);
        puct_score_sink = Q_child + U_term;
    }
    auto end_puct = std::chrono::high_resolution_clock::now();
    auto puct_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_puct - start_puct).count();

    // Using Google Test's logging for benchmark results is preferred over std::cout directly
    // as it integrates better with test runners and CI systems.
    RecordProperty("ThompsonSamplingTime_ns", std::to_string(thompson_time_ns));
    RecordProperty("PUCTLikeCalcTime_ns", std::to_string(puct_time_ns));
    RecordProperty("LastThompsonSample", std::to_string(ts_sample_sink));
    RecordProperty("LastPUCTScore", std::to_string(puct_score_sink));
    
    // Print to stdout as well for immediate visibility during local test runs
    std::cout << std::fixed << std::setprecision(2); // For better readability of nanoseconds
    std::cout << "[ BENCHMARK ] Thompson Sampling (1M samples): " << static_cast<double>(thompson_time_ns) / 1e6 << " ms. Last sample (sink): " << ts_sample_sink << std::endl;
    std::cout << "[ BENCHMARK ] PUCT-like calc  (1M samples): " << static_cast<double>(puct_time_ns) / 1e6 << " ms. Last score (sink): " << puct_score_sink << std::endl;
    
    ASSERT_GT(thompson_time_ns, 0);
    ASSERT_GT(puct_time_ns, 0);
}

/*
TODO: Add this file to the build system (e.g., CMakeLists.txt).
Example for CMake:

if (BUILD_TESTING)
    find_package(GTest REQUIRED)
    add_executable(lc0_tests
        src/tests/main.cc      # Assuming a common main for tests
        # src/tests/common_test.cc # Assuming common test utilities (if any)
        # ... other existing test files ...
        src/mcts/node_test.cc
    )
    # Ensure lc0_lib (or equivalent target for core lc0 library) is linked
    # Also link GTest::GTest and GTest::Main (or gtest_main if GTest::Main is not found)
    # Common libraries for lc0 tests might include:
    # lc0_lib (core lc0 logic), chess (chess specific types), utils (general utilities)
    target_link_libraries(lc0_tests PRIVATE lc0_lib chess utils GTest::GTest GTest::Main)
    
    # Add to CTest
    include(GoogleTest)
    gtest_discover_tests(lc0_tests)
endif()

Note: The exact target names (lc0_lib, chess, utils, GTest::GTest, GTest::Main) and structure 
might vary based on the project's CMake setup. `gtest_discover_tests` is the modern way
to add gtests to CTest. If using an older CMake, `add_test` might be used directly.
The `chess` and `utils` components are assumed to be part of `lc0_lib` or similarly linked if not separate.
Make sure headers from `mcts`, `chess`, `utils` are in the include path for compiling this test.
*/
