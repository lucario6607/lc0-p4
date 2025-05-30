# CMakeLists.txt for lczero search component

# Example: A minimal CMakeLists.txt if one doesn't exist.
# This will likely need to be adapted to Lc0's actual build system structure.

cmake_minimum_required(VERSION 3.10) # Or appropriate Lc0 version

project(lczero-search)

# Assuming lczero-search is a library
add_library(lczero-search)

target_sources(lczero-search PRIVATE
  thompson_sampling.h
  thompson_sampling_integration.cpp
  # Add other .cpp files from src/search that are part of this target
  # For example, if search.cpp, search_factory.cpp etc. belong here:
  # search.cpp
  # search_factory.cpp 
  # ...
)

# Add include directories if necessary, e.g., if headers are in parent dirs
# target_include_directories(lczero-search PUBLIC ../) # Example

# Link against other Lc0 libraries if needed
# target_link_libraries(lczero-search PRIVATE lczero-utils lczero-mcts ...) # Example
