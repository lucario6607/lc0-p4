#include "helper/auxiliary_helper_selector.h"
#include "mcts/node.h"      // For lczero::Node, lczero::NodeTree
#include "chess/position.h" // For lczero::Position, lczero::GetFen
#include "chess/board.h"    // For lczero::Move, MoveList

#include <iostream> // For std::cout, std::cerr (temporary logging)
#include <sstream>  // For std::ostringstream if needed for debugging

// --- Placeholder Implementations Update ---

bool AuxiliaryHelperSelector::IsNodeStillRelevant(lczero::Node* node, lczero::NodeTree* tree_context) {
    if (!node || !tree_context) return false;
    // TODO: Implement proper relevance check. This could involve:
    // 1. Checking if the node is still reachable from the current search head.
    // 2. Checking if the branch hasn't been pruned by other search heuristics.
    // 3. Comparing node's evaluation/visits against current best alternatives.
    // std::cerr << "Warning: IsNodeStillRelevant is a placeholder, returning true." << std::endl;
    return true;
}

lczero::Position AuxiliaryHelperSelector::GetNodePosition(lczero::Node* node, lczero::NodeTree* tree_context) {
    if (!node || !tree_context) {
        std::cerr << "Error: GetNodePosition called with null node or tree_context. Returning startpos." << std::endl;
        lczero::ChessBoard board;
        board.SetFromFen(lczero::ChessBoard::kStartposFen);
        return lczero::Position(board, 0, 0);
    }
    // TODO: This is a CRITICAL placeholder.
    // A proper implementation needs to reconstruct the position for the 'node'.
    // This typically involves:
    //   - Finding the path of moves from a known root position (e.g., tree_context->GetGameBeginNode() or tree_context->HeadPosition() if applicable) to 'node'.
    //   - Applying those moves to the root position.
    // This can be complex and computationally intensive if not careful.
    // For now, returning the HeadPosition of the tree_context. This is only correct if 'node' IS the head.
    std::cerr << "Warning: GetNodePosition is a placeholder and currently returns the NodeTree's HeadPosition. This is likely INCORRECT for nodes other than the head." << std::endl;
    return tree_context->HeadPosition();
}

int AuxiliaryHelperSelector::GetNodeDepth(lczero::Node* node, lczero::NodeTree* tree_context) {
    if (!node || !tree_context) return 0;
    // TODO: This is a placeholder. True node depth in the search tree is complex.
    // Using gamePly from the (placeholder) GetNodePosition as a proxy for depth from game start.
    // This might not be what's intended by "depth" in HelperRequest if it means search depth.
    lczero::Position pos = GetNodePosition(node, tree_context); // Uses the placeholder GetNodePosition
    // std::cerr << "Warning: GetNodeDepth is using gamePly from a placeholder GetNodePosition." << std::endl;
    return pos.GetGamePly();
}

int AuxiliaryHelperSelector::GetNodeVisits(lczero::Node* node) {
    if (!node) return 0;
    return node->GetN();
}

// GetNodeCreationTime was updated in the previous subtask.
std::chrono::steady_clock::time_point AuxiliaryHelperSelector::GetNodeCreationTime(lczero::Node* node) {
    if (!node) return std::chrono::steady_clock::now();
    return node->GetCreationTime();
}

bool AuxiliaryHelperSelector::IsTacticalPosition(lczero::Node* node, lczero::NodeTree* tree_context) {
    if (!node || !tree_context) return false;
    // Use the (placeholder) GetNodePosition to get the board state.
    lczero::Position pos = GetNodePosition(node, tree_context);
    const auto& board = pos.GetBoard(); // Gets board from player-to-move's perspective

    // Basic check 1: Is the current player in check?
    if (board.IsUnderCheck()) { // Corrected: Was GivesCheck(), which needs a move. IsUnderCheck() for current state.
        // std::cout << "IsTacticalPosition: Player is in check." << std::endl;
        return true;
    }

    // Basic check 2: Are there any capture moves available?
    lczero::MoveList legal_moves = board.GenerateLegalMoves();
    for (lczero::Move m : legal_moves) {
        // A capture occurs if the destination square 'm.to()' is occupied by an opponent's piece.
        if (board.theirs().get(m.to())) {
            // std::cout << "IsTacticalPosition: Capture move " << m.as_string() << " available." << std::endl;
            return true;
        }
    }
    // TODO: Add more sophisticated tactical checks (e.g., forks, pins, discovered attacks, queen threats).
    return false;
}

// --- Constructor, Destructor, Pool Management (mostly unchanged from previous version) ---
AuxiliaryHelperSelector::AuxiliaryHelperSelector() : rng_(std::random_device{}()) {
    engine_pool_ = std::make_unique<HelperEnginePool>();
    // InitializeEnginePool(); // Will be called by setters or explicitly when config is ready
}

AuxiliaryHelperSelector::~AuxiliaryHelperSelector() {
    ShutdownEnginePool();
}

void AuxiliaryHelperSelector::InitializeEnginePool() {
    if (!engine_pool_) {
        engine_pool_ = std::make_unique<HelperEnginePool>();
    }
    engine_busy_.assign(config_.max_helper_instances, false); // Ensure engine_busy_ is sized
    engine_pool_->Initialize(config_.max_helper_instances,
                             config_.helper_engine_path,
                             config_.engine_options,
                             config_.engine_restart_threshold);
    std::cout << "Helper engine pool initialized with " << config_.max_helper_instances << " instances." << std::endl;
}

void AuxiliaryHelperSelector::ShutdownEnginePool() {
    if (engine_pool_) {
        engine_pool_->Shutdown();
        std::cout << "Helper engine pool shut down." << std::endl;
    }
}

void AuxiliaryHelperSelector::ReconfigureEnginePool() {
    std::cout << "Reconfiguring helper engine pool..." << std::endl;
    ShutdownEnginePool();
    InitializeEnginePool(); // This will use the latest config_ values
}

// --- Main interface methods (Add*Node, GetNextRequest, OnHelperComplete) ---
void AuxiliaryHelperSelector::AddRootNode(lczero::Node* root, lczero::NodeTree* tree_context) {
    if (ShouldEnqueue(root, NodeSource::ROOT, tree_context)) {
        EnqueueNode(root, NodeSource::ROOT, tree_context);
    }
}

void AuxiliaryHelperSelector::AddDivergenceNode(lczero::Node* node, float eval_difference, lczero::NodeTree* tree_context) {
    if (std::abs(eval_difference) >= config_.eval_threshold &&
        ShouldEnqueue(node, NodeSource::DIVERGENCE, tree_context)) {
        EnqueueNode(node, NodeSource::DIVERGENCE, tree_context, eval_difference);
    }
}

void AuxiliaryHelperSelector::AddPVComparisonNode(lczero::Node* node, const std::vector<lczero::Move>& leela_pv,
                                              const std::vector<lczero::Move>& helper_pv, lczero::NodeTree* tree_context) {
    if (ShouldEnqueue(node, NodeSource::PV_COMPARISON, tree_context)) {
        float divergence_importance = CalculatePVDivergenceImportance(leela_pv, helper_pv);
        EnqueueNode(node, NodeSource::PV_COMPARISON, tree_context, divergence_importance);
    }
}

std::optional<std::pair<HelperRequest, int>> AuxiliaryHelperSelector::GetNextRequest() {
    CleanupExpiredRequests(nullptr); // TODO: Pass tree_context if IsNodeStillRelevant in cleanup needs it.

    if (request_queue_.empty()) {
        return std::nullopt;
    }

    int available_engine_idx = GetAvailableEngine();
    if (available_engine_idx == -1) {
        return std::nullopt;
    }

    HelperRequest request = request_queue_.top();
    request_queue_.pop();
    queued_nodes_.erase(request.node);

    // Using placeholder for tree_context in IsNodeStillRelevant and GetNodeDepth
    // This is a temporary measure.
    lczero::NodeTree* placeholder_tree_context = nullptr;
    // A real tree_context should be obtained if these calls are to be meaningful.
    // For now, assuming they can handle nullptr or are bypassed.

    if (!IsNodeStillRelevant(request.node, placeholder_tree_context)) {
        engine_pool_->ReleaseEngine(available_engine_idx);
        return GetNextRequest();
    }

    int current_depth = GetNodeDepth(request.node, placeholder_tree_context);
    if (current_depth > config_.max_depth && config_.max_depth > 0) { // Added check for config_.max_depth > 0
        float sample_prob = CalculateDepthSamplingProbability(current_depth);
        if (uniform_dist_(rng_) > sample_prob) {
            request.priority_score *= 0.8f;
            request_queue_.push(request);
            queued_nodes_.insert(request.node);
            engine_pool_->ReleaseEngine(available_engine_idx);
            return GetNextRequest();
        }
    }

    processing_nodes_[request.node] = available_engine_idx;
    if(static_cast<size_t>(available_engine_idx) < engine_busy_.size()){ // Bounds check
        engine_busy_[available_engine_idx] = true;
    } else {
        std::cerr << "Error: available_engine_idx out of bounds for engine_busy_." << std::endl;
        // Handle error: release engine, don't proceed with this request
        engine_pool_->ReleaseEngine(available_engine_idx);
        return GetNextRequest();
    }
    helper_start_times_[request.node] = std::chrono::steady_clock::now();
    stats_.total_requests++;

    // Populate the request with up-to-date (but possibly placeholder) position and depth
    request.position = GetNodePosition(request.node, placeholder_tree_context);
    request.depth = current_depth; // Use the depth calculated above

    return std::make_pair(request, available_engine_idx);
}

void AuxiliaryHelperSelector::OnHelperComplete(lczero::Node* node, bool success) {
    auto it = processing_nodes_.find(node);
    if (it != processing_nodes_.end()) {
        int engine_idx = it->second;
        if(static_cast<size_t>(engine_idx) < engine_busy_.size()){ // Bounds check
            engine_busy_[engine_idx] = false;
        } else {
            std::cerr << "Error: engine_idx out of bounds for engine_busy_ in OnHelperComplete." << std::endl;
        }
        engine_pool_->ReleaseEngine(engine_idx);
        processing_nodes_.erase(it);
    }

    helper_start_times_.erase(node);

    if (success) {
        stats_.successful_queries++;
    } else {
        stats_.timeouts++;
    }

    UpdateAdaptiveParameters();
}

// --- Configuration setters (mostly unchanged, ensure ReconfigureEnginePool is called appropriately) ---
void AuxiliaryHelperSelector::SetMaxDepth(int depth) { config_.max_depth = depth; }
void AuxiliaryHelperSelector::SetEvalThreshold(float threshold) { config_.eval_threshold = threshold; }
void AuxiliaryHelperSelector::SetMaxQueueSize(int size) { config_.max_queue_size = size; }

void AuxiliaryHelperSelector::SetMaxConcurrentHelpers(int count) {
    config_.max_concurrent_helpers = std::min(count, config_.max_helper_instances);
}

void AuxiliaryHelperSelector::SetMaxHelperInstances(int count) {
    if (config_.max_helper_instances != count) {
        config_.max_helper_instances = count;
        config_.max_concurrent_helpers = std::min(config_.max_concurrent_helpers, config_.max_helper_instances);
        if(engine_pool_) ReconfigureEnginePool(); // Only if pool exists
    }
}

void AuxiliaryHelperSelector::SetHelperEnginePath(const std::string& path) {
    if (config_.helper_engine_path != path) {
        config_.helper_engine_path = path;
        if(engine_pool_) ReconfigureEnginePool();
    }
}

void AuxiliaryHelperSelector::SetEngineOptions(const std::vector<std::string>& options) {
    if (config_.engine_options != options) {
        config_.engine_options = options;
        if(engine_pool_) ReconfigureEnginePool();
    }
}

void AuxiliaryHelperSelector::SetEngineRestartThreshold(int queries){
    if(config_.engine_restart_threshold != queries){
        config_.engine_restart_threshold = queries;
        if(engine_pool_) ReconfigureEnginePool();
    }
}

// --- Status and monitoring (mostly unchanged) ---
int AuxiliaryHelperSelector::GetActiveEngineCount() const {
    return std::count(engine_busy_.begin(), engine_busy_.end(), true);
}

int AuxiliaryHelperSelector::GetAvailableEngineCount() const {
    if (!engine_pool_) return 0;
    int busy_count = GetActiveEngineCount();
    return std::max(0, config_.max_concurrent_helpers - busy_count);
}

int AuxiliaryHelperSelector::GetQueueSize() const {
    return request_queue_.size();
}

float AuxiliaryHelperSelector::GetSuccessRate() const {
    return stats_.total_requests > 0 ?
           static_cast<float>(stats_.successful_queries) / stats_.total_requests : 0.0f;
}

// --- Internal methods (ShouldEnqueue, EnqueueNode, CalculatePriorityScore, etc.) ---
bool AuxiliaryHelperSelector::ShouldEnqueue(lczero::Node* node, NodeSource source, lczero::NodeTree* tree_context) {
    if (!node ) { // tree_context can be null for initial calls from LeelaAuxiliaryIntegration if not available
        return false;
    }
    if (queued_nodes_.count(node) || processing_nodes_.count(node)) {
        return false;
    }
    if (request_queue_.size() >= static_cast<size_t>(config_.max_queue_size)) {
        return false;
    }
    if (GetActiveEngineCount() >= config_.max_concurrent_helpers) {
        if (request_queue_.size() > static_cast<size_t>(config_.max_queue_size / 2)) {
            return false;
        }
    }
    if (!IsNodeStillRelevant(node, tree_context)) { // Pass tree_context here
        return false;
    }
    return true;
}

void AuxiliaryHelperSelector::EnqueueNode(lczero::Node* node, NodeSource source, lczero::NodeTree* tree_context, float importance) {
    HelperRequest request;
    request.node = node;
    request.position = GetNodePosition(node, tree_context); // Uses placeholder
    request.depth = GetNodeDepth(node, tree_context);       // Uses placeholder
    request.enqueue_time = std::chrono::steady_clock::now();
    request.priority_score = CalculatePriorityScore(node, source, tree_context, importance);

    request_queue_.push(request);
    queued_nodes_.insert(node);
}

float AuxiliaryHelperSelector::CalculatePriorityScore(lczero::Node* node, NodeSource source, lczero::NodeTree* tree_context, float importance) {
    float score = 0.0f;
    switch (source) {
        case NodeSource::ROOT: score = 1.0f; break;
        case NodeSource::DIVERGENCE: score = 0.8f + importance; break;
        case NodeSource::PV_COMPARISON: score = 0.6f + importance * 0.5f; break;
        default: score = 0.5f; break;
    }

    int visits = GetNodeVisits(node);
    if (visits > 0) {
        score += std::min(0.3f, std::log(static_cast<float>(std::max(1, visits))) / 10.0f); // std::max to avoid log(0)
    }
    if (IsTacticalPosition(node, tree_context)) { // Uses placeholder
        score += config_.tactical_position_bonus;
    }
    auto time_since_creation = std::chrono::steady_clock::now() - GetNodeCreationTime(node); // Uses new GetNodeCreationTime
    auto age_ms = std::chrono::duration_cast<std::chrono::milliseconds>(time_since_creation).count();
    if (age_ms < 5000) {
        score += config_.recent_divergence_bonus;
    }

    int calculated_depth = GetNodeDepth(node, tree_context); // Uses placeholder
    if (calculated_depth > config_.max_depth && config_.max_depth > 0) { // Added check for config_.max_depth > 0
        score *= std::max(0.1f, 1.0f - (calculated_depth - config_.max_depth) * 0.1f);
    }
    return score;
}

float AuxiliaryHelperSelector::CalculateDepthSamplingProbability(int depth) {
    if (depth <= config_.max_depth || config_.max_depth <= 0) { // Added check for config_.max_depth <= 0
        return 1.0f;
    }
    float base_prob = config_.base_probability;
    // Ensure arguments to log are positive
    float log_factor = std::log(static_cast<float>(config_.max_depth + 1)) / std::log(static_cast<float>(std::max(1, depth) + 1));

    auto it = stats_.depth_success_rate.find(depth);
    if (it != stats_.depth_success_rate.end() && stats_.total_requests > 0) {
        // This calculation needs to be: successful_at_depth / total_at_depth
        // For now, this part is commented out as stats_.depth_success_rate is not populated.
        // float success_rate_at_depth = static_cast<float>(it->second) / total_requests_at_depth;
        // base_prob *= (0.5f + success_rate_at_depth);
    }
    return std::min(1.0f, std::max(0.0f, base_prob * log_factor)); // Ensure probability is not negative
}

float AuxiliaryHelperSelector::CalculatePVDivergenceImportance(const std::vector<lczero::Move>& leela_pv,
                                                             const std::vector<lczero::Move>& helper_pv) {
    if (leela_pv.empty() || helper_pv.empty()) return 0.0f;
    size_t divergence_point = 0;
    size_t min_length = std::min(leela_pv.size(), helper_pv.size());
    bool diverged = false;
    for (size_t i = 0; i < min_length; ++i) {
        // lczero::Move needs a way to be compared. Using as_string() for now.
        if (leela_pv[i].as_string() != helper_pv[i].as_string()) {
            divergence_point = i;
            diverged = true;
            break;
        }
    }
    if (!diverged && leela_pv.size() != helper_pv.size()) {
        divergence_point = min_length;
    } else if (!diverged && leela_pv.size() == helper_pv.size()) { // PVs are identical
        return 0.0f; // No divergence
    }

    float importance = 1.0f - (static_cast<float>(divergence_point) / 10.0f); // Max importance if diverges at move 0
    return std::max(0.1f, importance);
}

void AuxiliaryHelperSelector::CleanupExpiredRequests(lczero::NodeTree* tree_context) {
    std::priority_queue<HelperRequest> temp_queue;
    auto now = std::chrono::steady_clock::now();
    while (!request_queue_.empty()) {
        HelperRequest request = request_queue_.top();
        request_queue_.pop();
        auto age = now - request.enqueue_time;
        if (age < config_.max_age_ms && IsNodeStillRelevant(request.node, tree_context)) {
            temp_queue.push(request);
        } else {
            queued_nodes_.erase(request.node);
        }
    }
    request_queue_ = std::move(temp_queue);
}

void AuxiliaryHelperSelector::UpdateAdaptiveParameters() {
    if (stats_.total_requests > 50) {
        float current_success_rate = GetSuccessRate();
        if (current_success_rate > 0.8f && config_.max_depth < 25) config_.max_depth++;
        else if (current_success_rate < 0.5f && config_.max_depth > 10) config_.max_depth--;
    }
    int active_engines = GetActiveEngineCount();
    float utilization = config_.max_helper_instances > 0 ? static_cast<float>(active_engines) / config_.max_helper_instances : 0.0f;
    if (utilization > 0.9f && config_.max_concurrent_helpers < config_.max_helper_instances) {
        config_.max_concurrent_helpers = std::min(config_.max_concurrent_helpers + 1, config_.max_helper_instances);
    } else if (utilization < 0.3f && config_.max_concurrent_helpers > 1) {
        config_.max_concurrent_helpers--;
    }
}

int AuxiliaryHelperSelector::GetAvailableEngine() {
    if (!engine_pool_ || !engine_pool_->IsEngineReady(0)) { // Quick check if pool even exists or first engine isn't ready
         // Try to initialize if not ready and path is set
        if (!config_.helper_engine_path.empty() && !engine_busy_.size()) { // engine_busy_.size() check prevents reinit loops
             std::cout << "GetAvailableEngine: Engine pool seems uninitialized or first engine not ready. Attempting to initialize..." << std::endl;
             InitializeEnginePool(); // Attempt to initialize it now
        } else if (!engine_pool_ || !engine_pool_->IsEngineReady(0)){
            // std::cerr << "GetAvailableEngine: Engine pool not initialized or no engines ready." << std::endl;
            return -1;
        }
    }
    for (int i = 0; i < config_.max_helper_instances; ++i) {
        // Ensure engine_busy_ is large enough
        if (static_cast<size_t>(i) < engine_busy_.size()) {
            if (!engine_busy_[i] && engine_pool_->IsEngineReady(i)) {
                return i;
            }
        }
    }
    return -1;
}
