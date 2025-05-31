#ifndef HELPER_ENGINE_POOL_H
#define HELPER_ENGINE_POOL_H

#include <vector>
#include <string>
#include <memory>
#include <chrono>
// No <cstdio> needed directly here anymore for FILE*

#ifdef _WIN32
#include <windows.h> // For HANDLE
#endif

struct EngineProcess {
    std::string id_str;
#ifndef _WIN32
    pid_t pid = 0;
#else
    HANDLE hProcess = NULL;
    // For Windows, we'd store pipe HANDLES here if not directly in EngineInstance
    HANDLE hStdInWrite = NULL; // Parent's handle to child's stdin
    HANDLE hStdOutRead = NULL; // Parent's handle to child's stdout
#endif
};


class HelperEnginePool {
public:
    HelperEnginePool();
    ~HelperEnginePool();

    void Initialize(int engine_count, const std::string& engine_path,
                    const std::vector<std::string>& options, int restart_threshold);

    void Shutdown();
    bool IsEngineReady(int engine_idx) const;
    void ReleaseEngine(int engine_idx);
    bool SendPosition(int engine_idx, const std::string& fen);
    bool RequestAnalysis(int engine_idx, int depth, int time_ms);
    std::string GetBestMove(int engine_idx);
    std::vector<std::string> GetPrincipalVariation(int engine_idx);

private:
    struct EngineInstance {
        std::unique_ptr<EngineProcess> process_handle;
        bool ready = false;
        int query_count = 0;
        std::chrono::steady_clock::time_point last_used;
        std::string engine_id_str;

#ifndef _WIN32
        int engine_stdin_fd = -1;  // Pipe to engine's stdin
        int engine_stdout_fd = -1; // Pipe from engine's stdout
#else
        // On Windows, pipe handles are stored in EngineProcess as hStdInWrite and hStdOutRead.
        // No separate members needed here for Windows pipe handles.
#endif
    };

    void StartEngine(int engine_idx);
    void RestartEngine(int engine_idx);
    bool SendCommand(EngineInstance& engine, const std::string& command);
    std::string ReadResponse(EngineInstance& engine, const std::string& expected_prefix, int timeout_ms = 5000);
    bool WaitForResponse(EngineInstance& engine, const std::string& expected_response, int timeout_ms = 5000);

    std::vector<EngineInstance> engines_;
    std::string engine_path_;
    std::vector<std::string> engine_options_;
    int restart_threshold_ = 1000;
};

#endif // HELPER_ENGINE_POOL_H
