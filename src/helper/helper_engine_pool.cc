#include "helper_engine_pool.h"
#include <iostream>
#include <thread>
#include <algorithm>
#include <cstring>
#include <array>
#include <cctype>
#include <sstream> // Required for std::istringstream

#ifndef _WIN32
#include <unistd.h> // For pipe, fork, dup2, execvp, close
#include <fcntl.h>  // For fcntl
#include <sys/wait.h> // For waitpid
#include <csignal>  // For signal, kill
#else
// Windows-specific includes would go here for CreateProcess, etc.
// For now, Windows support will be stubbed.
#include <windows.h>
#endif

// Helper function to trim whitespace from end of string
static void rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), s.end());
}

HelperEnginePool::HelperEnginePool() = default;

HelperEnginePool::~HelperEnginePool() {
    Shutdown();
}

void HelperEnginePool::Initialize(int engine_count, const std::string& engine_path,
                                  const std::vector<std::string>& options, int restart_threshold) {
    engine_path_ = engine_path;
    engine_options_ = options;
    engines_.resize(engine_count);
    restart_threshold_ = restart_threshold;

    for (int i = 0; i < engine_count; ++i) {
        engines_[i].engine_id_str = "helper_engine_" + std::to_string(i);
        StartEngine(i);
    }
}

void HelperEnginePool::Shutdown() {
    for (size_t i = 0; i < engines_.size(); ++i) {
        auto& engine = engines_[i];
        if (engine.process_handle) {
#ifndef _WIN32
            if (engine.engine_stdin_fd != -1) { // Using fd now
                 SendCommand(engine, "quit");
                 std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
#else
            // For Windows, check against the handle in EngineProcess
            if (engine.process_handle->hStdInWrite != NULL) {
                 SendCommand(engine, "quit");
                 std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
#endif

#ifndef _WIN32
            if (engine.engine_stdin_fd != -1) {
                close(engine.engine_stdin_fd);
                engine.engine_stdin_fd = -1;
            }
            if (engine.engine_stdout_fd != -1) {
                close(engine.engine_stdout_fd);
                engine.engine_stdout_fd = -1;
            }
            if (engine.process_handle->pid > 0) {
                int status;
                // Non-blocking check if process exited, then kill if not.
                if (waitpid(engine.process_handle->pid, &status, WNOHANG) == 0) {
                    kill(engine.process_handle->pid, SIGTERM);
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                    if (waitpid(engine.process_handle->pid, &status, WNOHANG) == 0) {
                         kill(engine.process_handle->pid, SIGKILL);
                         waitpid(engine.process_handle->pid, &status, 0); // blocking wait for SIGKILL
                    }
                }
                engine.process_handle->pid = 0;
            }
#else
            // Windows: TerminateProcess, CloseHandle, etc.
            if (engine.process_handle->hStdInWrite != NULL) { // This was parent's write handle to child's stdin
                CloseHandle(engine.process_handle->hStdInWrite);
                engine.process_handle->hStdInWrite = NULL;
            }
            if (engine.process_handle->hStdOutRead != NULL) { // This was parent's read handle from child's stdout
                CloseHandle(engine.process_handle->hStdOutRead);
                engine.process_handle->hStdOutRead = NULL;
            }
            if (engine.process_handle->hProcess != NULL) {
                TerminateProcess(engine.process_handle->hProcess, 0);
                CloseHandle(engine.process_handle->hProcess);
                engine.process_handle->hProcess = NULL;
            }
#endif
            engine.process_handle.reset();
        }
    }
    engines_.clear();
}

#ifndef _WIN32
// POSIX implementation of StartEngine
void HelperEnginePool::StartEngine(int engine_idx) {
    if (engine_idx < 0 || engine_idx >= engines_.size()) {
        std::cerr << "StartEngine: Invalid engine index " << engine_idx << std::endl;
        return;
    }
    auto& engine = engines_[engine_idx];

    engine.process_handle = std::make_unique<EngineProcess>();
    engine.process_handle->id_str = engine.engine_id_str;

    int parent_to_child_pipe[2];
    int child_to_parent_pipe[2];

    if (pipe(parent_to_child_pipe) == -1 || pipe(child_to_parent_pipe) == -1) {
        std::cerr << "Failed to create pipes for " << engine.engine_id_str << std::endl;
        engine.ready = false;
        return;
    }

    pid_t pid = fork();

    if (pid < 0) {
        std::cerr << "Failed to fork process for " << engine.engine_id_str << std::endl;
        close(parent_to_child_pipe[0]);
        close(parent_to_child_pipe[1]);
        close(child_to_parent_pipe[0]);
        close(child_to_parent_pipe[1]);
        engine.ready = false;
        return;
    }

    if (pid == 0) { // Child process
        close(parent_to_child_pipe[1]); // Close write end of parent_to_child
        close(child_to_parent_pipe[0]); // Close read end of child_to_parent

        dup2(parent_to_child_pipe[0], STDIN_FILENO);
        dup2(child_to_parent_pipe[1], STDOUT_FILENO);
        // Optional: redirect stderr as well, e.g., to /dev/null or a log file
        // int dev_null = open("/dev/null", O_WRONLY);
        // if (dev_null != -1) dup2(dev_null, STDERR_FILENO);

        close(parent_to_child_pipe[0]); // Close original fds after dup
        close(child_to_parent_pipe[1]);
        // if (dev_null != -1) close(dev_null);

        // Prepare arguments for execvp
        // engine_path_ might contain arguments, so we need to parse it or use a shell
        // For simplicity, assuming engine_path_ is just the executable if no spaces
        // A more robust solution would parse engine_path_ or use wordexp/similar
        char* argv[64]; // Max 63 arguments + null terminator
        std::string path_copy = engine_path_;
        char* token = strtok(&path_copy[0], " ");
        int i = 0;
        while (token != NULL && i < 63) {
            argv[i++] = token;
            token = strtok(NULL, " ");
        }
        argv[i] = NULL;

        execvp(argv[0], argv);
        // If execvp returns, it's an error
        std::cerr << "Failed to exec engine: " << engine_path_ << " (errno: " << errno << ")" << std::endl;
        _exit(1); // Use _exit in child after fork to avoid calling destructors, etc.
    }
    else { // Parent process
        close(parent_to_child_pipe[0]); // Close read end of parent_to_child
        close(child_to_parent_pipe[1]); // Close write end of child_to_parent

        engine.engine_stdin_fd = parent_to_child_pipe[1];
        engine.engine_stdout_fd = child_to_parent_pipe[0];
        engine.process_handle->pid = pid;

        // Optional: Set pipes to non-blocking if ReadResponse/SendCommand will handle it
        // fcntl(engine.engine_stdout_fd, F_SETFL, O_NONBLOCK);
        // fcntl(engine.engine_stdin_fd, F_SETFL, O_NONBLOCK); // Usually not needed for stdin pipe
    }

    engine.query_count = 0;
    engine.last_used = std::chrono::steady_clock::now();
    engine.ready = false;

    std::cout << "Attempting UCI handshake for " << engine.engine_id_str << std::endl;

    if (SendCommand(engine, "uci")) {
        if (WaitForResponse(engine, "uciok")) {
            std::cout << "UCI OK for " << engine.engine_id_str << std::endl;
            for (const auto& option : engine_options_) {
                SendCommand(engine, "setoption " + option);
            }
            SendCommand(engine, "isready");
            if (WaitForResponse(engine, "readyok")) {
                engine.ready = true;
                std::cout << engine.engine_id_str << " is ready." << std::endl;
                SendCommand(engine, "ucinewgame");
            } else {
                std::cerr << "Engine not ready (no readyok): " << engine.engine_id_str << std::endl;
            }
        } else {
            std::cerr << "UCI handshake failed (no uciok): " << engine.engine_id_str << std::endl;
        }
    } else {
         std::cerr << "Failed to send 'uci' command to " << engine.engine_id_str << std::endl;
    }

    if (!engine.ready) {
        std::cerr << "Engine " << engine.engine_id_str << " failed to initialize properly." << std::endl;
        if (engine.engine_stdin_fd != -1) { close(engine.engine_stdin_fd); engine.engine_stdin_fd = -1; }
        if (engine.engine_stdout_fd != -1) { close(engine.engine_stdout_fd); engine.engine_stdout_fd = -1; }
        if (engine.process_handle && engine.process_handle->pid > 0) {
            kill(engine.process_handle->pid, SIGKILL);
            waitpid(engine.process_handle->pid, NULL, 0);
            engine.process_handle->pid = 0;
        }
        engine.process_handle.reset();
    }
}

#else // _WIN32
// Windows implementation of StartEngine
void HelperEnginePool::StartEngine(int engine_idx) {
    if (engine_idx < 0 || engine_idx >= engines_.size()) {
        std::cerr << "StartEngine: Invalid engine index " << engine_idx << std::endl;
        return;
    }
    auto& engine = engines_[engine_idx];
    engine.process_handle = std::make_unique<EngineProcess>();
    engine.process_handle->id_str = engine.engine_id_str;

    // std::cerr << "Windows process creation for " << engine.engine_id_str << " is not yet implemented." << std::endl;

    HANDLE hChildStd_IN_Rd = NULL;
    HANDLE hChildStd_IN_Wr = NULL;
    HANDLE hChildStd_OUT_Rd = NULL;
    HANDLE hChildStd_OUT_Wr = NULL;

    SECURITY_ATTRIBUTES saAttr;
    saAttr.nLength = sizeof(SECURITY_ATTRIBUTES);
    saAttr.bInheritHandle = TRUE;
    saAttr.lpSecurityDescriptor = NULL;

    if (!CreatePipe(&hChildStd_OUT_Rd, &hChildStd_OUT_Wr, &saAttr, 0)) {
        std::cerr << "StdoutRd CreatePipe failed for " << engine.engine_id_str << " (Error: " << GetLastError() << ")" << std::endl;
        engine.ready = false; return;
    }
    if (!SetHandleInformation(hChildStd_OUT_Rd, HANDLE_FLAG_INHERIT, 0)) {
        std::cerr << "Stdout SetHandleInformation failed for " << engine.engine_id_str << " (Error: " << GetLastError() << ")" << std::endl;
        CloseHandle(hChildStd_OUT_Rd); CloseHandle(hChildStd_OUT_Wr);
        engine.ready = false; return;
    }
    if (!CreatePipe(&hChildStd_IN_Rd, &hChildStd_IN_Wr, &saAttr, 0)) {
        std::cerr << "Stdin CreatePipe failed for " << engine.engine_id_str << " (Error: " << GetLastError() << ")" << std::endl;
        CloseHandle(hChildStd_OUT_Rd); CloseHandle(hChildStd_OUT_Wr);
        engine.ready = false; return;
    }
    if (!SetHandleInformation(hChildStd_IN_Wr, HANDLE_FLAG_INHERIT, 0)) {
        std::cerr << "Stdin SetHandleInformation failed for " << engine.engine_id_str << " (Error: " << GetLastError() << ")" << std::endl;
        CloseHandle(hChildStd_OUT_Rd); CloseHandle(hChildStd_OUT_Wr);
        CloseHandle(hChildStd_IN_Rd); CloseHandle(hChildStd_IN_Wr);
        engine.ready = false; return;
    }

    PROCESS_INFORMATION piProcInfo;
    STARTUPINFOA siStartInfo;
    ZeroMemory(&piProcInfo, sizeof(PROCESS_INFORMATION));
    ZeroMemory(&siStartInfo, sizeof(STARTUPINFOA));
    siStartInfo.cb = sizeof(STARTUPINFOA);
    siStartInfo.hStdError = hChildStd_OUT_Wr;
    siStartInfo.hStdOutput = hChildStd_OUT_Wr;
    siStartInfo.hStdInput = hChildStd_IN_Rd;
    siStartInfo.dwFlags |= STARTF_USESTDHANDLES;

    std::string commandLine = engine_path_;

    BOOL bSuccess = CreateProcessA(
        NULL,
        &commandLine[0],
        NULL,
        NULL,
        TRUE,
        CREATE_NO_WINDOW,
        NULL,
        NULL,
        &siStartInfo,
        &piProcInfo);

    if (!bSuccess) {
        std::cerr << "CreateProcess failed for " << engine.engine_id_str << " (error: " << GetLastError() << ")" << std::endl;
        CloseHandle(hChildStd_OUT_Rd); CloseHandle(hChildStd_OUT_Wr);
        CloseHandle(hChildStd_IN_Rd); CloseHandle(hChildStd_IN_Wr);
        engine.ready = false; return;
    }

    engine.process_handle->hProcess = piProcInfo.hProcess;
    CloseHandle(piProcInfo.hThread);

    CloseHandle(hChildStd_OUT_Wr); // Parent closes child's write end of stdout pipe
    CloseHandle(hChildStd_IN_Rd);  // Parent closes child's read end of stdin pipe

    engine.process_handle->hStdInWrite = hChildStd_IN_Wr; // Parent uses this to write to child's stdin
    engine.process_handle->hStdOutRead = hChildStd_OUT_Rd; // Parent uses this to read from child's stdout

    engine.query_count = 0;
    engine.last_used = std::chrono::steady_clock::now();
    engine.ready = false;

    std::cout << "Attempting UCI handshake for " << engine.engine_id_str << std::endl;

    if (SendCommand(engine, "uci")) {
        if (WaitForResponse(engine, "uciok")) {
            std::cout << "UCI OK for " << engine.engine_id_str << std::endl;
            for (const auto& option : engine_options_) {
                SendCommand(engine, "setoption " + option);
            }
            SendCommand(engine, "isready");
            if (WaitForResponse(engine, "readyok")) {
                engine.ready = true;
                std::cout << engine.engine_id_str << " is ready." << std::endl;
                SendCommand(engine, "ucinewgame");
            } else {
                std::cerr << "Engine not ready (no readyok): " << engine.engine_id_str << std::endl;
            }
        } else {
            std::cerr << "UCI handshake failed (no uciok): " << engine.engine_id_str << std::endl;
        }
    } else {
         std::cerr << "Failed to send 'uci' command to " << engine.engine_id_str << std::endl;
    }

    if (!engine.ready) {
        std::cerr << "Engine " << engine.engine_id_str << " failed to initialize properly." << std::endl;
        if (engine.process_handle) {
            if (engine.process_handle->hStdInWrite != NULL) {
                CloseHandle(engine.process_handle->hStdInWrite);
                engine.process_handle->hStdInWrite = NULL;
            }
            if (engine.process_handle->hStdOutRead != NULL) {
                CloseHandle(engine.process_handle->hStdOutRead);
                engine.process_handle->hStdOutRead = NULL;
            }
            if (engine.process_handle->hProcess != NULL) {
                 TerminateProcess(engine.process_handle->hProcess, 1);
                 CloseHandle(engine.process_handle->hProcess);
                 engine.process_handle->hProcess = NULL;
            }
            engine.process_handle.reset();
        }
    }
}
#endif

void HelperEnginePool::RestartEngine(int engine_idx) {
    if (engine_idx < 0 || engine_idx >= engines_.size()) return;
    auto& engine = engines_[engine_idx];

    std::cout << "Restarting engine: " << engine.engine_id_str << std::endl;

    if (engine.process_handle) {
#ifndef _WIN32
        if (engine.engine_stdin_fd != -1) {
            SendCommand(engine, "quit");
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            close(engine.engine_stdin_fd);
            engine.engine_stdin_fd = -1;
        }
        if (engine.engine_stdout_fd != -1) {
            close(engine.engine_stdout_fd);
            engine.engine_stdout_fd = -1;
        }
        if (engine.process_handle->pid > 0) {
            kill(engine.process_handle->pid, SIGTERM);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            int status;
            if (waitpid(engine.process_handle->pid, &status, WNOHANG) == 0) {
                kill(engine.process_handle->pid, SIGKILL);
                waitpid(engine.process_handle->pid, &status, 0);
            }
            engine.process_handle->pid = 0;
        }
#else
        if (engine.process_handle->hStdInWrite != NULL) {
            SendCommand(engine, "quit");
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            CloseHandle(engine.process_handle->hStdInWrite);
            engine.process_handle->hStdInWrite = NULL;
        }
        if (engine.process_handle->hStdOutRead != NULL) {
            CloseHandle(engine.process_handle->hStdOutRead);
            engine.process_handle->hStdOutRead = NULL;
        }
        if (engine.process_handle->hProcess != NULL) {
            TerminateProcess(engine.process_handle->hProcess, 0);
            CloseHandle(engine.process_handle->hProcess);
            engine.process_handle->hProcess = NULL;
        }
#endif
        engine.process_handle.reset();
    }

    engine.ready = false;
    engine.query_count = 0;

    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    StartEngine(engine_idx);
}

bool HelperEnginePool::SendCommand(EngineInstance& engine, const std::string& command) {
    if (!engine.process_handle) {
        std::cerr << "SendCommand: Engine " << engine.engine_id_str << " no process handle." << std::endl;
        return false;
    }

    std::string full_command = command + "\n";
#ifndef _WIN32
    if (engine.engine_stdin_fd == -1) {
        std::cerr << "SendCommand: Engine " << engine.engine_id_str << " stdin_fd is -1." << std::endl;
        return false;
    }
    // std::cout << "[" << engine.engine_id_str << "] CMD: " << command << std::endl;
    ssize_t bytes_written = write(engine.engine_stdin_fd, full_command.c_str(), full_command.length());
    if (bytes_written < 0 || static_cast<size_t>(bytes_written) != full_command.length()) {
        std::cerr << "SendCommand: Failed to write to engine " << engine.engine_id_str << " stdin (errno: " << errno << ")." << std::endl;
        engine.ready = false;
        return false;
    }
#else
    if (engine.process_handle->hStdInWrite == NULL) {
        std::cerr << "SendCommand: Engine " << engine.engine_id_str << " hStdInWrite is NULL." << std::endl;
        return false;
    }
    DWORD dwWritten;
    // std::cout << "[" << engine.engine_id_str << "] CMD: " << command << std::endl;
    if (!WriteFile(engine.process_handle->hStdInWrite, full_command.c_str(), full_command.length(), &dwWritten, NULL) || dwWritten != full_command.length()) {
        std::cerr << "SendCommand: Failed to write to engine " << engine.engine_id_str << " stdin (error: " << GetLastError() << ")." << std::endl;
        engine.ready = false;
        return false;
    }
#endif
    return true;
}

// This ReadResponse is simplified. A robust version needs better timeout handling,
// especially for POSIX (using select/poll) and Windows (Overlapped I/O or PeekNamedPipe).
// The current POSIX version uses a blocking char-by-char read which is inefficient and only has a coarse timeout.
// The Windows version reads a chunk, which is better, but also has basic timeout/error handling.
std::string HelperEnginePool::ReadResponse(EngineInstance& engine, const std::string& expected_prefix, int timeout_ms) {
    if (!engine.process_handle) {
        std::cerr << "ReadResponse: Engine " << engine.engine_id_str << " no process handle." << std::endl;
        return "";
    }

    std::string line_buffer;
    line_buffer.reserve(1024); // Pre-allocate some space

#ifndef _WIN32
    if (engine.engine_stdout_fd == -1) {
         std::cerr << "ReadResponse: Engine " << engine.engine_id_str << " stdout_fd is -1." << std::endl;
        return "";
    }

    // Basic timeout setup using select
    fd_set read_fds;
    FD_ZERO(&read_fds);
    FD_SET(engine.engine_stdout_fd, &read_fds);

    struct timeval tv;
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;

    char ch;
    auto overall_start_time = std::chrono::steady_clock::now();

    while(true) {
        // Recalculate timeout for select
        auto now = std::chrono::steady_clock::now();
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - overall_start_time).count();
        long remaining_ms = timeout_ms - elapsed_ms;
        if (remaining_ms <= 0) {
            std::cerr << "ReadResponse: Overall timeout for " << engine.engine_id_str << std::endl;
            return ""; // Overall timeout
        }
        tv.tv_sec = remaining_ms / 1000;
        tv.tv_usec = (remaining_ms % 1000) * 1000;

        int retval = select(engine.engine_stdout_fd + 1, &read_fds, NULL, NULL, &tv);

        if (retval == -1) {
            std::cerr << "ReadResponse: select() error for " << engine.engine_id_str << " (errno: " << errno << ")" << std::endl;
            engine.ready = false;
            return "";
        } else if (retval == 0) {
            //std::cerr << "ReadResponse: select() timeout for " << engine.engine_id_str << std::endl;
            return ""; // Timeout
        }

        // Data is available, read one char
        ssize_t bytes_read = read(engine.engine_stdout_fd, &ch, 1);
        if (bytes_read > 0) {
            if (ch == '\r') continue; // Skip carriage return
            if (ch == '\n') {
                break; // End of line
            }
            line_buffer += ch;
            if (line_buffer.length() >= 4095) { // Prevent overflow (4096 with null term)
                 std::cerr << "ReadResponse: Line buffer overflow for " << engine.engine_id_str << std::endl;
                 break;
            }
        } else if (bytes_read == 0) { // EOF
            std::cerr << "ReadResponse: EOF reached for " << engine.engine_id_str << std::endl;
            engine.ready = false;
            return "";
        } else { // Error
            if (errno == EINTR) continue; // Interrupted by signal, try again
            std::cerr << "ReadResponse: read error for " << engine.engine_id_str << " (errno: " << errno << ")" << std::endl;
            engine.ready = false;
            return "";
        }
        // Re-set FD_SET as select might modify it
        FD_ZERO(&read_fds);
        FD_SET(engine.engine_stdout_fd, &read_fds);
    }
#else
    // Windows: Using PeekNamedPipe and ReadFile for a slightly better timeout mechanism
    if (engine.process_handle->hStdOutRead == NULL) {
        std::cerr << "ReadResponse: Engine " << engine.engine_id_str << " hStdOutRead is NULL." << std::endl;
        return "";
    }

    std::array<char, 4096> buffer;
    DWORD dwRead = 0;
    DWORD dwAvail = 0;
    auto start_time = std::chrono::steady_clock::now();

    // Temporary buffer for characters read one by one to find newline
    char ch_buf[1];

    while(true) {
        // Check if data is available
        if (!PeekNamedPipe(engine.process_handle->hStdOutRead, NULL, 0, NULL, &dwAvail, NULL)) {
            std::cerr << "ReadResponse: PeekNamedPipe failed for " << engine.engine_id_str << " (error: " << GetLastError() << ")" << std::endl;
            engine.ready = false;
            return "";
        }

        if (dwAvail > 0) {
            // Read one character at a time to find newline
            if (ReadFile(engine.process_handle->hStdOutRead, ch_buf, 1, &dwRead, NULL) && dwRead > 0) {
                if (ch_buf[0] == '\r') continue; // Skip carriage return
                if (ch_buf[0] == '\n') {
                    break; // End of line
                }
                line_buffer += ch_buf[0];
                if (line_buffer.length() >= buffer.size() -1) {
                    std::cerr << "ReadResponse: Line buffer overflow for " << engine.engine_id_str << std::endl;
                    break;
                }
            } else { // ReadFile failed or read 0 bytes when dwAvail > 0 (should be rare)
                DWORD error = GetLastError();
                if (error == ERROR_BROKEN_PIPE) { // EOF
                    std::cerr << "ReadResponse: EOF (broken pipe) for " << engine.engine_id_str << std::endl;
                } else {
                    std::cerr << "ReadResponse: ReadFile error for " << engine.engine_id_str << " (error: " << error << ")" << std::endl;
                }
                engine.ready = false;
                return "";
            }
        } else { // No data available
            if (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time).count() > timeout_ms) {
                //std::cerr << "ReadResponse: Timeout (no data) for " << engine.engine_id_str << std::endl;
                return ""; // Timeout
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1)); // Small sleep to avoid busy waiting
        }
    }
#endif

    rtrim(line_buffer);
    // std::cout << "[" << engine.engine_id_str << "] RAW_RESP: " << line_buffer << std::endl;
    return line_buffer;
}


bool HelperEnginePool::WaitForResponse(EngineInstance& engine, const std::string& expected_response, int timeout_ms) {
    if (!engine.process_handle) return false;

#ifndef _WIN32
    if (engine.engine_stdout_fd == -1 && !engine.process_handle) return false;
#else
    if (engine.process_handle->hStdOutRead == NULL && !engine.process_handle) return false;
#endif


    auto start_time = std::chrono::steady_clock::now();
    std::string line;
    do {
        // Calculate remaining time for ReadResponse
        auto current_time_iter = std::chrono::steady_clock::now();
        long elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(current_time_iter - start_time).count();
        int remaining_timeout = timeout_ms - static_cast<int>(elapsed_ms);
        if (remaining_timeout <= 0) {
            //std::cerr << "WaitForResponse: Overall timeout before ReadResponse for '" << expected_response << "' from " << engine.engine_id_str << std::endl;
            return false;
        }

        line = ReadResponse(engine, expected_response, remaining_timeout);

        if (!line.empty() && line.find(expected_response) != std::string::npos) {
            return true;
        }
        // If ReadResponse returns empty, it could be a timeout or an error.
        // If engine became not ready, ReadResponse handled it.
        if (!engine.ready && line.empty()) {
             //std::cerr << "WaitForResponse: ReadResponse indicated engine not ready or error for " << engine.engine_id_str << std::endl;
            return false;
        }

        // Check overall timeout again after ReadResponse
        auto current_time_after_read = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(current_time_after_read - start_time).count() > timeout_ms) {
            //std::cerr << "WaitForResponse: Timeout waiting for '" << expected_response << "' from " << engine.engine_id_str << std::endl;
            return false;
        }
        // If line is empty due to ReadResponse timeout but engine is still ready, loop again
    } while (engine.ready);

    // std::cerr << "WaitForResponse: Engine not ready, exiting loop for " << engine.engine_id_str << std::endl;
    return false;
}


bool HelperEnginePool::IsEngineReady(int engine_idx) const {
    if (engine_idx >= 0 && engine_idx < engines_.size()) {
        const auto& engine = engines_[engine_idx];
        bool pipes_ok = false;
#ifndef _WIN32
        pipes_ok = engine.engine_stdin_fd != -1 && engine.engine_stdout_fd != -1;
#else
        pipes_ok = engine.process_handle && engine.process_handle->hStdInWrite != NULL && engine.process_handle->hStdOutRead != NULL;
#endif
        return engine.ready && engine.process_handle && pipes_ok;
    }
    return false;
}

void HelperEnginePool::ReleaseEngine(int engine_idx) {
    if (engine_idx >= 0 && engine_idx < engines_.size()) {
        auto& engine = engines_[engine_idx];
        engine.last_used = std::chrono::steady_clock::now();
        engine.query_count++;

        if (restart_threshold_ > 0 && engine.query_count >= restart_threshold_) {
            std::cout << "Engine " << engine.engine_id_str << " reached query threshold (" << engine.query_count << "). Restarting." << std::endl;
            RestartEngine(engine_idx);
        }
    }
}

bool HelperEnginePool::SendPosition(int engine_idx, const std::string& fen) {
    if (!IsEngineReady(engine_idx)) {
        //std::cerr << "SendPosition: Engine " << engines_[engine_idx].engine_id_str << " not ready." << std::endl;
        return false;
    }
    return SendCommand(engines_[engine_idx], "position fen " + fen);
}

bool HelperEnginePool::RequestAnalysis(int engine_idx, int depth, int time_ms) {
    if (!IsEngineReady(engine_idx)) {
        // std::cerr << "RequestAnalysis: Engine " << engines_[engine_idx].engine_id_str << " not ready." << std::endl;
        return false;
    }

    std::string command = "go";
    if (depth > 0) {
        command += " depth " + std::to_string(depth);
    }
    if (time_ms > 0) {
        command += " movetime " + std::to_string(time_ms);
    } else if (depth <= 0) { // Avoid 'go' without parameters unless intended (e.g. infinite)
        command += " infinite"; // Or some default if no time/depth
    }
    return SendCommand(engines_[engine_idx], command);
}

std::string HelperEnginePool::GetBestMove(int engine_idx) {
    if (!IsEngineReady(engine_idx)) {
        // std::cerr << "GetBestMove: Engine " << engine_idx << " not ready at start." << std::endl;
        return "";
    }

    std::string bestmove_line;
    auto start_time = std::chrono::steady_clock::now();
    // Timeout for GetBestMove should be generous, engine might be thinking.
    // This timeout is for receiving the 'bestmove' line itself, not total engine thinking time.
    // Total engine thinking time is controlled by 'movetime' in RequestAnalysis.
    int response_line_timeout_ms = 15000; // Max time to wait for *any* line, including bestmove.

    while(IsEngineReady(engine_idx)) { // Check IsEngineReady to ensure pipes are still good
        auto current_time_iter = std::chrono::steady_clock::now();
        long elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(current_time_iter - start_time).count();
        int remaining_timeout = response_line_timeout_ms - static_cast<int>(elapsed_ms);

        if (remaining_timeout <= 0) {
            std::cerr << "GetBestMove: Overall timeout waiting for bestmove line from " << engines_[engine_idx].engine_id_str << std::endl;
            return "";
        }

        bestmove_line = ReadResponse(engines_[engine_idx], "bestmove", remaining_timeout);

        if (bestmove_line.rfind("bestmove", 0) == 0) { // Check if line starts with "bestmove"
            // Expected format: "bestmove <move> [ponder <move>]"
            std::string token;
            std::string move_str;
            std::istringstream iss(bestmove_line);
            iss >> token; // "bestmove"
            if (iss >> move_str) { // next token is the move
                // Ponder is optional, move_str will contain only the best move here.
                return move_str;
            } else {
                 std::cerr << "GetBestMove: Malformed bestmove line from " << engines_[engine_idx].engine_id_str << ": " << bestmove_line << std::endl;
                 return ""; // Malformed
            }
        }

        // If ReadResponse returns empty, it could be a timeout or an error.
        // If engine became not ready, ReadResponse would have set engine.ready to false.
        if (bestmove_line.empty() && !engines_[engine_idx].ready) {
            std::cerr << "GetBestMove: Engine " << engines_[engine_idx].engine_id_str << " not ready after ReadResponse." << std::endl;
            return "";
        }
        // If line is empty (e.g. ReadResponse timed out) but engine is still ready, continue loop.
        // Other info lines from the engine will be consumed and ignored by this loop.
    }
    // std::cerr << "GetBestMove: Engine " << engines_[engine_idx].engine_id_str << " not ready, exiting loop." << std::endl;
    return "";
}

std::vector<std::string> HelperEnginePool::GetPrincipalVariation(int engine_idx) {
    if (!IsEngineReady(engine_idx)) return {};
    // This is a placeholder. A real implementation would need to parse "info ... pv ..." lines
    // which are typically received *before* the "bestmove" line.
    // This might require changes to ReadResponse/GetBestMove to buffer/process info lines.
    std::cerr << "GetPrincipalVariation for " << engines_[engine_idx].engine_id_str << " is not implemented." << std::endl;
    return {};
}
