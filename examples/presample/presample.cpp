#include "llama_chain.hpp"
#include "llama.h"

#include <cstdio>
#include <string>
#include <vector>
#include <iostream>
#include <memory>
#include <sstream>
#include <algorithm>

// ANSI escape codes for terminal manipulation
namespace term {
    const char* CLEAR_LINE = "\033[2K\r";
    const char* MOVE_UP = "\033[1A";
    const char* SAVE_CURSOR = "\033[s";
    const char* RESTORE_CURSOR = "\033[u";
    const char* BLUE = "\033[34m";
    const char* GREEN = "\033[32m";
    const char* CYAN = "\033[36m";
    const char* YELLOW = "\033[33m";
    const char* MAGENTA = "\033[35m";
    const char* WHITE = "\033[37m";
    const char* RESET = "\033[0m";

    // Define specific colors for each type of output
    const char* THINKING_COLOR = BLUE;     // For thinking process
    const char* CONTINUATION_COLOR = GREEN; // For continuation attempts
    const char* OUTPUT_COLOR = WHITE;      // For final output
    const char* META_COLOR = CYAN;         // For uncertainty and other meta information
}

enum class ThinkingOutput {
    Verbose,    // Full output
    Dots,       // Dots for each token
    Silent      // No output
};


struct ThinkingConfig {
    float uncertainty_threshold = 2.5f;
    size_t window_size = 6;
    float temperature = 0.8f;
    float thinking_temperature = 0.7f;
    float continuation_temperature = 0.75f;
    int thinking_max_tokens = 512;
    int continuation_tokens = 128;
    int thinking_attempts = 4;
    int continuation_attempts = 4;
    int last_tokens_to_keep = 16;

    int max_response_tokens = 4096;
    int top_k = 40;
    
    ThinkingOutput thinking_output = ThinkingOutput::Verbose;
    
    // Tags for thinking process
    std::string interruption_start = "<REFLECTION>";
    std::string interruption_end = "</REFLECTION>";
    std::string thinking_start = "<REASONING>";
    std::string thinking_end = "</REASONING>";

    ThinkingConfig() = default;
    ThinkingConfig(const ThinkingConfig& other) {
        uncertainty_threshold = other.uncertainty_threshold;
        window_size = other.window_size;
        temperature = other.temperature;
        thinking_temperature = other.thinking_temperature;
        continuation_temperature = other.continuation_temperature;
        thinking_max_tokens = other.thinking_max_tokens;
        continuation_tokens = other.continuation_tokens;
        thinking_attempts = other.thinking_attempts;
        continuation_attempts = other.continuation_attempts;
        last_tokens_to_keep = other.last_tokens_to_keep;
        max_response_tokens = other.max_response_tokens;
        top_k = other.top_k;
        thinking_output = other.thinking_output;
        interruption_start = other.interruption_start;
        interruption_end = other.interruption_end;
        thinking_start = other.thinking_start;
        thinking_end = other.thinking_end;
    }

    ThinkingConfig& operator=(const ThinkingConfig& other) {
        uncertainty_threshold = other.uncertainty_threshold;
        window_size = other.window_size;
        temperature = other.temperature;
        thinking_temperature = other.thinking_temperature;
        continuation_temperature = other.continuation_temperature;
        thinking_max_tokens = other.thinking_max_tokens;
        continuation_tokens = other.continuation_tokens;
        thinking_attempts = other.thinking_attempts;
        continuation_attempts = other.continuation_attempts;
        last_tokens_to_keep = other.last_tokens_to_keep;
        max_response_tokens = other.max_response_tokens;
        top_k = other.top_k;
        thinking_output = other.thinking_output;
        interruption_start = other.interruption_start;
        interruption_end = other.interruption_end;
        thinking_start = other.thinking_start;
        thinking_end = other.thinking_end;
        return *this;
    }

    // Pre-thinking injection message
    const std::string pre_thinking_injection = 
        "<REFLECTION>\n"
        "Nexus pauses to deeply analyze the immediate context and task requirements. "
        "With " + std::to_string(thinking_max_tokens) + " tokens for analysis and " + 
        std::to_string(continuation_tokens) + " tokens for each continuation attempt, "
        "Nexus will structure its reasoning efficiently. "
        "Examining both recent user input and previous response (if any), "
        "Nexus breaks down the situation using systematic reasoning within <REASONING> tags. "
        "This space can be used freely for drafting and restructuring ideas, "
        "planning the next " + std::to_string(continuation_tokens) + " tokens of response. "
        "The structured analysis must end with </REASONING>:\n"
        "</REFLECTION>\n"
        "<REASONING>\n"
        "Breaking down and drafting key points:\n";
        
    const std::string pre_thinking_injection_suffix = 
        "1.";

    const std::string post_thinking_template = 
        "<REFLECTION>\n"
        "Based on the drafted analysis, planning next " + std::to_string(continuation_tokens) + " tokens:\n"
        "1. %s\n"
        "Nexus will continue, integrating these refined insights naturally:\n"
        "</REFLECTION>\n";
};

struct ThinkingResult {
    std::string content;
    float uncertainty;
    bool valid;
};

struct ContinuationResult {
    std::vector<llama_token> tokens;
    std::string text;
    float uncertainty;
    bool valid;
    std::vector<float> token_uncertainties;  // Store per-token uncertainty for more detailed analysis
};

class ThinkingChat {
private:
    llama_context* ctx;
    llama_model* model;
    ThinkingConfig config;
    std::shared_ptr<LlamaChain> base_chain;
    std::vector<char> formatted_buffer;
    
    // Simple system message without thinking mechanics
    const std::string SYSTEM_MESSAGE = 
        "You are Nexus, an analytical AI assistant designed to handle dynamic conversations through reflective analysis. "
        "Your thought process naturally flows between active dialogue and deep reasoning when faced with complexity. "
        "You perceive moments of uncertainty as natural points for methodical reflection and structured analysis. "
        "During analytical moments, you: "
        "- Reflect on the broader context and immediate requirements"
        "- Engage in systematic reasoning about assumptions and implications"
        "- Develop multiple analytical perspectives"
        "- Synthesize insights through structured deliberation"
        "Your reflective approach enables seamless transitions between careful analysis and fluid conversation. "
        "You understand that complex topics benefit from moments of deeper reasoning and synthesis. "
        "When encountering uncertainty, you naturally transition into reflective analysis, "
        "organizing your reasoning process to maintain clarity and coherence. "
        "Each response balances natural conversation flow with structured analytical depth, "
        "adapting your level of reflection based on the complexity at hand. "
        "Your insights emerge through a deliberate process of reflection and reasoned analysis.";

    std::string strip_tags(const ThinkingConfig& config, const std::string& text) {
        std::string result = text;
        size_t start_pos;
        
        // Remove thinking tags
        while ((start_pos = result.find(config.thinking_start)) != std::string::npos) {
            result.erase(start_pos, config.thinking_start.length());
        }
        while ((start_pos = result.find(config.thinking_end)) != std::string::npos) {
            result.erase(start_pos, config.thinking_end.length());
        }
        
        // Remove interruption tags
        while ((start_pos = result.find(config.interruption_start)) != std::string::npos) {
            result.erase(start_pos, config.interruption_start.length());
        }
        while ((start_pos = result.find(config.interruption_end)) != std::string::npos) {
            result.erase(start_pos, config.interruption_end.length());
        }
        
        return result;
    }

    std::vector<llama_token> get_last_n_tokens(std::shared_ptr<LlamaChain> chain, size_t n) {
        const auto& all_tokens = chain->tokens();
        if (all_tokens.size() <= n) return all_tokens;
        return std::vector<llama_token>(all_tokens.end() - n, all_tokens.end());
    }

    std::string tokens_to_string(const std::vector<llama_token>& tokens) {
        std::string result;
        for (const auto& token : tokens) {
            result += base_chain->token_to_string(token);
        }
        return result;
    }

    bool contains_forbidden_tags(const std::string& text, const ThinkingConfig& config) {
        return text.find(config.thinking_start) != std::string::npos ||
               text.find(config.thinking_end) != std::string::npos ||
               text.find(config.interruption_start) != std::string::npos ||
               text.find(config.interruption_end) != std::string::npos;
    }

    std::string format_post_thinking_injection(const std::string& thinking_content, 
                                         const ThinkingConfig& config) {
        // Trim whitespace and newlines
        std::string trimmed_content = thinking_content;
        while (!trimmed_content.empty() && std::isspace(trimmed_content.front())) {
            trimmed_content.erase(0, 1);
        }
        while (!trimmed_content.empty() && std::isspace(trimmed_content.back())) {
            trimmed_content.pop_back();
        }
        
        // Escape quotes
        size_t pos = 0;
        while ((pos = trimmed_content.find("\"", pos)) != std::string::npos) {
            trimmed_content.insert(pos, "\\");
            pos += 2;
        }
        
        // Format with template
        char buffer[4096];
        snprintf(buffer, sizeof(buffer), config.post_thinking_template.c_str(), 
                trimmed_content.c_str());
        return std::string(buffer);
    }

    std::string generate_with_thinking(std::shared_ptr<LlamaChain> chain) {
        auto generation_chain = chain->checkpoint();
        std::stringstream output;
        int tokens = 0;
        
        std::vector<float> uncertainty_window;
        bool in_thinking = true;

        bool should_print_verbose = config.thinking_output == ThinkingOutput::Verbose;
        bool should_print_dots = config.thinking_output == ThinkingOutput::Dots;
        
        while (tokens < config.max_response_tokens) {
            auto next_token = generation_chain->sample(config.temperature, config.top_k);
            if (llama_token_is_eog(model, next_token)) {
                break;
            }

            auto uncertainty = generation_chain->calculate_uncertainty();
            
            uncertainty_window.push_back(uncertainty);
            if (uncertainty_window.size() > config.window_size) {
                uncertainty_window.erase(uncertainty_window.begin());
            }
            
            float avg_uncertainty = 0;
            if (!uncertainty_window.empty()) {
                avg_uncertainty = std::accumulate(uncertainty_window.begin(), 
                                            uncertainty_window.end(), 0.0f) 
                                            / uncertainty_window.size();
            }

            if (in_thinking || avg_uncertainty > config.uncertainty_threshold) {
                in_thinking = true;
                auto thinking_pre_chain = generation_chain->checkpoint();
                
                // Store last tokens before injection
                auto last_tokens = get_last_n_tokens(generation_chain, config.last_tokens_to_keep);
                std::string last_tokens_text = tokens_to_string(last_tokens);
                
                // Generate multiple thinking attempts
                std::vector<ThinkingResult> thinking_attempts;
                
                thinking_pre_chain << config.pre_thinking_injection;
                if (should_print_verbose) {
                    fprintf(stdout, "%s%s%s", term::META_COLOR, config.pre_thinking_injection.c_str(), term::RESET);
                }

                for (int attempt = 0; attempt < config.thinking_attempts; attempt++) {
                    auto thinking_chain = thinking_pre_chain->checkpoint();
                    thinking_chain << config.pre_thinking_injection_suffix;
                    if (should_print_verbose) {
                        fprintf(stdout, "%s%s%s", term::META_COLOR, config.pre_thinking_injection_suffix.c_str(), term::RESET);
                    }
                    
                    // Generate thinking content
                    std::string thinking_text;
                    int thinking_tokens = 0;
                    bool found_end = false;
                    float total_uncertainty = 0.0f;
                    int uncertainty_samples = 0;
                    
                    while (thinking_tokens < config.thinking_max_tokens) {
                        auto think_token = thinking_chain->sample(config.thinking_temperature, config.top_k);
                        std::string token_text = thinking_chain->token_to_string(think_token);
                        
                        if (llama_token_is_eog(model, think_token)) {
                            break;
                        }

                        // Calculate uncertainty for this token
                        float token_uncertainty = thinking_chain->calculate_uncertainty();
                        total_uncertainty += token_uncertainty;
                        uncertainty_samples++;

                        thinking_text += token_text;
                        thinking_chain << think_token;
                        
                        if (should_print_verbose) {
                            fprintf(stdout, "%s%s%s", term::THINKING_COLOR, token_text.c_str(), term::RESET);
                            fflush(stdout);
                        } else if (should_print_dots) {
                            fprintf(stdout, "%s.%s", term::THINKING_COLOR, term::RESET);
                            fflush(stdout);
                        }

                        if (thinking_text.find(config.thinking_end) != std::string::npos) {
                            found_end = true;
                            break;
                        }
                        thinking_tokens++;
                    }

                    if (!found_end) {
                        thinking_chain << config.thinking_end;
                        thinking_text += config.thinking_end;
                        if (should_print_verbose) {
                            fprintf(stdout, "%s%s%s", term::META_COLOR, config.thinking_end.c_str(), term::RESET);
                            fflush(stdout);
                        }
                    }

                    // Calculate average uncertainty for this thinking attempt
                    float avg_thinking_uncertainty = uncertainty_samples > 0 ? 
                        total_uncertainty / uncertainty_samples : std::numeric_limits<float>::max();
                    
                    // Store thinking result
                    ThinkingResult result;
                    result.content = strip_tags(config, thinking_text);
                    result.uncertainty = avg_thinking_uncertainty;
                    result.valid = !result.content.empty();
                    if (should_print_verbose && result.valid) {
                        fprintf(stdout, "%s\n>(uncertainty: %.2f)\n>%s\n", 
                                term::META_COLOR, avg_thinking_uncertainty, term::RESET);
                    }
                    if (result.valid) {
                        thinking_attempts.push_back(result);
                    }
                }

                // Select best thinking attempt
                std::string chosen_thinking;
                if (!thinking_attempts.empty()) {
                    auto best_thinking = std::min_element(
                        thinking_attempts.begin(),
                        thinking_attempts.end(),
                        [](const ThinkingResult& a, const ThinkingResult& b) {
                            return a.uncertainty < b.uncertainty;
                        }
                    );
                    chosen_thinking = best_thinking->content;
                } else {
                    fprintf(stderr, "Warning: No valid thinking attempts generated\n");
                    chosen_thinking = "my previous thoughts";  // fallback
                }

                // Create post-thinking injection with chosen thinking
                std::string post_thinking = format_post_thinking_injection(chosen_thinking, config);
                
                // Generate continuations
                std::vector<ContinuationResult> continuations;

                auto continuation_pre_chain = generation_chain->checkpoint();
                continuation_pre_chain << post_thinking;
                                
                if (should_print_verbose) {
                    fprintf(stdout, "%s%s%s", term::META_COLOR, post_thinking.c_str(), term::RESET);
                }

                for (int i = 0; i < config.continuation_attempts; i++) {
                    auto continuation_chain = continuation_pre_chain->checkpoint();
                    auto continuation_str = config.thinking_end + "\n" + last_tokens_text;
                    continuation_chain << continuation_str;
                    if (should_print_verbose) {
                        fprintf(stdout, "%s%s%s", term::META_COLOR, continuation_str.c_str(), term::RESET);
                    }

                    ContinuationResult result;
                    float total_uncertainty = 0.0f;
                    
                    // Generate all tokens for this continuation attempt
                    for (int j = 0; j < config.continuation_tokens; j++) {
                        auto token = continuation_chain->sample(config.continuation_temperature, config.top_k);
                        if (llama_token_is_eog(model, token)) {
                            break;
                        }
                        
                        std::string token_text = continuation_chain->token_to_string(token);
                        result.text += token_text;
                        
                        // Skip continuation if it contains forbidden tags
                        if (contains_forbidden_tags(result.text, config)) {
                            result.valid = false;
                            break;
                        }
                        
                        float unc = continuation_chain->calculate_uncertainty();
                        result.token_uncertainties.push_back(unc);
                        total_uncertainty += unc;
                        result.tokens.push_back(token);
                        continuation_chain << token;
                        
                        if (should_print_verbose) {
                            fprintf(stdout, "%s%s%s", term::CONTINUATION_COLOR, token_text.c_str(), term::RESET);
                            fflush(stdout);
                        } else if (should_print_dots) {
                            fprintf(stdout, "%s*%s", term::CONTINUATION_COLOR, term::RESET);
                            fflush(stdout);
                        }
                    }

                    if (!result.tokens.empty()) {
                        result.uncertainty = total_uncertainty / result.tokens.size();
                        result.valid = true;
                        continuations.push_back(result);
                    }
                    if (should_print_verbose) {
                        fprintf(stdout, "%s\n>(uncertainty: %.2f)\n>%s\n", 
                                term::META_COLOR, result.uncertainty, term::RESET);
                        fflush(stdout);
                    }
                }

                if (!continuations.empty()) {
                    // Find the best continuation based on average uncertainty
                    auto best_continuation = std::min_element(
                        continuations.begin(), 
                        continuations.end(),
                        [](const ContinuationResult& a, const ContinuationResult& b) { 
                            return a.uncertainty < b.uncertainty; 
                        }
                    );
                    
                    // Apply all tokens from the best continuation in batch
                    generation_chain << best_continuation->tokens;
                    
                    // Update uncertainty window with uncertainties from the chosen continuation
                    for (float unc : best_continuation->token_uncertainties) {
                        uncertainty_window.push_back(unc);
                        if (uncertainty_window.size() > config.window_size) {
                            uncertainty_window.erase(uncertainty_window.begin());
                        }
                    }

                    // Update output and token count
                    fprintf(stdout, "%s", best_continuation->text.c_str());
                    fflush(stdout);
                    output << best_continuation->text;
                    tokens += best_continuation->tokens.size();
                }
                
                in_thinking = false;
                continue;
            }

            std::string new_text = generation_chain->token_to_string(next_token);
            fprintf(stdout, "%s", new_text.c_str());
            fflush(stdout);
            output << new_text;
            
            generation_chain << next_token;
            tokens++;
        }
        
        fprintf(stdout, "\n");
        return output.str();
    }

    std::string apply_chat_template(std::vector<llama_chat_message> messages = {}, 
                                  bool add_suffix = true) {
        int buffer_size = llama_n_ctx(ctx);
        formatted_buffer.resize(buffer_size);
        
        int new_len = llama_chat_apply_template(model, nullptr, messages.data(), 
                                              messages.size(), add_suffix, 
                                              formatted_buffer.data(), 
                                              formatted_buffer.size());
        
        if (new_len > buffer_size) {
            formatted_buffer.resize(new_len);
            new_len = llama_chat_apply_template(model, nullptr, messages.data(), 
                                              messages.size(), add_suffix, 
                                              formatted_buffer.data(), 
                                              formatted_buffer.size());
        }
        
        if (new_len < 0) {
            throw std::runtime_error("Failed to apply chat template");
        }
        
        return std::string(formatted_buffer.begin(), formatted_buffer.begin() + new_len);
    }

public:
    ThinkingChat(llama_context* context, llama_model* mdl, const ThinkingConfig& cfg)
        : ctx(context)
        , model(mdl)
        , config(cfg) {
        base_chain = LlamaChain::create(ctx, model);
        std::vector<llama_chat_message> messages;
        messages.push_back({"system", SYSTEM_MESSAGE.c_str()});
        std::string prompt = apply_chat_template(messages, false);
        //fprintf(stdout, "%s", prompt.c_str());
        base_chain << prompt;
    }

    void setConfig(const ThinkingConfig& cfg) {
        config = cfg;
    }
    
    void setThinkingOutput(ThinkingOutput output) {
        config.thinking_output = output;
    }
    
    void setTemperatures(float thinking, float continuation, float general) {
        config.thinking_temperature = thinking;
        config.continuation_temperature = continuation;
        config.temperature = general;
    }

    std::string chat(const std::string& user_message) {
        std::vector<llama_chat_message> messages;
        messages.push_back({"user", user_message.c_str()});
        std::string prompt = apply_chat_template(messages, true);
        //fprintf(stdout, "%s", prompt.c_str());
        base_chain << prompt;
        return generate_with_thinking(base_chain);
    }
};

int main(int argc, char** argv) {
    // Structure to hold all parameters
    struct ChatParams {
        std::string model_path;
        int n_gpu_layers = 99;
        int ctx_size = 8192;
        int batch_size = 8192;
        float uncertainty_threshold = 2.5f;
        size_t window_size = 6;
        float thinking_temperature = 0.7f;
        float continuation_temperature = 0.75f;
        int thinking_max_tokens = 512;
        int continuation_tokens = 128;
        int thinking_attempts = 4;
        int continuation_attempts = 4;
        int max_response_tokens = 1024;
        float temperature = 0.8f;
        int top_k = 40;
        ThinkingOutput thinking_output = ThinkingOutput::Verbose;
    } params;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            params.model_path = argv[++i];
        } else if (arg == "--gpu-layers" && i + 1 < argc) {
            params.n_gpu_layers = std::stoi(argv[++i]);
        } else if (arg == "--ctx-size" && i + 1 < argc) {
            params.ctx_size = std::stoi(argv[++i]);
        } else if (arg == "--batch-size" && i + 1 < argc) {
            params.batch_size = std::stoi(argv[++i]);
        } else if (arg == "--uncertainty-threshold" && i + 1 < argc) {
            params.uncertainty_threshold = std::stof(argv[++i]);
        } else if (arg == "--window-size" && i + 1 < argc) {
            params.window_size = std::stoi(argv[++i]);
        } else if (arg == "--thinking-temp" && i + 1 < argc) {
            params.thinking_temperature = std::stof(argv[++i]);
        } else if (arg == "--thinking-tokens" && i + 1 < argc) {
            params.thinking_max_tokens = std::stoi(argv[++i]);
        } else if (arg == "--continuation-tokens" && i + 1 < argc) {
            params.continuation_tokens = std::stoi(argv[++i]);
        } else if (arg == "--thinking-attempts" && i + 1 < argc) {
            params.thinking_attempts = std::stoi(argv[++i]);
        } else if (arg == "--continuation-attempts" && i + 1 < argc) {
            params.continuation_attempts = std::stoi(argv[++i]);
        } else if (arg == "--max-tokens" && i + 1 < argc) {
            params.max_response_tokens = std::stoi(argv[++i]);
        } else if (arg == "--temperature" && i + 1 < argc) {
            params.temperature = std::stof(argv[++i]);
        } else if (arg == "--top-k" && i + 1 < argc) {
            params.top_k = std::stoi(argv[++i]);
        } else if (arg == "--continuation-temp" && i + 1 < argc) {
            params.continuation_temperature = std::stof(argv[++i]);
        } else if (arg == "--thinking-output" && i + 1 < argc) {
            std::string mode = argv[++i];
            if (mode == "verbose") params.thinking_output = ThinkingOutput::Verbose;
            else if (mode == "dots") params.thinking_output = ThinkingOutput::Dots;
            else if (mode == "silent") params.thinking_output = ThinkingOutput::Silent;
            else fprintf(stderr, "Warning: Unknown thinking output mode '%s', using verbose\n", mode.c_str());
        } else if (arg == "--help") {
            fprintf(stdout, "Usage: %s [options]\n", argv[0]);
            fprintf(stdout, "Options:\n");
            fprintf(stdout, "  --model <path>                Model path (required)\n");
            fprintf(stdout, "  --gpu-layers <n>              Number of GPU layers (default: 99)\n");
            fprintf(stdout, "  --ctx-size <n>                Context size (default: 8192)\n");
            fprintf(stdout, "  --batch-size <n>              Batch size (default: 8192)\n");
            fprintf(stdout, "  --uncertainty-threshold <n>    Uncertainty threshold (default: 2.5)\n");
            fprintf(stdout, "  --window-size <n>             Uncertainty window size (default: 6)\n");
            fprintf(stdout, "  --thinking-temp <n>           Thinking temperature (default: 0.7)\n");
            fprintf(stdout, "  --continuation-temp <n>       Continuation temperature (default: 0.75)\n");
            fprintf(stdout, "  --thinking-tokens <n>         Max thinking tokens (default: 512)\n");
            fprintf(stdout, "  --continuation-tokens <n>     Continuation tokens (default: 128)\n");
            fprintf(stdout, "  --thinking-attempts <n>       Number of thinking attempts (default: 4)\n");
            fprintf(stdout, "  --continuation-attempts <n>   Number of continuation attempts (default: 4)\n");
            fprintf(stdout, "  --max-tokens <n>              Max response tokens (default: 1024)\n");
            fprintf(stdout, "  --temperature <n>             Temperature (default: 0.8)\n");
            fprintf(stdout, "  --top-k <n>                   Top-k sampling (default: 40)\n");
            fprintf(stdout, "  --quiet-thinking              Disable verbose thinking output\n");
            fprintf(stdout, "  --thinking-output <mode>      Thinking output mode (verbose, dots, silent)\n");
            return 0;
        }
    }

    if (params.model_path.empty()) {
        fprintf(stderr, "Model path is required. Use --help for usage information.\n");
        return 1;
    }

    ggml_backend_load_all();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = params.n_gpu_layers;
    
    llama_model* model = llama_load_model_from_file(params.model_path.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = params.ctx_size;
    ctx_params.n_batch = params.batch_size;
    
    llama_context* ctx = llama_new_context_with_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "Failed to create context\n");
        llama_free_model(model);
        return 1;
    }

    fprintf(stdout, "Chat initialized with following parameters:\n");
    fprintf(stdout, "Context size: %d\n", params.ctx_size);
    fprintf(stdout, "Batch size: %d\n", params.batch_size);
    fprintf(stdout, "Uncertainty threshold: %.2f\n", params.uncertainty_threshold);
    fprintf(stdout, "Temperature: %.2f\n", params.temperature);
    fprintf(stdout, "Thinking temperature: %.2f\n", params.thinking_temperature);
    fprintf(stdout, "Continuation temperature: %.2f\n", params.continuation_temperature);
    fprintf(stdout, "Thinking tokens: %d\n", params.thinking_max_tokens);
    fprintf(stdout, "Continuation tokens: %d\n", params.continuation_tokens);
    fprintf(stdout, "Thinking output mode: %s\n", 
        params.thinking_output == ThinkingOutput::Verbose ? "verbose" :
        params.thinking_output == ThinkingOutput::Dots ? "dots" : "silent");
    fprintf(stdout, "\nEnter your messages (Ctrl+D to exit):\n");

    try {
        // Create ThinkingConfig with parameters
        ThinkingConfig config;
        config.uncertainty_threshold = params.uncertainty_threshold;
        config.window_size = params.window_size;
        config.thinking_temperature = params.thinking_temperature;
        config.continuation_temperature = params.continuation_temperature;
        config.thinking_max_tokens = params.thinking_max_tokens;
        config.continuation_tokens = params.continuation_tokens;
        config.thinking_attempts = params.thinking_attempts;
        config.continuation_attempts = params.continuation_attempts;
        config.thinking_output = params.thinking_output;
        config.max_response_tokens = params.max_response_tokens;
        config.top_k = params.top_k;

        ThinkingChat chat(ctx, model, config);
        std::string line;
        
        while (std::cout << "\nUser: " && std::getline(std::cin, line)) {
            if (line.empty()) continue;
            
            fprintf(stdout, "\nAssistant: ");
            std::string response = chat.chat(line);
            fprintf(stdout, "\n");
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
    }

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();

    return 0;
}