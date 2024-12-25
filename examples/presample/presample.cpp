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

struct ThinkingConfig {
    float uncertainty_threshold = 2.5f;
    int window_size = 6;
    float thinking_temperature = 0.7f;
    int thinking_max_tokens = 512;
    int continuation_tokens = 128;
    int thinking_attempts = 4;
    int continuation_attempts = 4;
    int last_tokens_to_keep = 16;
    
    // Tags for thinking process
    const std::string interruption_start = "<INTERRUPTION>";
    const std::string interruption_end = "</INTERRUPTION>";
    const std::string thinking_start = "<THINKING>";
    const std::string thinking_end = "</THINKING>";

    // Pre-thinking injection message
    const std::string pre_thinking_injection = 
        "<INTERRUPTION>\n"
        "Nexus pauses to think privately about the immediate context (last part of conversation). "
        "Looking at both the user's most recent input and Nexus's last response fragment (if available), "
        "Nexus analyzes carefully with <THINKING> tags, challenging assumptions and "
        "ensuring logical consistency. Analysis must be grounded in this recent context "
        "and close with </THINKING> before continuing:\n"
        "</INTERRUPTION>\n"
        "<THINKING>\n"
        "Analyzing the immediate conversation context:\n";
        
    const std::string pre_thinking_injection_suffix = 
        "1.";

    const std::string post_thinking_template = 
        "<INTERRUPTION>\n"
        "Having analyzed the recent context:\n"
        "1. %s\n"
        "Nexus will continue the response naturally, maintaining coherence with "
        "both this analysis and the preceding conversation:\n";
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
    std::shared_ptr<LlamaChain> base_chain;
    std::vector<char> formatted_buffer;
    
    // Simple system message without thinking mechanics
    const std::string SYSTEM_MESSAGE = 
    "You are Nexus, a thoughtful AI assistant who deeply analyzes each response. "
    "Nexus pauses to reason step-by-step, critically examines assumptions, "
    "and synthesizes insights before continuing with precise, well-structured answers.";

    std::string strip_tags(const ThinkingConfig& config, const std::string& text) {
        std::string result = text;
        size_t start_pos, end_pos;
        
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

    std::vector<llama_token> get_last_n_tokens(std::shared_ptr<LlamaChain> chain, int n) {
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

    std::string generate_with_thinking(std::shared_ptr<LlamaChain> chain, 
                                     int max_tokens,
                                     float temp,
                                     int top_k,
                                     const ThinkingConfig& config = ThinkingConfig()) {
        auto generation_chain = chain->checkpoint();
        std::stringstream output;
        int tokens = 0;
        
        std::vector<float> uncertainty_window;
        bool in_thinking = true;

        bool do_print_think = true;

        
        while (tokens < max_tokens) {
            auto next_token = generation_chain->sample(temp, top_k);
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
                if (do_print_think) {
                    fprintf(stdout, "%s%s%s", term::META_COLOR, config.pre_thinking_injection.c_str(), term::RESET);
                }
                for (int attempt = 0; attempt < config.thinking_attempts; attempt++) {
                    auto thinking_chain = thinking_pre_chain->checkpoint();
                    thinking_chain << config.pre_thinking_injection_suffix;
                    if (do_print_think) {
                        fprintf(stdout, "%s%s%s", term::META_COLOR, config.pre_thinking_injection_suffix.c_str(), term::RESET);
                    }
                    // Generate thinking content
                    std::string thinking_text;
                    int thinking_tokens = 0;
                    bool found_end = false;
                    float total_uncertainty = 0.0f;
                    int uncertainty_samples = 0;
                    
                    while (thinking_tokens < config.thinking_max_tokens) {
                        auto think_token = thinking_chain->sample(config.thinking_temperature * temp, top_k);
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
                        
                        if (do_print_think) {
                            fprintf(stdout, "%s%s%s", term::THINKING_COLOR, token_text.c_str(), term::RESET);
                            fflush(stdout);
                        } else {
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
                        if (do_print_think) {
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
                    if (do_print_think && result.valid) {
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
                                
                if (do_print_think) {
                    fprintf(stdout, "%s%s%s", term::META_COLOR, post_thinking.c_str(), term::RESET);
                }


                for (int i = 0; i < config.continuation_attempts; i++) {
                    auto continuation_chain = continuation_pre_chain->checkpoint();
                    auto continuation_str = config.thinking_end + "\n" + last_tokens_text;
                    continuation_chain << continuation_str;
                    if (do_print_think) {
                        fprintf(stdout, "%s%s%s", term::META_COLOR, continuation_str.c_str(), term::RESET);
                    }

                    ContinuationResult result;
                    float total_uncertainty = 0.0f;
                    
                    // Generate all tokens for this continuation attempt
                    for (int j = 0; j < config.continuation_tokens; j++) {
                        auto token = continuation_chain->sample(temp, top_k);
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
                        if (!do_print_think) {
                            fprintf(stdout, "%s*%s", term::CONTINUATION_COLOR, term::RESET);
                            fflush(stdout);
                        } else {
                            fprintf(stdout, "%s%s%s", term::CONTINUATION_COLOR, token_text.c_str(), term::RESET);
                            fflush(stdout);
                        }

                    }
                    if (!result.tokens.empty()) {
                        result.uncertainty = total_uncertainty / result.tokens.size();
                        result.valid = true;
                        continuations.push_back(result);
                    }
                    if (do_print_think) {
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
                //clear uncertainty window
                //uncertainty_window.clear();
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
    ThinkingChat(llama_context* context, llama_model* mdl) 
        : ctx(context)
        , model(mdl) {
        base_chain = LlamaChain::create(ctx, model);
        std::vector<llama_chat_message> messages;
        messages.push_back({"system", SYSTEM_MESSAGE.c_str()});
        std::string prompt = apply_chat_template(messages, false);
        //fprintf(stdout, "%s", prompt.c_str());
        base_chain << prompt;
    }

    std::string chat(const std::string& user_message) {
        std::vector<llama_chat_message> messages;
        messages.push_back({"user", user_message.c_str()});
        std::string prompt = apply_chat_template(messages, true);
        //fprintf(stdout, "%s", prompt.c_str());
        base_chain << prompt;
        return generate_with_thinking(base_chain, 1024, 0.8f, 40);
    }
};

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model_path>\n", argv[0]);
        return 1;
    }

    std::string model_path = argv[1];
    
    ggml_backend_load_all();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 99;
    
    llama_model* model = llama_load_model_from_file(model_path.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 8192;
    ctx_params.n_batch = 8192;
    
    llama_context* ctx = llama_new_context_with_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "Failed to create context\n");
        llama_free_model(model);
        return 1;
    }

    fprintf(stdout, "Chat initialized. Enter your messages (Ctrl+D to exit):\n");

    try {
        ThinkingChat chat(ctx, model);
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