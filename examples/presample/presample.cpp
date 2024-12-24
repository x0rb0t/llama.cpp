#include "llama_chain.hpp"
#include "llama.h"

#include <cstdio>
#include <string>
#include <vector>
#include <iostream>
#include <memory>
#include <sstream>

// ANSI escape codes for terminal manipulation
namespace term {
    const char* CLEAR_LINE = "\033[2K\r";
    const char* MOVE_UP = "\033[1A";
    const char* SAVE_CURSOR = "\033[s";
    const char* RESTORE_CURSOR = "\033[u";
    const char* BLUE = "\033[34m";
    const char* GREEN = "\033[32m";
    const char* RESET = "\033[0m";
}

struct ThinkingConfig {
    float uncertainty_threshold = 5.0f;
    int window_size = 5;
    float thinking_temperature = 0.7f;
    int thinking_max_tokens = 100;
    int continuation_tokens = 5;
    int continuation_attempts = 5;
    
    std::string thinking_start = "<thinking>Let me reason about this...";
    
    std::string thinking_end = "</thinking>";
};

class ThinkingChat {
private:
    llama_context* ctx;
    llama_model* model;
    std::shared_ptr<LlamaChain> base_chain;
    std::vector<llama_chat_message> messages;
    std::vector<char> formatted_buffer;
    int prev_len = 0;

    // Single SYSTEM_MESSAGE definition with improved content
    const std::string SYSTEM_MESSAGE = R"(You are Zero, an AI that can reason through uncertainty. Important: The system will automatically inject <thinking> tags when uncertainty is detected - you should never add these tags yourself. When you see a thinking tag opened by the system:
        1. Reason through your uncertainty
        2. Close the tag with </thinking> when done reasoning
        3. Continue your response naturally as if the thinking section never happened

        Remember: Thinking tags are for your internal reasoning process only. After closing the tag, just continue your normal response flow. Let's start chatting!)";

    std::string apply_chat_template() {
        int buffer_size = llama_n_ctx(ctx);
        formatted_buffer.resize(buffer_size);
        
        int new_len = llama_chat_apply_template(model, nullptr, messages.data(), messages.size(), 
                                              true, formatted_buffer.data(), formatted_buffer.size());
        
        if (new_len > buffer_size) {
            formatted_buffer.resize(new_len);
            new_len = llama_chat_apply_template(model, nullptr, messages.data(), messages.size(), 
                                              true, formatted_buffer.data(), formatted_buffer.size());
        }
        
        if (new_len < 0) {
            throw std::runtime_error("Failed to apply chat template");
        }
        
        std::string prompt(formatted_buffer.begin() + prev_len, formatted_buffer.begin() + new_len);
        prev_len = new_len;
        return prompt;
    }

    std::string generate_with_thinking(std::shared_ptr<LlamaChain> chain, 
                                     int max_tokens,
                                     float temp,
                                     int top_k,
                                     const ThinkingConfig& config = ThinkingConfig()) {
        auto generation_chain = chain->checkpoint(LlamaChain::StringMode::SEPARATE);
        std::stringstream output;
        int tokens = 0;
        
        std::vector<float> uncertainty_window;
        bool in_thinking = false;
        
        // Save initial cursor position for output management
        fprintf(stdout, "%s", term::SAVE_CURSOR);
        
        while (tokens < max_tokens) {
            auto next_token = generation_chain->sample(temp, top_k);
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

            if (!in_thinking && avg_uncertainty > config.uncertainty_threshold) {
                in_thinking = true;
                
                // Save position before thinking output
                //fprintf(stdout, "%s", term::SAVE_CURSOR);
                
                auto thinking_chain = generation_chain->checkpoint();
                thinking_chain << config.thinking_start;
                fprintf(stdout, "%s", config.thinking_start.c_str());
                
                std::string thinking_text;
                int thinking_tokens = 0;
                bool found_end = false;
                
                while (thinking_tokens < config.thinking_max_tokens) {
                    auto think_token = thinking_chain->sample(config.thinking_temperature * temp, top_k);
                    std::string token_text = thinking_chain->token_to_string(think_token);
                    thinking_text += token_text;
                    thinking_chain << think_token;
                    fprintf(stdout, "%s", token_text.c_str());
                    fflush(stdout);
                    
                    if (llama_token_is_eog(model, think_token)) {
                        found_end = false;
                        break;
                    }
                    if (thinking_text.find(config.thinking_end) != std::string::npos) {
                        found_end = true;
                        break;
                    }
                    thinking_tokens++;
                }
                
                if (!found_end) {
                    thinking_chain << config.thinking_end;
                    fprintf(stdout, "%s", config.thinking_end.c_str());
                }
                
                // Sample continuations with better formatting
                std::vector<std::pair<std::vector<llama_token>, float>> continuations;
                //fprintf(stdout, "\n%sAnalyzing possible continuations:%s\n", term::GREEN, term::RESET);
                
                for (int i = 0; i < config.continuation_attempts; i++) {
                    //fprintf(stdout, "%s[%d]%s ", term::GREEN, i + 1, term::RESET);
                    auto continuation_chain = thinking_chain->checkpoint();
                    std::vector<llama_token> cont_tokens;
                    float total_uncertainty = 0.0f;
                    
                    for (int j = 0; j < config.continuation_tokens; j++) {
                        auto token = continuation_chain->sample(temp, top_k);
                        std::string token_text = thinking_chain->token_to_string(token);
                        float unc = continuation_chain->calculate_uncertainty();
                        total_uncertainty += unc;
                        cont_tokens.push_back(token);
                        continuation_chain << token;
                        //fprintf(stdout, "%s", token_text.c_str());
                    }
                    //fprintf(stdout, " (uncertainty: %.3f)\n", total_uncertainty / config.continuation_tokens);
                    
                    continuations.push_back({cont_tokens, total_uncertainty / config.continuation_tokens});
                }
                
                auto best_continuation = std::min_element(
                    continuations.begin(), 
                    continuations.end(),
                    [](const auto& a, const auto& b) { return a.second < b.second; }
                );
                
                // Restore cursor and clear thinking output
                //fprintf(stdout, "%s", term::RESTORE_CURSOR);
                /*for (int i = 0; i < thinking_tokens + config.continuation_attempts + 3; i++) {
                    fprintf(stdout, "%s%s", term::CLEAR_LINE, term::MOVE_UP);
                }
                fprintf(stdout, "%s", term::CLEAR_LINE);*/
                
                // Apply best continuation
                for (const auto& token : best_continuation->first) {
                    if (llama_token_is_eog(model, token)) {
                        break;
                    }
                    generation_chain << token;
                    std::string token_text = generation_chain->token_to_string(token);
                    fprintf(stdout, "%s", token_text.c_str());
                    output << token_text;
                    tokens++;
                }
                
                in_thinking = false;
                continue;
            }

            if (llama_token_is_eog(model, next_token)) {
                break;
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

public:
    ThinkingChat(llama_context* context, llama_model* mdl) 
        : ctx(context)
        , model(mdl) {
        base_chain = LlamaChain::create(ctx, model);
        messages.push_back({"system", SYSTEM_MESSAGE.c_str()});
    }

    std::string chat(const std::string& user_message) {
        messages.push_back({"user", user_message.c_str()});
        std::string prompt = apply_chat_template();
        
        auto response_chain = base_chain->checkpoint();
        response_chain << prompt;
        
        std::string response = generate_with_thinking(response_chain, 1024, 0.8f, 40);
        messages.push_back({"assistant", response.c_str()});
        
        return response;
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
    ctx_params.n_batch = 512;
    
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