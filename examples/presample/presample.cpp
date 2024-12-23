#include "llama_chain.hpp"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <sstream>
#include <unistd.h>
#include <memory>
#include <stdexcept>
#include <regex>

// Structured configuration
struct GenerationConfig {
    float temperature = 0.8f;
    int top_k = 40;
    int n_predict = 512;
    int n_gpu_layers = 99;
    int max_question_tokens = 150;
    int max_thinking_tokens = 200;
    int max_answer_tokens = 300;
};

class StagedGenerator {
private:
    llama_context* ctx;
    llama_model* model;
    std::shared_ptr<LlamaChain> base_chain;
    GenerationConfig config;
    std::vector<llama_chat_message> messages;
    std::vector<char> formatted_buffer;
    int prev_len = 0;

    // Enhanced prompts for better structure
    const std::string SYSTEM_MESSAGE = R"(You are a helpful AI assistant that reasons about a problem. For every question:
1. First, break it down into clear, numbered key points for analysis
2. Then, analyze each point thoroughly and systematically
3. Finally, provide a clear conclusion based on your analysis.
Always start each point on a new line and number them clearly.)";

    const std::string QUESTION_ROLE = R"(<reasoning>I will break down this question into key points for analysis.
Each point will be numbered and start on a new line:

)";

    const std::string THINKING_ROLE = R"(<reasoning>Let me analyze each point systematically:

)";

    const std::string ANSWER_ROLE = R"(<reasoning>Based on my thorough analysis, here is my precise answer:

)";

    // Helper to clean text
    std::string clean_text(const std::string& text) {
        std::string cleaned = text;
        // Remove excessive whitespace while preserving single newlines
        cleaned = std::regex_replace(cleaned, std::regex("[ \t]+"), " ");
        cleaned = std::regex_replace(cleaned, std::regex("\n{3,}"), "\n\n");
        cleaned = std::regex_replace(cleaned, std::regex("^ +| +$"), "");
        return cleaned;
    }

    // Enhanced chat template application
    std::string apply_chat_template(const std::vector<llama_chat_message>& msgs) {
        int buffer_size = llama_n_ctx(ctx);
        formatted_buffer.resize(buffer_size);
        
        int new_len = llama_chat_apply_template(model, nullptr, msgs.data(), msgs.size(), 
                                              true, formatted_buffer.data(), formatted_buffer.size());
        
        if (new_len > buffer_size) {
            formatted_buffer.resize(new_len);
            new_len = llama_chat_apply_template(model, nullptr, msgs.data(), msgs.size(), 
                                              true, formatted_buffer.data(), formatted_buffer.size());
        }
        
        if (new_len < 0) {
            throw std::runtime_error("Failed to apply chat template");
        }
        
        std::string prompt(formatted_buffer.begin() + prev_len, formatted_buffer.begin() + new_len);
        prev_len = new_len;
        return prompt;
    }

    // Improved text generation with real-time output
    std::string generate_safe(std::shared_ptr<LlamaChain> chain, 
                            int max_tokens,
                            float temp,
                            int top_k,
                            const std::vector<std::string>& stop_sequences = {}) {
        auto generation_chain = chain->checkpoint(LlamaChain::StringMode::SEPARATE);
        std::stringstream output;
        int tokens = 0;
        
        while (tokens < max_tokens) {
            auto next_token = generation_chain->sample(temp, top_k);
            
            if (llama_token_is_eog(model, next_token)) {
                break;
            }

            std::string new_text = generation_chain->token_to_string(next_token);
            
            // Check for stop sequences
            bool should_stop = false;
            std::string current_output = output.str() + new_text;
            for (const auto& stop : stop_sequences) {
                if (current_output.find(stop) != std::string::npos) {
                    should_stop = true;
                    break;
                }
            }
            if (should_stop) break;
            
            fprintf(stdout, "%s", new_text.c_str());
            fflush(stdout);
            output << new_text;
            
            generation_chain << next_token;
            tokens++;
        }
        
        fprintf(stdout, "\n");
        return clean_text(output.str());
    }

    // Enhanced question generation with better parsing
    std::vector<std::string> generate_questions(const std::string& input) {
        messages.clear();
        messages.push_back({"system", SYSTEM_MESSAGE.c_str()});
        messages.push_back({"user", input.c_str()});
        
        std::string prompt = apply_chat_template(messages) + QUESTION_ROLE;
        
        auto question_chain = base_chain->checkpoint();
        question_chain << prompt;
        
        std::string questions_text = generate_safe(question_chain,
                                                 config.max_question_tokens,
                                                 0.7f,
                                                 30,
                                                 {"Therefore", "Thus", "In conclusion"});

        std::vector<std::string> questions;
        std::stringstream ss(questions_text);
        std::string line;
        std::string current_question;
        bool in_question = false;
        
        while (std::getline(ss, line)) {
            line = clean_text(line);
            if (line.empty()) continue;
            
            std::smatch match;
            if (std::regex_match(line, match, std::regex("^(\\d+\\.|-|•)\\s*(.+)$"))) {
                // Save previous question if exists
                if (in_question && !current_question.empty()) {
                    questions.push_back(clean_text(current_question));
                }
                
                // Start new question with the content after the marker
                current_question = match[2].str();
                in_question = true;
            } else if (in_question) {
                // Continuation of current question
                current_question += " " + line;
            }
        }
        
        // Don't forget the last question
        if (in_question && !current_question.empty()) {
            questions.push_back(clean_text(current_question));
        }

        return questions;
    }

    // Enhanced thought generation with better structure
    std::vector<std::string> generate_thoughts(const std::vector<std::string>& questions) {
        std::vector<std::string> thoughts;
        messages.push_back({"assistant", "Let me analyze these points:"});
        
        for (size_t i = 0; i < questions.size(); ++i) {
            std::string analysis_prompt = apply_chat_template(messages) + 
                                        THINKING_ROLE + 
                                        "Point " + std::to_string(i + 1) + ": " + questions[i] + "\n\nAnalysis:\n";
            
            auto thinking_chain = base_chain->checkpoint();
            thinking_chain << analysis_prompt;
            
            std::string thought = generate_safe(thinking_chain,
                                              config.max_thinking_tokens,
                                              config.temperature,
                                              config.top_k,
                                              {"\n\n", "Next point:", "Point:", "In conclusion"});
            
            if (!thought.empty()) {
                thoughts.push_back(clean_text(thought));
                messages.push_back({"assistant", thought.c_str()});
            }
        }

        return thoughts;
    }

    // Enhanced final answer generation
    std::string generate_final_answer(const std::vector<std::string>& thoughts) {
        std::string analysis_summary;
        if (!thoughts.empty()) {
            analysis_summary = "Based on my analysis of all points:\n\n";
            for (size_t i = 0; i < thoughts.size(); ++i) {
                analysis_summary += std::to_string(i + 1) + ". " + thoughts[i] + "\n\n";
            }
            analysis_summary += "Therefore, my final conclusion is:\n";
        }
        
        messages.push_back({"assistant", "Let me provide my final answer."});
        std::string prompt = apply_chat_template(messages) + ANSWER_ROLE + analysis_summary;
        
        auto answer_chain = base_chain->checkpoint();
        answer_chain << prompt;
        
        return generate_safe(answer_chain,
                           config.max_answer_tokens,
                           config.temperature,
                           config.top_k,
                           {"\n\nNext", "\n\nQuestion:"});
    }

public:
    StagedGenerator(llama_context* context, llama_model* mdl, const GenerationConfig& conf)
        : ctx(context)
        , model(mdl)
        , config(conf) {
        base_chain = LlamaChain::create(ctx, model);
        formatted_buffer.reserve(llama_n_ctx(ctx));
    }

    std::string process_prompt(const std::string& prompt) {
        try {
            messages.clear();
            prev_len = 0;
            
            fprintf(stdout, "\n[Processing prompt]: %s\n", prompt.c_str());

            // Stage 1: Generate and analyze questions
            fprintf(stdout, "\n[Stage 1] Breaking down the question...\n");
            auto questions = generate_questions(prompt);
            
            if (questions.empty()) {
                fprintf(stdout, "No analysis points generated. Providing direct answer.\n");
                return generate_final_answer({});
            }

            fprintf(stdout, "\n[Analysis Points:]\n");
            for (size_t i = 0; i < questions.size(); ++i) {
                fprintf(stdout, "%zu. %s\n", i + 1, questions[i].c_str());
            }

            // Stage 2: Generate thoughts
            fprintf(stdout, "\n[Stage 2] Analyzing each point...\n");
            auto thoughts = generate_thoughts(questions);
            
            if (thoughts.empty()) {
                fprintf(stdout, "No analysis generated. Providing direct answer.\n");
                return generate_final_answer({});
            }

            fprintf(stdout, "\n[Analysis Results:]\n");
            for (size_t i = 0; i < thoughts.size(); ++i) {
                fprintf(stdout, "Point %zu: %s\n", i + 1, thoughts[i].c_str());
            }

            // Stage 3: Generate final answer
            fprintf(stdout, "\n[Stage 3] Forming final conclusion...\n");
            return generate_final_answer(thoughts);

        } catch (const std::exception& e) {
            fprintf(stderr, "Error in prompt processing: %s\n", e.what());
            return "Error occurred while processing the prompt.";
        }
    }
};

// Main function implementation remains the same
static void print_usage(const char* prog) {
    fprintf(stderr, "Usage: %s -m <model_path> [-t <temperature>] [-k <top_k>] [-n <n_predict>] [prompt]\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -m <path>        Model path (required)\n");
    fprintf(stderr, "  -t <float>       Temperature (default: 0.8)\n");
    fprintf(stderr, "  -k <int>         Top-k sampling (default: 40)\n");
    fprintf(stderr, "  -n <int>         Number of tokens to predict (default: 512)\n");
    fprintf(stderr, "  -ngl <int>       Number of GPU layers (default: 99)\n");
}

int main(int argc, char** argv) {
    std::string model_path;
    std::string prompt = "What are the key considerations for implementing a secure authentication system?";
    GenerationConfig config;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            config.temperature = std::stof(argv[++i]);
        } else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) {
            config.top_k = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            config.n_predict = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) {
            config.n_gpu_layers = std::stoi(argv[++i]);
        } else {
            prompt = argv[i];
            while (++i < argc) {
                prompt += " " + std::string(argv[i]);
            }
            break;
        }
    }

    if (model_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    ggml_backend_load_all();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = config.n_gpu_layers;
    
    llama_model* model = llama_load_model_from_file(model_path.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "Failed to load model '%s'\n", model_path.c_str());
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;
    ctx_params.n_batch = 512;
    
    llama_context* ctx = llama_new_context_with_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "Failed to create context\n");
        llama_free_model(model);
        return 1;
    }

    fprintf(stdout, "Model loaded successfully\n");

    try {
        StagedGenerator generator(ctx, model, config);
        std::string result = generator.process_prompt(prompt);
        fprintf(stdout, "\n[Final Response]:\n%s\n", result.c_str());
    } catch (const std::exception& e) {
        fprintf(stderr, "Error during generation: %s\n", e.what());
    }

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();

    return 0;
}