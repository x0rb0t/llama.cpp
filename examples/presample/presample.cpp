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
    float low_prob_threshold = -5.0f;
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

    // Improved prompts with better structure
    const std::string SYSTEM_PROMPT = "\nI am a helpful AI assistant. I will analyze questions step by step.\n";
    const std::string QUESTION_PROMPT = "To answer this effectively, I'll break it down into key questions:\n";
    const std::string THINKING_PROMPT = "\nLet me think about each aspect in detail:\n";
    const std::string FINAL_PROMPT = "\nBased on my analysis, here is my clear and concise answer:\n";

    // Helper to clean text
    std::string clean_text(const std::string& text) {
        std::string cleaned = text;
        // Remove multiple spaces
        cleaned = std::regex_replace(cleaned, std::regex("\\s+"), " ");
        // Remove multiple newlines
        cleaned = std::regex_replace(cleaned, std::regex("\n+"), "\n");
        // Trim start and end
        cleaned = std::regex_replace(cleaned, std::regex("^\\s+|\\s+$"), "");
        return cleaned;
    }

    // Improved text generation with better state management
    std::string generate_safe(std::shared_ptr<LlamaChain> chain, 
                            int max_tokens,
                            float temp,
                            int top_k,
                            const std::vector<std::string>& stop_sequences = {}) {
        auto generation_chain = chain->checkpoint(LlamaChain::StringMode::SEPARATE);
        std::string generated;
        int tokens = 0;
        
        while (tokens < max_tokens) {
            auto next_token = generation_chain->sample(temp, top_k);
            
            if (llama_token_is_eog(model, next_token)) {
                break;
            }

            std::string new_text = generation_chain->token_to_string(next_token);

            // Check for stop sequences
            bool should_stop = false;
            for (const auto& stop : stop_sequences) {
                if (new_text.find(stop) != std::string::npos) {
                    should_stop = true;
                    break;
                }
            }
            
            // Update chain and text
            generation_chain << next_token;
            generated = generation_chain->string();
            tokens++;
        }
        fprintf(stdout, "[Generated text]: %s\n", generated.c_str());

        return clean_text(generated);
    }

    // Improved question generation
    std::vector<std::string> generate_questions(const std::string& input) {
        auto question_chain = base_chain->checkpoint();
        question_chain << QUESTION_PROMPT;
        
        std::string questions_text = generate_safe(question_chain, 
                                                 config.max_question_tokens,
                                                 0.7f,
                                                 30,
                                                 {"\n\n", "Next:", "Now"});

        std::vector<std::string> questions;
        std::stringstream ss(questions_text);
        std::string line;
        
        while (std::getline(ss, line)) {
            line = clean_text(line);
            if (line.empty()) continue;
            
            // Only process lines that look like questions
            if (line.find("1.") == 0 || 
                line.find("-") == 0 || 
                line.find("•") == 0) {
                // Remove leading markers and clean
                line = std::regex_replace(line, std::regex("^[-•\\d.]+\\s*"), "");
                if (!line.empty()) {
                    questions.push_back(line);
                }
            }
        }

        return questions;
    }

    // Improved thought generation
    std::vector<std::string> generate_thoughts(const std::vector<std::string>& questions) {
        std::vector<std::string> thoughts;
        auto thinking_chain = base_chain->checkpoint();
        thinking_chain << THINKING_PROMPT;

        for (const auto& question : questions) {
            // Create new checkpoint for each question
            auto question_chain = thinking_chain->checkpoint(LlamaChain::StringMode::SEPARATE);
            question_chain << "Question: " << question << "\nAnalysis: ";
            
            std::string thought = generate_safe(question_chain,
                                              config.max_thinking_tokens,
                                              config.temperature,
                                              config.top_k,
                                              {"\n\n", "Next:", "Question:"});
            
            if (!thought.empty()) {
                thoughts.push_back(clean_text(thought));
            }
        }

        return thoughts;
    }

    // Improved final answer generation
    std::string generate_final_answer(const std::vector<std::string>& thoughts) {
        auto answer_chain = base_chain->checkpoint();
        
        // Add thought summary if we have thoughts
        if (!thoughts.empty()) {
            answer_chain << "Based on my analysis:\n";
            for (const auto& thought : thoughts) {
                if (!thought.empty()) {
                    answer_chain << "- " << thought << "\n";
                }
            }
        }
        
        answer_chain << FINAL_PROMPT;
        
        return generate_safe(answer_chain,
                           config.max_answer_tokens,
                           config.temperature,
                           config.top_k,
                           {"\n\n", "Next:"});
    }

public:
    StagedGenerator(llama_context* context, llama_model* mdl, const GenerationConfig& conf)
        : ctx(context)
        , model(mdl)
        , config(conf) {
        base_chain = LlamaChain::create(ctx, model);
        base_chain << SYSTEM_PROMPT;
    }

    std::string process_prompt(const std::string& prompt) {
        try {
            // Initialize with user prompt
            base_chain << "\nQuestion: " << prompt << "\n";
            fprintf(stdout, "\n[Processing prompt]: %s\n", prompt.c_str());

            // Stage 1: Generate and analyze questions
            fprintf(stdout, "\n[Stage 1] Analyzing question...\n");
            auto questions = generate_questions(prompt);
            
            if (questions.empty()) {
                // If no questions generated, proceed with direct answer
                return generate_final_answer({});
            }

            // Print questions
            fprintf(stdout, "\n[Analysis Questions]:\n");
            for (const auto& question : questions) {
                fprintf(stdout, "- %s\n", question.c_str());
            }

            // Stage 2: Generate thoughts
            fprintf(stdout, "\n[Stage 2] Thinking about aspects...\n");
            auto thoughts = generate_thoughts(questions);
            
            if (thoughts.empty()) {
                // If no thoughts generated, proceed with direct answer
                return generate_final_answer({});
            }

            // Print thoughts
            fprintf(stdout, "\n[Thoughts]:\n");
            for (const auto& thought : thoughts) {
                fprintf(stdout, "- %s\n", thought.c_str());
            }

            // Stage 3: Generate final answer
            fprintf(stdout, "\n[Stage 3] Formulating final answer...\n");
            return generate_final_answer(thoughts);

        } catch (const std::exception& e) {
            fprintf(stderr, "Error in prompt processing: %s\n", e.what());
            return "Error occurred while processing the prompt.";
        }
    }
};

// Main function remains the same as in your current version

// Utility function to print usage
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

    // Parse command line arguments
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
            // Collect remaining args as prompt
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

    // Initialize llama backend
    ggml_backend_load_all();

    // Load model
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = config.n_gpu_layers;
    
    llama_model* model = llama_load_model_from_file(model_path.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "Failed to load model '%s'\n", model_path.c_str());
        return 1;
    }

    // Create context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;  // Adjust based on your needs
    ctx_params.n_batch = 512;
    
    llama_context* ctx = llama_new_context_with_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "Failed to create context\n");
        llama_free_model(model);
        return 1;
    }

    fprintf(stdout, "Model loaded successfully\n");

    try {
        // Create generator and process prompt
        StagedGenerator generator(ctx, model, config);
        std::string result = generator.process_prompt(prompt);

        // Print final result
        fprintf(stdout, "\n[Final Response]:\n%s\n", result.c_str());

    } catch (const std::exception& e) {
        fprintf(stderr, "Error during generation: %s\n", e.what());
    }

    // Cleanup
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();

    return 0;
}