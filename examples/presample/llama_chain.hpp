#pragma once

#include <memory>
#include <string>
#include <vector>
#include <optional>
#include <cassert>
#include <stdexcept>
#include <cmath>
#include <random>
#include <array>
#include <atomic>
#include <algorithm>

#include "llama.h"

// Custom exceptions
class LlamaChainException : public std::runtime_error {
public:
    explicit LlamaChainException(const std::string& msg) : std::runtime_error(msg) {}
};

class TokenizationError : public LlamaChainException {
public:
    explicit TokenizationError(const std::string& msg) : LlamaChainException(msg) {}
};

class DecodingError : public LlamaChainException {
public:
    explicit DecodingError(const std::string& msg) : LlamaChainException(msg) {}
};

// Forward declarations
class LlamaChain;
class LlamaChainRollbackGuard;
using LlamaChainPtr = std::shared_ptr<LlamaChain>;
using LogProbResult = std::vector<float>;

class LlamaChain : public std::enable_shared_from_this<LlamaChain> {
public:
    LogProbResult token_logprobs;

    enum class StringMode {
        CONCAT,     // Concatenate with parent's string
        SEPARATE    // Start new string from this point
    };

    // Public constructor to allow make_shared access
    LlamaChain(llama_context* ctx, llama_model* model, 
               std::shared_ptr<LlamaChain> parent = nullptr,
               StringMode mode = StringMode::CONCAT)
        : ctx_(ctx)
        , model_(model)
        , parent_(parent)
        , string_mode_(mode)
        , kv_cache_token_count_start_(llama_get_kv_cache_token_count(ctx))
        , vocab_size_(llama_n_vocab(model))
        , is_rolling_back_(false)
    {
        // Pre-allocate buffers
        logprobs_buffer_.reserve(vocab_size_);
        tokens_.reserve(1024);           // Initial reasonable size
        accumulated_text_.reserve(4096); // Initial reasonable size
    }

    static LlamaChainPtr create(llama_context* ctx, llama_model* model) {
        return std::make_shared<LlamaChain>(ctx, model);
    }

    // Convert a single token to string with bounds checking
    std::string token_to_string(llama_token token) const {
        std::array<char, 256> buf;
        int n = llama_token_to_piece(model_, token, buf.data(), buf.size(), false, true);
        if (n <= 0) {
            return "";
        }
        if (n >= static_cast<int>(buf.size())) {
            throw LlamaChainException("Buffer overflow prevented in token_to_string");
        }
        return std::string(buf.data(), n);
    }

    // Create a checkpoint (child chain)
    LlamaChainPtr checkpoint(StringMode mode = StringMode::CONCAT) {
        if (active_child_) {
            cache_active_child_logprobs();
            active_child_->rollback();
            active_child_.reset();
        }

        auto child = std::make_shared<LlamaChain>(ctx_, model_, shared_from_this(), mode);
        active_child_ = child;
        (*child) << cached_logprobs_;
        return child;
    }

    // Token addition
    LlamaChain& operator<<(llama_token token) {
        deactivate_siblings();
        add_token(token);
        update_logprobs();
        return *this;
    }

    // String addition
    LlamaChain& operator<<(const std::string& text) {
        deactivate_siblings();
        add_string(text);
        update_logprobs();
        return *this;
    }

    // Multiple tokens addition
    LlamaChain& operator<<(const std::vector<llama_token>& tokens) {
        deactivate_siblings();
        add_tokens(tokens);
        update_logprobs();
        return *this;
    }

    // Get cached logprobs
    LogProbResult get_logprobs() const {
        return cached_logprobs_;
    }

    // Operator for getting logprobs
    friend const LogProbResult& operator>>(const LlamaChain& chain, LogProbResult& result) {
        result = chain.cached_logprobs_;
        return result;
    }

    // Set logprobs
    friend LlamaChain& operator<<(LlamaChain& chain, const LogProbResult& logprobs) {
        chain.cached_logprobs_ = logprobs;
        return chain;
    }

    // Get accumulated string
    std::string string() const {
        if (string_mode_ == StringMode::SEPARATE) {
            return accumulated_text_;
        }

        std::string result;
        // Get parent string first without any locks
        if (auto parent = parent_.lock()) {
            result = parent->string();
        }

        // Then get our text
        result += accumulated_text_;
        return result;
    }

    // Get tokens
    std::vector<llama_token> tokens() const {
        return tokens_;
    }

    // Rollback
    void rollback() {
        if (is_rolling_back_) {
            // Prevent recursive rollback
            return;
        }
        is_rolling_back_ = true;

        if (active_child_ && active_child_.get() != this) {
            cache_active_child_logprobs();
            active_child_->rollback();
            active_child_.reset();
        }

        if (auto parent_ptr = parent_.lock()) {
            remove_tokens_from_kv_cache();
        }

        is_rolling_back_ = false;
    }

    virtual ~LlamaChain() = default;

    // Sample next token with temperature and top-k
    llama_token sample(float temp, int top_k) {
        if (temp <= 0.0f) {
            return greedy_sample();
        }

        LogProbResult logits = get_logprobs();
        std::vector<llama_token_data> candidates;
        candidates.reserve(vocab_size_);

        for (llama_token token_id = 0; token_id < vocab_size_; token_id++) {
            candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
        }

        // Ensure 'size' matches the expected type in llama_token_data_array
        llama_token_data_array cur_p = {
            candidates.data(),
            static_cast<size_t>(candidates.size()), // size cast to size_t
            -1,
            false
        };

        // Apply temperature with numerical stability check
        constexpr float max_logit = 100.0f;  // Prevent overflow
        for (size_t i = 0; i < cur_p.size; ++i) { // using size_t as per struct definition
            cur_p.data[i].logit = std::min(cur_p.data[i].logit / temp, max_logit);
        }

        // Apply top-k
        top_k = (top_k <= 0) ? cur_p.size : std::min(top_k, static_cast<int>(cur_p.size));

        std::partial_sort(
            cur_p.data,
            cur_p.data + top_k,
            cur_p.data + cur_p.size,
            [](const llama_token_data& a, const llama_token_data& b) {
                return a.logit > b.logit;
            }
        );

        cur_p.size = top_k;

        // Compute softmax probabilities with improved numerical stability
        float max_logit_val = cur_p.data[0].logit;
        double sum_exp = 0.0;

        std::vector<double> scaled_exp(top_k);
        for (int i = 0; i < top_k; ++i) {
            scaled_exp[i] = std::exp(double(cur_p.data[i].logit - max_logit_val));
            sum_exp += scaled_exp[i];
        }

        // Normalize probabilities
        for (int i = 0; i < top_k; ++i) {
            cur_p.data[i].p = static_cast<float>(scaled_exp[i] / sum_exp);
        }

        // Sample using thread-safe random generator
        static thread_local std::random_device rd;
        static thread_local std::mt19937 gen(rd());
        std::discrete_distribution<> dist(scaled_exp.begin(), scaled_exp.end());

        return cur_p.data[dist(gen)].id;
    }

    double calculate_uncertainty() {
        if (vocab_size_ <= 0) {
            throw std::runtime_error("Invalid vocabulary size");
        }
        
        double entropy = 0.0;
        double max_logp = *std::max_element(cached_logprobs_.begin(), cached_logprobs_.end());
        
        for (int i = 0; i < vocab_size_; i++) {
            double logp = cached_logprobs_[i];
            if (std::isfinite(logp)) {
                // Subtract max_logp for numerical stability
                double scaled_logp = logp - max_logp;
                double p = std::exp(scaled_logp);
                entropy -= p * logp;
            }
        }
        
        if (!std::isfinite(entropy)) {
            throw std::runtime_error("Entropy calculation resulted in non-finite value");
        }
        
        return entropy;
    }

private:
    llama_context* ctx_;
    llama_model* model_;
    std::weak_ptr<LlamaChain> parent_;
    LlamaChainPtr active_child_;

    std::vector<llama_token> tokens_;
    std::string accumulated_text_;
    StringMode string_mode_;
    std::atomic<int> kv_cache_token_count_start_;
    LogProbResult cached_logprobs_;
    mutable LogProbResult logprobs_buffer_;  // Pre-allocated buffer for compute_logprobs
    int vocab_size_;

    bool is_rolling_back_; // Flag to indicate rollback in progress

    // Greedy sampling
    llama_token greedy_sample() {
        LogProbResult logits = compute_logprobs();
        return static_cast<llama_token>(std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
    }

    // Deactivate siblings to maintain a single active child
    void deactivate_siblings() {
        LlamaChainPtr parent_ptr = parent_.lock();
        if (parent_ptr) {
            if (parent_ptr->active_child_ && parent_ptr->active_child_.get() != this) {
                parent_ptr->cache_active_child_logprobs();
                parent_ptr->active_child_->rollback();
                parent_ptr->active_child_.reset(); // Remove reference to the old active child
            }
        }
    }

    // Cache logprobs of the active child
    void cache_active_child_logprobs() {
        if (active_child_) {
            active_child_->update_logprobs();
            active_child_->cached_logprobs_ = active_child_->compute_logprobs();
        }
    }

    // Add a single token
    void add_token(llama_token token) {
        if (active_child_) {
            cache_active_child_logprobs();
            active_child_->rollback();
            active_child_.reset();
        }

        tokens_.push_back(token);
        decode_token(token);
        
        std::string piece = token_to_string(token);
        if (!piece.empty()) {
            accumulated_text_ += piece;
        }
    }

    // Add a string by tokenizing it
    void add_string(const std::string& text) {
        std::vector<llama_token> new_tokens;
        new_tokens.reserve(text.size() / 2);  // Rough estimate
        
        int needed = -llama_tokenize(model_, text.c_str(), static_cast<int>(text.size()), nullptr, 0, true, true);
        if (needed < 0) {
            throw TokenizationError("Failed to estimate tokens needed");
        }
        
        new_tokens.resize(needed);
        int n_tokens = llama_tokenize(model_, text.c_str(), static_cast<int>(text.size()), 
                                    new_tokens.data(), needed, true, true);
        
        if (n_tokens < 0) {
            throw TokenizationError("Failed to tokenize string");
        }
        
        new_tokens.resize(n_tokens); // Adjust size based on actual tokens
        add_tokens(new_tokens);
    }

    // Add multiple tokens
    void add_tokens(const std::vector<llama_token>& new_tokens) {
        if (new_tokens.empty()) return;

        // Batch decode
        llama_batch batch = llama_batch_get_one(const_cast<llama_token*>(new_tokens.data()), new_tokens.size());
        if (llama_decode(ctx_, batch) != 0) {
            throw DecodingError("Failed to decode tokens batch");
        }

        // Reserve space for efficiency
        tokens_.reserve(tokens_.size() + new_tokens.size());
        accumulated_text_.reserve(accumulated_text_.size() + new_tokens.size() * 4);

        for (llama_token token : new_tokens) {
            add_token(token);
        }
    }

    // Update cached logprobs
    void update_logprobs() {
        cached_logprobs_ = compute_logprobs();
    }

    // Compute log probabilities
    LogProbResult compute_logprobs() const  {
        if (logprobs_buffer_.size() != static_cast<size_t>(vocab_size_)) {
            logprobs_buffer_.resize(vocab_size_);
        }

        const float* logits = llama_get_logits(ctx_);
        
        // Find max logit for numerical stability 
        float max_logit = *std::max_element(logits, logits + vocab_size_);

        // Compute log probabilities with improved numerical stability
        constexpr double max_exp = 709.0;  // Max value for exp
        double sum_exp = 0.0;
        std::vector<double> scaled_exp(vocab_size_);

        for (int i = 0; i < vocab_size_; i++) {
            double scaled_logit = std::min(double(logits[i] - max_logit), max_exp);
            scaled_exp[i] = std::exp(scaled_logit);
            sum_exp += scaled_exp[i];
        }

        double log_sum_exp = std::log(sum_exp);

        for (int i = 0; i < vocab_size_; i++) {
            logprobs_buffer_[i] = float(std::log(scaled_exp[i]) - log_sum_exp);
        }

        return logprobs_buffer_;
    }

    // Decode a single token
    void decode_token(llama_token token) {
        llama_batch batch = llama_batch_get_one(&token, 1);
        if (llama_decode(ctx_, batch) != 0) {
            throw DecodingError("Failed to decode token");
        }
    }

    // Remove tokens from KV cache
    void remove_tokens_from_kv_cache() {
        int current_count = llama_get_kv_cache_token_count(ctx_);
        if (current_count > kv_cache_token_count_start_) {
            int n_remove = current_count - kv_cache_token_count_start_;
            llama_kv_cache_seq_rm(ctx_, 
                                0,
                                kv_cache_token_count_start_,
                                kv_cache_token_count_start_ + n_remove);
        }
    }
};

// Enhanced RAII wrapper for automatic rollback with additional safety features
class LlamaChainRollbackGuard {
public:
    explicit LlamaChainRollbackGuard(LlamaChain& chain) 
        : chain_(chain)
        , active_(true) 
    {}
    
    // Prevent copying
    LlamaChainRollbackGuard(const LlamaChainRollbackGuard&) = delete;
    LlamaChainRollbackGuard& operator=(const LlamaChainRollbackGuard&) = delete;
    
    // Allow moving
    LlamaChainRollbackGuard(LlamaChainRollbackGuard&& other) noexcept 
        : chain_(other.chain_)
        , active_(other.active_) 
    {
        other.active_ = false;
    }
    
    LlamaChainRollbackGuard& operator=(LlamaChainRollbackGuard&& other) noexcept {
        if (this != &other) {
            if (active_) {
                chain_.rollback();
            }
            active_ = other.active_;
            other.active_ = false;
        }
        return *this; // Corrected to return *this
    }
    
    void commit() { active_ = false; }
    
    void rollback() {
        if (active_) {
            chain_.rollback();
            active_ = false;
        }
    }
    
    ~LlamaChainRollbackGuard() {
        if (active_) {
            chain_.rollback();
        }
    }

private:
    LlamaChain& chain_;  // Reference instead of copy
    bool active_;
};

// Thread-safe global operator<< overloads for std::shared_ptr<LlamaChain>
inline std::shared_ptr<LlamaChain> operator<<(std::shared_ptr<LlamaChain> chain, llama_token token) {
    (*chain) << token;
    return chain;
}

inline std::shared_ptr<LlamaChain> operator<<(std::shared_ptr<LlamaChain> chain, const std::string &text) {
    (*chain) << text;
    return chain;
}

inline std::shared_ptr<LlamaChain> operator<<(std::shared_ptr<LlamaChain> chain, const std::vector<llama_token>& tokens) {
    (*chain) << tokens;
    return chain;
}
