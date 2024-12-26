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
#include <queue>
#include <unordered_map>

#include "llama.h"

// -----------------------------------------------------------------------------
// Custom exceptions
// -----------------------------------------------------------------------------
class LlamaChainException : public std::runtime_error {
public:
    explicit LlamaChainException(const std::string& msg)
        : std::runtime_error(msg) {}
};

class TokenizationError : public LlamaChainException {
public:
    explicit TokenizationError(const std::string& msg)
        : LlamaChainException(msg) {}
};

class DecodingError : public LlamaChainException {
public:
    explicit DecodingError(const std::string& msg)
        : LlamaChainException(msg) {}
};

// -----------------------------------------------------------------------------
// Types and forward declarations
// -----------------------------------------------------------------------------
using LogProbResult = std::vector<float>;

// -----------------------------------------------------------------------------
// Uncertainty metrics
// -----------------------------------------------------------------------------
struct UncertaintyMetrics {
    double entropy = 0.0;
    double kl_divergence = 0.0;
    double top_k_concentration = 0.0;
    double variance = 0.0;

    static UncertaintyMetrics calculate(const std::vector<float>& logprobs,
                                      int vocab_size,
                                      int top_k = 40)
    {
        UncertaintyMetrics metrics;
        if (logprobs.empty() || vocab_size <= 0) {
            return metrics;
        }

        // Convert logprobs to probabilities
        double max_logp = *std::max_element(logprobs.begin(), logprobs.end());
        std::vector<double> probs(vocab_size);
        double sum_exp = 0.0;
        
        for (int i = 0; i < vocab_size; i++) {
            double val = std::exp(double(logprobs[i]) - max_logp);
            probs[i] = val;
            sum_exp += val;
        }
        
        for (int i = 0; i < vocab_size; i++) {
            probs[i] /= sum_exp;
        }
        
        // Calculate metrics
        double uniform_p = 1.0 / double(vocab_size);
        
        // Entropy and KL divergence
        for (double p : probs) {
            if (p > 1e-15) {
                metrics.entropy -= p * std::log2(p);
                metrics.kl_divergence += p * std::log2(p / uniform_p);
            }
        }
        
        // Top-k concentration
        std::vector<double> sorted_probs = probs;
        std::sort(sorted_probs.begin(), sorted_probs.end(), std::greater<double>());
        int limit = std::min(top_k, vocab_size);
        for (int i = 0; i < limit; i++) {
            metrics.top_k_concentration += sorted_probs[i];
        }
        
        // Variance
        for (double p : probs) {
            double diff = p - uniform_p;
            metrics.variance += diff * diff;
        }
        metrics.variance /= double(vocab_size);
        
        return metrics;
    }
};

// -----------------------------------------------------------------------------
// Sampling strategies
// -----------------------------------------------------------------------------
class SamplingStrategy {
public:
    virtual ~SamplingStrategy() = default;
    virtual llama_token sample(const std::vector<float>& logprobs, int vocab_size) = 0;

    static std::unique_ptr<SamplingStrategy> create_temperature(float temp);
    static std::unique_ptr<SamplingStrategy> create_nucleus(float p);
    static std::unique_ptr<SamplingStrategy> create_mirostat(float tau, float eta);
};

class TemperatureSampling : public SamplingStrategy {
    float temp_;
public:
    explicit TemperatureSampling(float temp) : temp_(temp) {}

    llama_token sample(const std::vector<float>& logprobs, int vocab_size) override {
        if (temp_ <= 0.0f) {
            auto it = std::max_element(logprobs.begin(), logprobs.end());
            return static_cast<llama_token>(std::distance(logprobs.begin(), it));
        }

        std::vector<double> probs(vocab_size);
        float max_logit = *std::max_element(logprobs.begin(), logprobs.end());
        double sum = 0.0;
        
        for (int i = 0; i < vocab_size; i++) {
            double val = std::exp(double(logprobs[i] - max_logit) / double(temp_));
            probs[i] = val;
            sum += val;
        }
        
        for (int i = 0; i < vocab_size; i++) {
            probs[i] /= sum;
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::discrete_distribution<> dist(probs.begin(), probs.end());
        return static_cast<llama_token>(dist(gen));
    }

    void set_temperature(float t) { temp_ = t; }
    float get_temperature() const { return temp_; }
};

class NucleusSampling : public SamplingStrategy {
    float p_;
public:
    explicit NucleusSampling(float p) : p_(p) {}

    llama_token sample(const std::vector<float>& logprobs, int vocab_size) override {
        if (logprobs.empty() || vocab_size <= 0) return 0;

        float max_logit = *std::max_element(logprobs.begin(), logprobs.end());
        std::vector<std::pair<double,int>> pairs;
        pairs.reserve(vocab_size);

        for (int i = 0; i < vocab_size; i++) {
            double val = std::exp(double(logprobs[i] - max_logit));
            pairs.emplace_back(val, i);
        }
        
        std::sort(pairs.begin(), pairs.end(),
                 [](auto &a, auto &b){ return a.first > b.first; });

        double cumsum = 0.0;
        int cutoff_index = 0;
        for (auto &pr : pairs) {
            cumsum += pr.first;
            cutoff_index++;
            if (cumsum >= double(p_)) break;
        }
        cutoff_index = std::max(1, cutoff_index);

        double sum_top = 0.0;
        for (int i = 0; i < cutoff_index; i++) {
            sum_top += pairs[i].first;
        }

        std::vector<double> dist_probs(cutoff_index);
        for (int i = 0; i < cutoff_index; i++) {
            dist_probs[i] = pairs[i].first / sum_top;
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::discrete_distribution<> dist(dist_probs.begin(), dist_probs.end());
        return static_cast<llama_token>(pairs[dist(gen)].second);
    }
};

class MirostatSampling : public SamplingStrategy {
    float tau_;
    float eta_;
    float mu_;
public:
    MirostatSampling(float tau, float eta)
        : tau_(tau), eta_(eta), mu_(5.0f) {}

    llama_token sample(const std::vector<float>& logprobs, int vocab_size) override {
        if (logprobs.empty() || vocab_size <= 0) return 0;

        float max_logit = *std::max_element(logprobs.begin(), logprobs.end());
        std::vector<double> probs(vocab_size, 0.0);

        double sum = 0.0;
        double entropy = 0.0;
        
        for (int i = 0; i < vocab_size; i++) {
            if (double(logprobs[i]) > double(mu_)) {
                double val = std::exp(double(logprobs[i] - max_logit));
                probs[i] = val;
                sum += val;
            }
        }
        
        if (sum <= 0.0) {
            for (int i = 0; i < vocab_size; i++) {
                double val = std::exp(double(logprobs[i] - max_logit));
                probs[i] = val;
                sum += val;
            }
        }

        for (int i = 0; i < vocab_size; i++) {
            if (probs[i] > 0.0) {
                double p = probs[i] / sum;
                probs[i] = p;
                entropy -= p * std::log2(p);
            }
        }

        mu_ = mu_ + eta_ * (tau_ - float(entropy));

        std::random_device rd;
        std::mt19937 gen(rd());
        std::discrete_distribution<> dist(probs.begin(), probs.end());
        return static_cast<llama_token>(dist(gen));
    }
};

// Factory implementations
inline std::unique_ptr<SamplingStrategy> SamplingStrategy::create_temperature(float temp) {
    return std::make_unique<TemperatureSampling>(temp);
}

inline std::unique_ptr<SamplingStrategy> SamplingStrategy::create_nucleus(float p) {
    return std::make_unique<NucleusSampling>(p);
}

inline std::unique_ptr<SamplingStrategy> SamplingStrategy::create_mirostat(float tau, float eta) {
    return std::make_unique<MirostatSampling>(tau, eta);
}

// -----------------------------------------------------------------------------
// Main LlamaChain class
// -----------------------------------------------------------------------------
class LlamaChain {
public:
    LogProbResult token_logprobs;

    // Constructor
    LlamaChain(llama_context* ctx, llama_model* model, int start_kv_pos = -1)
        : ctx_(ctx)
        , model_(model)
        , vocab_size_(llama_n_vocab(model))
        , start_kv_pos_(start_kv_pos < 0 ? llama_get_kv_cache_token_count(ctx) : start_kv_pos)
        , current_kv_pos_(start_kv_pos_)
    {
        logprobs_buffer_.reserve(vocab_size_);
        tokens_.reserve(1024);
        text_.reserve(4096);
        sampling_strategy_ = SamplingStrategy::create_temperature(0.8f);
    }

    // Factory
    static std::shared_ptr<LlamaChain> create(llama_context* ctx, llama_model* model) {
        return std::make_shared<LlamaChain>(ctx, model);
    }

    

    // Checkpoint/rollback
    std::shared_ptr<LlamaChain> checkpoint() {
        auto chain = std::make_shared<LlamaChain>(ctx_, model_, current_kv_pos_);
        chain->cached_logprobs_ = cached_logprobs_;
        return chain;
    }

    void rollback() {
        if (current_kv_pos_ > start_kv_pos_) {
            llama_kv_cache_seq_rm(ctx_, 0, start_kv_pos_, current_kv_pos_);
            current_kv_pos_ = start_kv_pos_;
            tokens_.clear();
            text_.clear();
            cached_logprobs_.clear();
        }
    }

    // Operators
    LlamaChain& operator<<(llama_token token) {
        add_token(token);
        update_logprobs();
        return *this;
    }

    LlamaChain& operator<<(const std::string& text) {
        add_string(text);
        update_logprobs();
        return *this;
    }

    LlamaChain& operator<<(const std::vector<llama_token>& tokens) {
        add_tokens(tokens);
        update_logprobs();
        return *this;
    }

    friend const LogProbResult& operator>>(const LlamaChain& chain, LogProbResult& result) {
        result = chain.cached_logprobs_;
        return result;
    }

    friend LlamaChain& operator<<(LlamaChain& chain, const LogProbResult& logprobs) {
        chain.cached_logprobs_ = logprobs;
        return chain;
    }

    // Accessors
    std::string string() const { return text_; }
    std::vector<llama_token> tokens() const { return tokens_; }
    LogProbResult get_logprobs() const { return cached_logprobs_; }

    // Sampling interface
    llama_token sample(int top_k = 0) {
        auto logprobs = get_logprobs();
        apply_repetition_penalty(logprobs);

        // Calculate metrics before top-k filtering
        auto metrics = UncertaintyMetrics::calculate(logprobs, vocab_size_);
        metrics_history_.push(metrics);
        if (metrics_history_.size() > MAX_HISTORY) {
            metrics_history_.pop();
        }

        // Apply top-k before sampling
        if (top_k > 0 && top_k < vocab_size_) {
            std::vector<std::pair<float, int>> pairs;
            pairs.reserve(vocab_size_);
            for (int i = 0; i < vocab_size_; i++) {
                pairs.emplace_back(logprobs[i], i);
            }
            
            // Partial sort to find top-k elements
            std::partial_sort(pairs.begin(), 
                            pairs.begin() + top_k, 
                            pairs.end(),
                            std::greater<>());
            
            // Set all logprobs outside top-k to negative infinity
            float min_logprob = pairs[top_k - 1].first;
            for (int i = 0; i < vocab_size_; i++) {
                if (logprobs[i] < min_logprob) {
                    logprobs[i] = -INFINITY;
                }
            }
        }

        if (!sampling_strategy_) {
            sampling_strategy_ = SamplingStrategy::create_temperature(0.8f);
        }
        return sampling_strategy_->sample(logprobs, vocab_size_);
    }

    llama_token sample(float temp, int top_k) {
        auto* temp_strat = dynamic_cast<TemperatureSampling*>(sampling_strategy_.get());
        if (!temp_strat) {
            sampling_strategy_ = SamplingStrategy::create_temperature(temp);
        } else {
            temp_strat->set_temperature(temp);
        }
        return sample(top_k);
    }

    void set_sampling_strategy(std::unique_ptr<SamplingStrategy> strategy) {
        sampling_strategy_ = std::move(strategy);
    }

    // Token conversion
    std::string token_to_string(llama_token token) const {
        std::array<char,256> buf;
        int n = llama_token_to_piece(model_, token, buf.data(), buf.size(), false, true);
        if (n <= 0) return "";
        if (n >= int(buf.size())) {
            throw LlamaChainException("Buffer overflow in token_to_string");
        }
        return std::string(buf.data(), n);
    }

    double calculate_uncertainty() {
        auto metrics = UncertaintyMetrics::calculate(cached_logprobs_, vocab_size_);
        return metrics.entropy;
    }

    // Metrics
    UncertaintyMetrics get_current_metrics() const {
        return metrics_history_.empty() ? UncertaintyMetrics{} : metrics_history_.back();
    }

    double get_uncertainty_trend(size_t window = 10) const {
        if (metrics_history_.size() < 2) return 0.0;

        std::vector<double> entropies;
        entropies.reserve(window);

        auto temp_queue = metrics_history_;
        while (!temp_queue.empty() && entropies.size() < window) {
            entropies.push_back(temp_queue.front().entropy);
            temp_queue.pop();
        }
        if (entropies.size() < 2) return 0.0;

        // Linear regression
        double x_mean = 0.0, y_mean = 0.0;
        for (size_t i = 0; i < entropies.size(); i++) {
            x_mean += double(i);
            y_mean += entropies[i];
        }
        x_mean /= double(entropies.size());
        y_mean /= double(entropies.size());

        double num = 0.0, den = 0.0;
        for (size_t i = 0; i < entropies.size(); i++) {
            double x_diff = double(i) - x_mean;
            double y_diff = entropies[i] - y_mean;
            num += x_diff * y_diff;
            den += x_diff * x_diff;
        }
        return den == 0.0 ? 0.0 : num / den;
    }

protected:
    llama_context* ctx_;
    llama_model* model_;
    int vocab_size_;
    
    // KV cache tracking
    int start_kv_pos_;
    int current_kv_pos_;

    std::vector<llama_token> tokens_;
    std::string text_;
    LogProbResult cached_logprobs_;
    LogProbResult logprobs_buffer_;

    std::unique_ptr<SamplingStrategy> sampling_strategy_;
    std::queue<UncertaintyMetrics> metrics_history_;
    static constexpr size_t MAX_HISTORY = 100;

    // Repetition penalty settings
    float repetition_penalty_ = 1.1f;
    static constexpr size_t MAX_REPEAT_NGRAM = 4;

    void add_token(llama_token token) {
        int current_cache_size = llama_get_kv_cache_token_count(ctx_);
        if (current_cache_size > current_kv_pos_) {
            llama_kv_cache_seq_rm(ctx_, 0, current_kv_pos_, current_cache_size);
        }

        tokens_.push_back(token);
        decode_token(token);
        current_kv_pos_++;

        std::string piece = token_to_string(token);
        if (!piece.empty()) {
            text_ += piece;
        }
    }

    void add_string(const std::string& text) {
        if (text.empty()) return;

        std::vector<llama_token> new_tokens;
        new_tokens.reserve(text.size()/2);

        int needed = -llama_tokenize(model_, text.c_str(), int(text.size()),
                                   nullptr, 0, true, true);
        if (needed < 0) {
            throw TokenizationError("Failed to estimate tokens needed");
        }

        new_tokens.resize(needed);
        int n_tokens = llama_tokenize(model_, text.c_str(), int(text.size()),
                                    new_tokens.data(), needed, true, true);
        if (n_tokens < 0) {
            throw TokenizationError("Failed to tokenize string");
        }
        new_tokens.resize(n_tokens);
        add_tokens(new_tokens);
    }

    void add_tokens(const std::vector<llama_token>& new_tokens) {
        if (new_tokens.empty()) return;

        int current_cache_size = llama_get_kv_cache_token_count(ctx_);
        if (current_cache_size > current_kv_pos_) {
            llama_kv_cache_seq_rm(ctx_, 0, current_kv_pos_, current_cache_size);
        }

        llama_batch batch = llama_batch_get_one(const_cast<llama_token*>(new_tokens.data()),
                                              new_tokens.size());
        if (llama_decode(ctx_, batch) != 0) {
            throw DecodingError("Failed to decode tokens batch");
        }

        tokens_.reserve(tokens_.size() + new_tokens.size());
        text_.reserve(text_.size() + new_tokens.size() * 4);

        for (llama_token t : new_tokens) {
            tokens_.push_back(t);
            std::string piece = token_to_string(t);
            if (!piece.empty()) {
                text_ += piece;
            }
        }
        current_kv_pos_ += new_tokens.size();
    }

    void update_logprobs() {
        cached_logprobs_ = compute_logprobs();
    }

    LogProbResult compute_logprobs() {
        if (logprobs_buffer_.size() != size_t(vocab_size_)) {
            logprobs_buffer_.resize(vocab_size_);
        }

        const float* logits = llama_get_logits(ctx_);
        if (!logits) {
            throw DecodingError("Null pointer from llama_get_logits()");
        }

        float max_logit = *std::max_element(logits, logits + vocab_size_);
        constexpr double max_exp = 709.0; // numeric stability
        double sum_exp = 0.0;
        std::vector<double> scaled_exp(vocab_size_);

        for (int i = 0; i < vocab_size_; i++) {
            double scaled_logit = std::min(double(logits[i] - max_logit), max_exp);
            double val = std::exp(scaled_logit);
            scaled_exp[i] = val;
            sum_exp += val;
        }
        double log_sum_exp = std::log(sum_exp);

        for (int i = 0; i < vocab_size_; i++) {
            logprobs_buffer_[i] = float(std::log(scaled_exp[i]) - log_sum_exp);
        }
        return logprobs_buffer_;
    }

    void decode_token(llama_token token) {
        llama_batch batch = llama_batch_get_one(&token, 1);
        if (llama_decode(ctx_, batch) != 0) {
            throw DecodingError("Failed to decode token");
        }
    }

    void apply_repetition_penalty(std::vector<float>& logprobs) {
        if (tokens_.empty()) return;

        std::unordered_map<llama_token, int> token_counts;
        for (const auto& token : tokens_) {
            token_counts[token]++;
        }

        for (const auto& [token, count] : token_counts) {
            if (count > 1) {
                float penalty = std::log(repetition_penalty_) * (count - 1);
                if (token < vocab_size_) {
                    logprobs[token] = std::max(logprobs[token] - penalty, -20.0f);
                }
            }
        }
    }
};

// RAII guard
class LlamaChainRollbackGuard {
    LlamaChain& chain_;
    bool active_;
public:
    explicit LlamaChainRollbackGuard(LlamaChain& chain)
        : chain_(chain), active_(true) {}
    
    void commit() { active_ = false; }
    void rollback() {
        if (active_) {
            chain_.rollback();
            active_ = false;
        }
    }
    
    ~LlamaChainRollbackGuard() {
        if (active_) chain_.rollback();
    }

    LlamaChainRollbackGuard(const LlamaChainRollbackGuard&) = delete;
    LlamaChainRollbackGuard& operator=(const LlamaChainRollbackGuard&) = delete;
    
    LlamaChainRollbackGuard(LlamaChainRollbackGuard&& other) noexcept
        : chain_(other.chain_), active_(other.active_) {
        other.active_ = false;
    }
    LlamaChainRollbackGuard& operator=(LlamaChainRollbackGuard&&) = delete;
};

// Thread-safe operator<< overloads for shared_ptr<LlamaChain>
inline std::shared_ptr<LlamaChain> operator<<(std::shared_ptr<LlamaChain> chain,
                                            llama_token token) {
    (*chain) << token;
    return chain;
}

inline std::shared_ptr<LlamaChain> operator<<(std::shared_ptr<LlamaChain> chain,
                                            const std::string& text) {
    (*chain) << text;
    return chain;
}

inline std::shared_ptr<LlamaChain> operator<<(std::shared_ptr<LlamaChain> chain,
                                            const std::vector<llama_token>& tokens) {
    (*chain) << tokens;
    return chain;
}