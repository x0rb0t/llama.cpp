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
// Forward declarations
// -----------------------------------------------------------------------------
class LlamaChain;
class LlamaChainRollbackGuard;
using LlamaChainPtr = std::shared_ptr<LlamaChain>;
using LogProbResult = std::vector<float>;

// -----------------------------------------------------------------------------
// Advanced uncertainty metrics
// -----------------------------------------------------------------------------
struct UncertaintyMetrics {
    double entropy = 0.0;                // Shannon entropy
    double kl_divergence = 0.0;          // KL divergence from uniform
    double top_k_concentration = 0.0;    // Probability mass in top-k
    double variance = 0.0;              // Token probability variance
    
    static UncertaintyMetrics calculate(const std::vector<float>& logprobs,
                                        int vocab_size,
                                        int top_k = 40)
    {
        UncertaintyMetrics metrics;
        if (logprobs.empty() || vocab_size <= 0) {
            return metrics;
        }

        // Convert logprobs to probabilities with numerical stability
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
        
        // Entropy
        for (double p : probs) {
            if (p > 1e-15) {
                metrics.entropy -= p * std::log2(p);
            }
        }
        
        // KL divergence from uniform
        double uniform_p = 1.0 / double(vocab_size);
        for (double p : probs) {
            if (p > 1e-15) {
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
        
        // Probability variance (distance from mean_prob)
        double mean_prob = 1.0 / double(vocab_size);
        double var_sum = 0.0;
        for (double p : probs) {
            double diff = p - mean_prob;
            var_sum += diff * diff;
        }
        metrics.variance = var_sum / double(vocab_size);
        
        return metrics;
    }
};

// -----------------------------------------------------------------------------
// Advanced sampling strategies
// -----------------------------------------------------------------------------
class SamplingStrategy {
public:
    virtual ~SamplingStrategy() = default;
    // Sample token ID given logprobs
    virtual llama_token sample(const std::vector<float>& logprobs, int vocab_size) = 0;

    // Factory shortcuts
    static std::unique_ptr<SamplingStrategy> create_temperature(float temp);
    static std::unique_ptr<SamplingStrategy> create_nucleus(float p);
    static std::unique_ptr<SamplingStrategy> create_mirostat(float tau, float eta);
};

// Temperature
class TemperatureSampling : public SamplingStrategy {
    float temp_;
public:
    explicit TemperatureSampling(float temp) : temp_(temp) {}

    llama_token sample(const std::vector<float>& logprobs, int vocab_size) override {
        // Greedy if temp <= 0
        if (temp_ <= 0.0f) {
            auto it = std::max_element(logprobs.begin(), logprobs.end());
            return static_cast<llama_token>(std::distance(logprobs.begin(), it));
        }

        // Exponentiate
        std::vector<double> probs(vocab_size);
        float max_logit = *std::max_element(logprobs.begin(), logprobs.end());
        double sum = 0.0;
        for (int i = 0; i < vocab_size; i++) {
            double val = std::exp(double(logprobs[i] - max_logit) / double(temp_));
            probs[i] = val;
            sum += val;
        }
        // Normalize
        for (int i = 0; i < vocab_size; i++) {
            probs[i] /= sum;
        }

        // Sample
        std::random_device rd;
        std::mt19937 gen(rd());
        std::discrete_distribution<> dist(probs.begin(), probs.end());
        return static_cast<llama_token>(dist(gen));
    }

    // For convenience, let us edit temperature on the fly
    void set_temperature(float t) { temp_ = t; }
    float get_temperature() const { return temp_; }
};

// Nucleus (top-p)
class NucleusSampling : public SamplingStrategy {
    float p_;
public:
    explicit NucleusSampling(float p) : p_(p) {}

    llama_token sample(const std::vector<float>& logprobs, int vocab_size) override {
        if (logprobs.empty() || vocab_size <= 0) {
            return 0;
        }

        // Convert to unnormalized probs
        float max_logit = *std::max_element(logprobs.begin(), logprobs.end());
        std::vector<std::pair<double,int>> pairs;
        pairs.reserve(vocab_size);

        for (int i = 0; i < vocab_size; i++) {
            double val = std::exp(double(logprobs[i] - max_logit));
            pairs.emplace_back(val, i);
        }
        // Sort desc
        std::sort(pairs.begin(), pairs.end(),
                  [](auto &a, auto &b){ return a.first > b.first; });

        // find top region
        double cumsum = 0.0;
        int cutoff_index = 0;
        for (auto &pr : pairs) {
            cumsum += pr.first;
            cutoff_index++;
            if (cumsum >= double(p_)) {
                break;
            }
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

        // Sample
        std::random_device rd;
        std::mt19937 gen(rd());
        std::discrete_distribution<> dist(dist_probs.begin(), dist_probs.end());
        int chosen = dist(gen);
        return static_cast<llama_token>(pairs[chosen].second);
    }
};

// Mirostat
class MirostatSampling : public SamplingStrategy {
    float tau_;   // target entropy
    float eta_;   // learning rate
    float mu_;    // internal factor
public:
    MirostatSampling(float tau, float eta)
        : tau_(tau), eta_(eta), mu_(5.0f)
    {}

    llama_token sample(const std::vector<float>& logprobs, int vocab_size) override {
        if (logprobs.empty() || vocab_size <= 0) {
            return 0;
        }

        float max_logit = *std::max_element(logprobs.begin(), logprobs.end());
        std::vector<double> probs(vocab_size, 0.0);

        // Build distribution above mu_
        double sum = 0.0;
        double entropy = 0.0;
        for (int i = 0; i < vocab_size; i++) {
            if (double(logprobs[i]) > double(mu_)) {
                double val = std::exp(double(logprobs[i] - max_logit));
                probs[i] = val;
                sum += val;
            }
        }
        // fallback if sum == 0
        if (sum <= 0.0) {
            double fallback_sum = 0.0;
            for (int i = 0; i < vocab_size; i++) {
                double val = std::exp(double(logprobs[i] - max_logit));
                probs[i] = val;
                fallback_sum += val;
            }
            sum = fallback_sum;
        }

        // Normalize & compute entropy
        for (int i = 0; i < vocab_size; i++) {
            if (probs[i] > 0.0) {
                double p = probs[i] / sum;
                probs[i] = p; // store normalized
                entropy -= p * std::log2(p);
            }
        }
        // Mirostat update
        mu_ = mu_ + eta_ * (tau_ - float(entropy));

        // Sample
        std::random_device rd;
        std::mt19937 gen(rd());
        std::discrete_distribution<> dist(probs.begin(), probs.end());
        return static_cast<llama_token>(dist(gen));
    }
};

// Factories
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
// The SINGLE LlamaChain class, with advanced sampling & metrics integrated
// -----------------------------------------------------------------------------
class LlamaChain : public std::enable_shared_from_this<LlamaChain> {
public:
    LogProbResult token_logprobs;

    enum class StringMode {
        CONCAT,     // Concatenate parent's string
        SEPARATE    // Start new string from this point
    };

    // -------------------------------------------------------------------------
    // Constructors
    // -------------------------------------------------------------------------
    LlamaChain(llama_context* ctx,
               llama_model* model,
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
        tokens_.reserve(1024);
        accumulated_text_.reserve(4096);

        // Default strategy is temperature=0.8f
        sampling_strategy_ = SamplingStrategy::create_temperature(0.8f);
    }

    virtual ~LlamaChain() = default;

    // Factory
    static std::shared_ptr<LlamaChain> create(llama_context* ctx, llama_model* model) {
        return std::make_shared<LlamaChain>(ctx, model);
    }

    // -------------------------------------------------------------------------
    // Checkpoint/rollback
    // -------------------------------------------------------------------------
    std::shared_ptr<LlamaChain> checkpoint(StringMode mode = StringMode::CONCAT) {
        if (active_child_) {
            cache_active_child_logprobs();
            active_child_->rollback();
            active_child_.reset();
        }

        auto child = std::make_shared<LlamaChain>(ctx_, model_, shared_from_this(), mode);
        active_child_ = child;
        // Pass down parent's cached logprobs
        (*child) << cached_logprobs_;
        return child;
    }

    void rollback() {
        if (is_rolling_back_) {
            return; // guard recursion
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

    // -------------------------------------------------------------------------
    // Operators to add tokens or strings
    // -------------------------------------------------------------------------
    LlamaChain& operator<<(llama_token token) {
        deactivate_siblings();
        add_token(token);
        update_logprobs();
        return *this;
    }

    LlamaChain& operator<<(const std::string& text) {
        deactivate_siblings();
        add_string(text);
        update_logprobs();
        return *this;
    }

    LlamaChain& operator<<(const std::vector<llama_token>& tokens) {
        deactivate_siblings();
        add_tokens(tokens);
        update_logprobs();
        return *this;
    }

    // Also allow writing/reading logprobs
    friend const LogProbResult& operator>>(const LlamaChain& chain,
                                           LogProbResult& result)
    {
        result = chain.cached_logprobs_;
        return result;
    }

    friend LlamaChain& operator<<(LlamaChain& chain, const LogProbResult& logprobs) {
        chain.cached_logprobs_ = logprobs;
        return chain;
    }

    // -------------------------------------------------------------------------
    // Accumulated string
    // -------------------------------------------------------------------------
    std::string string() const {
        if (string_mode_ == StringMode::SEPARATE) {
            return accumulated_text_;
        }
        // Prepend parent's text
        std::string result;
        if (auto parent_sp = parent_.lock()) {
            result = parent_sp->string();
        }
        result += accumulated_text_;
        return result;
    }

    // Access tokens
    std::vector<llama_token> tokens() const {
        return tokens_;
    }

    // -------------------------------------------------------------------------
    // Logprob & uncertainty
    // -------------------------------------------------------------------------
    LogProbResult get_logprobs() const {
        return cached_logprobs_;
    }

    // Now calls advanced metrics to get the entropy
    double calculate_uncertainty() {
        auto metrics = UncertaintyMetrics::calculate(cached_logprobs_, vocab_size_);
        return metrics.entropy;  // your original code used entropy, so let's return that
    }

    // Store advanced metrics each time we sample, retrieve them
    UncertaintyMetrics get_current_metrics() const {
        if (metrics_history_.empty()) {
            return UncertaintyMetrics{};
        }
        return metrics_history_.back();
    }

    // E.g. slope of entropy over last N steps
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

        // linear regression
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
        if (den == 0.0) return 0.0;
        return num / den;
    }

    // -------------------------------------------------------------------------
    // Sampling interface
    // -------------------------------------------------------------------------
    // 1) sample() uses the currently set strategy, applies repetition penalty,
    //    updates advanced metrics, returns next token
    llama_token sample() {
        auto logprobs = get_logprobs();
        
        // Repetition penalty
        apply_repetition_penalty(logprobs);

        // Compute advanced metrics
        auto metrics = UncertaintyMetrics::calculate(logprobs, vocab_size_);
        metrics_history_.push(metrics);
        if (metrics_history_.size() > MAX_HISTORY) {
            metrics_history_.pop();
        }

        // Strategy-based sampling
        if (!sampling_strategy_) {
            // If none is set for some reason, create default (Temperature=0.8f)
            sampling_strategy_ = SamplingStrategy::create_temperature(0.8f);
        }

        return sampling_strategy_->sample(logprobs, vocab_size_);
    }

    // 2) sample(float temp, int top_k):
    //    - If current strategy is TemperatureSampling, set that temp,
    //      ignore top_k for now (or we can do a partial clamp).
    //    - If not TemperatureSampling, throw.
    llama_token sample(float temp, int top_k) {
        (void) top_k; // top_k is not relevant for advanced strategies except "classic" path
        // If the current strategy is TemperatureSampling, update it
        if (!sampling_strategy_) {
            sampling_strategy_ = SamplingStrategy::create_temperature(temp);
        } else {
            auto* temp_strat = dynamic_cast<TemperatureSampling*>(sampling_strategy_.get());
            if (!temp_strat) {
                // We only allow setting temp if the strategy is TemperatureSampling
                throw LlamaChainException(
                    "sample(float temp, int top_k) is only valid if strategy is TemperatureSampling.");
            }
            // Update temperature
            temp_strat->set_temperature(temp);
        }
        return sample();
    }

    // Allow user to set different strategy (nucleus, mirostat, etc.)
    void set_sampling_strategy(std::unique_ptr<SamplingStrategy> strategy) {
        sampling_strategy_ = std::move(strategy);
    }

    // Convert token -> string with boundary checks
    std::string token_to_string(llama_token token) const {
        std::array<char,256> buf;
        int n = llama_token_to_piece(model_, token,
                                     buf.data(), buf.size(),
                                     false,
                                     true);
        if (n <= 0) {
            return "";
        }
        if (n >= int(buf.size())) {
            throw LlamaChainException("Buffer overflow in token_to_string");
        }
        return std::string(buf.data(), n);
    }

    // -------------------------------------------------------------------------
    // Private/protected stuff
    // -------------------------------------------------------------------------
protected:
    llama_context* ctx_;
    llama_model* model_;
    std::weak_ptr<LlamaChain> parent_;
    std::shared_ptr<LlamaChain> active_child_;

    std::vector<llama_token> tokens_;
    std::string accumulated_text_;
    StringMode string_mode_;
    std::atomic<int> kv_cache_token_count_start_;
    LogProbResult cached_logprobs_;
    mutable LogProbResult logprobs_buffer_;
    int vocab_size_;

    bool is_rolling_back_;

    // Sampling additions
    std::unique_ptr<SamplingStrategy> sampling_strategy_;

    // Advanced metrics history
    std::queue<UncertaintyMetrics> metrics_history_;
    static constexpr size_t MAX_HISTORY = 100;

    // Repetition penalty settings
    std::unordered_map<llama_token, int> token_frequency_;
    static constexpr size_t MAX_REPEAT_NGRAM = 4;
    float repetition_penalty_ = 1.1f;

    // Deactivate siblings
    void deactivate_siblings() {
        std::shared_ptr<LlamaChain> parent_ptr = parent_.lock();
        if (parent_ptr) {
            if (parent_ptr->active_child_ && parent_ptr->active_child_.get() != this) {
                parent_ptr->cache_active_child_logprobs();
                parent_ptr->active_child_->rollback();
                parent_ptr->active_child_.reset();
            }
        }
    }

    // Cache child's logprobs
    void cache_active_child_logprobs() {
        if (active_child_) {
            active_child_->update_logprobs();
            active_child_->cached_logprobs_ = active_child_->compute_logprobs();
        }
    }

    // Token addition
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

    // String addition
    void add_string(const std::string& text) {
        if (text.empty()) return;

        // Tokenize
        std::vector<llama_token> new_tokens;
        new_tokens.reserve(text.size()/2);

        int needed = -llama_tokenize(model_, text.c_str(), int(text.size()),
                                     nullptr, 0,
                                     true,  // add_bos
                                     true); // add_eos
        if (needed < 0) {
            throw TokenizationError("Failed to estimate tokens needed");
        }

        new_tokens.resize(needed);
        int n_tokens = llama_tokenize(model_, text.c_str(), int(text.size()),
                                      new_tokens.data(),
                                      needed,
                                      true,
                                      true);
        if (n_tokens < 0) {
            throw TokenizationError("Failed to tokenize string");
        }
        new_tokens.resize(n_tokens);
        add_tokens(new_tokens);
    }

    // Multi-token addition
    void add_tokens(const std::vector<llama_token>& new_tokens) {
        if (new_tokens.empty()) return;

        llama_batch batch = llama_batch_get_one(const_cast<llama_token*>(new_tokens.data()),
                                                new_tokens.size());
        if (llama_decode(ctx_, batch) != 0) {
            throw DecodingError("Failed to decode tokens batch");
        }

        tokens_.reserve(tokens_.size() + new_tokens.size());
        accumulated_text_.reserve(accumulated_text_.size() + new_tokens.size()*4);

        for (llama_token t : new_tokens) {
            add_token(t);
        }
    }

    // Update cached logprobs
    void update_logprobs() {
        cached_logprobs_ = compute_logprobs();
    }

    // Calculate logprobs from llama's logits
    LogProbResult compute_logprobs() const {
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

    // Decode single token
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

    // Repetition penalty
    void apply_repetition_penalty(std::vector<float>& logprobs) {
        if (tokens_.empty()) return;

        // Count frequency of each token
        std::unordered_map<llama_token, int> token_counts;
        for (const auto& token : tokens_) {
            token_counts[token]++;
        }

        // Apply penalty to repeated tokens
        for (const auto& [token, count] : token_counts) {
            if (count > 1) {
                // Calculate penalty in log-space
                float penalty = std::log(repetition_penalty_) * (count - 1);
                // Ensure token index is within bounds
                if (token < vocab_size_) {
                    logprobs[token] = std::max(logprobs[token] - penalty, -20.0f); // Clamp to avoid underflow
                }
            }
        }
    }

};

// -----------------------------------------------------------------------------
// RAII rollback guard
// -----------------------------------------------------------------------------
class LlamaChainRollbackGuard {
public:
    explicit LlamaChainRollbackGuard(LlamaChain& chain)
        : chain_(chain)
        , active_(true)
    {}

    LlamaChainRollbackGuard(const LlamaChainRollbackGuard&) = delete;
    LlamaChainRollbackGuard& operator=(const LlamaChainRollbackGuard&) = delete;

    // Move
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
        return *this;
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
    LlamaChain& chain_;
    bool active_;
};

// -----------------------------------------------------------------------------
// Thread-safe operator<< overloads for shared_ptr<LlamaChain>
// -----------------------------------------------------------------------------
inline std::shared_ptr<LlamaChain> operator<<(std::shared_ptr<LlamaChain> chain,
                                              llama_token token)
{
    (*chain) << token;
    return chain;
}

inline std::shared_ptr<LlamaChain> operator<<(std::shared_ptr<LlamaChain> chain,
                                              const std::string& text)
{
    (*chain) << text;
    return chain;
}

inline std::shared_ptr<LlamaChain> operator<<(std::shared_ptr<LlamaChain> chain,
                                              const std::vector<llama_token>& tokens)
{
    (*chain) << tokens;
    return chain;
}
