#pragma once
#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <deque>
#include <algorithm>
#include <cmath>
#include <random>
#include <cassert>

// ============================================================================
// Global chemical environment
// ============================================================================
// A single global vector of chemical activations, each in [0,1] (soft bound,
// not hard-clamped here -- something upstream, e.g. output neurons, is
// responsible for keeping it sane).
struct ChemicalEnvironment {
    std::vector<double> levels;

    explicit ChemicalEnvironment(size_t num_chemicals) : levels(num_chemicals, 0.0) {}

    size_t size() const { return levels.size(); }
};

// ============================================================================
// Phase state machine shared by every neuron type
// ============================================================================
enum class Phase {
    Receive,
    Compute,
    Send
};

// A message is just a signed scalar traveling along a path (connection).
using Signal = double;

// ============================================================================
// Connection identifiers
// ============================================================================
// Neurons are owned by the Brain in a vector and referenced by index.
using NeuronId = size_t;
constexpr NeuronId INVALID_NEURON = static_cast<NeuronId>(-1);

// ============================================================================
// Base Neuron
// ============================================================================
// Implements:
//   - 3-phase cycle (Receive / Compute / Send), independently timed per neuron
//   - Mailbox with cross-tick caching (messages arriving off-phase are queued)
//   - Chemical affectation vector + muting formula
//   - Basic transformation (identity-ish, scaled by mute factor)
//
// MegaNeuron, InputNeuron, and OutputNeuron all derive from this and override
// behavior as needed, but keep the same external phase contract.
class Neuron {
public:
    Neuron(size_t num_chemicals, std::mt19937& rng)
        : affectation(num_chemicals, 0.0),
          phase(Phase::Receive),
          mega_compute_ticks_remaining(0)
    {
        // Small random init for affectation -- neutral-ish start so early
        // behavior isn't dominated by chemistry before anything is learned.
        std::uniform_real_distribution<double> dist(-0.2, 0.2);
        for (auto& a : affectation) a = dist(rng);
    }

    virtual ~Neuron() = default;

    // Outgoing connections: indices into Brain's neuron list.
    std::vector<NeuronId> outgoing;

    // Per-neuron chemical affectation vector, same length as global chemical array.
    std::vector<double> affectation;

    Phase phase;

    // For MegaNeuron: how many more ticks the current compute phase needs.
    // Base Neuron always computes in exactly 1 tick, so this is unused there.
    int mega_compute_ticks_remaining;

    // -------------------------------------------------------------
    // Mailbox: messages "in flight" this tick, plus a cache for messages
    // that arrived while this neuron wasn't in Receive phase.
    // -------------------------------------------------------------
    std::vector<Signal> inbox_this_tick;   // messages delivered exactly this tick
    std::deque<Signal> cached_inbox;       // messages waiting for next Receive phase

    // Result of the most recent Compute phase, ready to be sent out.
    Signal pending_output = 0.0;

    // Deliver a message to this neuron "right now" (called by Brain during
    // the Send step of some other neuron). If this neuron isn't in Receive
    // phase, the message is cached for its next Receive phase instead.
    void deliverMessage(Signal s) {
        if (phase == Phase::Receive) {
            inbox_this_tick.push_back(s);
        } else {
            cached_inbox.push_back(s);
        }
    }

    // Compute the chemical mute scalar given the current global chemical state.
    // mute = sum(global[i] * affectation[i])
    // Then output = base_output * clamp(1 + mute, 0, mute_cap)
    static double muteFactor(const std::vector<double>& global, const std::vector<double>& affect) {
        double mute = 0.0;
        size_t n = std::min(global.size(), affect.size());
        for (size_t i = 0; i < n; ++i) mute += global[i] * affect[i];
        return mute;
    }

    static double applyMute(double base_output, double mute, double mute_cap = 3.0) {
        double factor = 1.0 + mute;
        if (factor < 0.0) factor = 0.0;      // full suppression floor
        if (factor > mute_cap) factor = mute_cap;
        return base_output * factor;
    }

    // ------------- Phase step functions (default = basic neuron) -------------
    // Called once per tick by the Brain, in the order: receive, compute, send
    // are each their own global pass so all neurons finish one phase before
    // any neuron starts the next (keeps ordering well-defined).

    // Pull cached messages into this tick's inbox at the start of Receive phase.
    virtual void onReceiveEnter() {
        while (!cached_inbox.empty()) {
            inbox_this_tick.push_back(cached_inbox.front());
            cached_inbox.pop_front();
        }
    }

    // After Receive phase ends, move to Compute.
    virtual void onReceiveExit() {
        phase = Phase::Compute;
    }

    // Perform the actual computation. Default: pick strongest-magnitude
    // message (collision rule), apply chemical mute, store as pending_output.
    virtual void onCompute(const ChemicalEnvironment& env) {
        double strongest = 0.0;
        bool any = false;
        for (double s : inbox_this_tick) {
            if (!any || std::fabs(s) > std::fabs(strongest)) {
                strongest = s;
                any = true;
            }
        }
        double mute = muteFactor(env.levels, affectation);
        pending_output = any ? applyMute(strongest, mute) : 0.0;

        inbox_this_tick.clear();
        phase = Phase::Send;
    }

    // Send pending_output to all outgoing connections. Returns true when
    // this neuron is actually ready to send this tick (base neurons always are).
    virtual bool readyToSend() const { return true; }

    virtual void onSendExit() {
        phase = Phase::Receive;
    }
};

// ============================================================================
// MegaNeuron
// ============================================================================
// Same external shape (n inputs -> m outputs) but internally a strict
// feedforward stack of basic Neurons, one tick per layer during Compute.
// Seed behavior: pass the strongest input through to all outputs (near
// identity), so it starts as a stable no-op and can drift from there.
class MegaNeuron : public Neuron {
public:
    MegaNeuron(size_t num_inputs, size_t num_outputs, size_t num_layers,
               size_t num_chemicals, std::mt19937& rng)
        : Neuron(num_chemicals, rng), n_inputs(num_inputs), n_outputs(num_outputs)
    {
        num_layers = std::max<size_t>(1, num_layers);
        // layer_sizes: input width -> ... -> output width, num_layers hidden stops.
        // Simplest v1: every layer has n_outputs width except the first, which
        // reads n_inputs. This keeps the "pass strongest input to all outputs"
        // seed behavior trivial to implement layer-by-layer.
        layers.resize(num_layers);
        for (size_t l = 0; l < num_layers; ++l) {
            size_t in_w = (l == 0) ? n_inputs : n_outputs;
            layers[l].weights.assign(n_outputs, std::vector<double>(in_w, 0.0));
            layers[l].bias.assign(n_outputs, 0.0);
            // Seed: each output = the strongest single input, achieved by
            // setting all weights to 0 initially; the "strongest passthrough"
            // is instead implemented explicitly in computeLayer() when
            // untouched (see use_passthrough_seed).
        }
        use_passthrough_seed = true;
        current_layer_input.assign(n_inputs, 0.0);
    }

    size_t n_inputs, n_outputs;

    struct Layer {
        std::vector<std::vector<double>> weights; // [out][in]
        std::vector<double> bias;                 // [out]
    };
    std::vector<Layer> layers;
    bool use_passthrough_seed;

    // Raw multi-value input for this tick (gathered during Receive).
    std::vector<double> current_layer_input;
    std::vector<double> current_layer_output;

    // Which layer we're currently computing (reset at start of Compute phase).
    size_t active_layer = 0;

    void onReceiveEnter() override {
        Neuron::onReceiveEnter();
    }

    void onCompute(const ChemicalEnvironment& env) override {
        // First entry into compute this "compute session": gather inputs.
        if (mega_compute_ticks_remaining == 0 && active_layer == 0) {
            // Take one value per incoming path this tick (collision rule per input slot
            // is handled the same way as base neuron: strongest magnitude per slot).
            // For simplicity in v1, we treat each inbox message as belonging to
            // input slot (index in inbox order); real wiring truncates/pads.
            std::fill(current_layer_input.begin(), current_layer_input.end(), 0.0);
            for (size_t i = 0; i < inbox_this_tick.size() && i < n_inputs; ++i) {
                current_layer_input[i] = inbox_this_tick[i];
            }
            inbox_this_tick.clear();
            mega_compute_ticks_remaining = static_cast<int>(layers.size());
        }

        // Compute exactly one layer this tick.
        Layer& L = layers[active_layer];
        current_layer_output.assign(n_outputs, 0.0);

        if (use_passthrough_seed) {
            double strongest = 0.0;
            bool any = false;
            for (double v : current_layer_input) {
                if (!any || std::fabs(v) > std::fabs(strongest)) { strongest = v; any = true; }
            }
            for (size_t o = 0; o < n_outputs; ++o) current_layer_output[o] = strongest;
        } else {
            for (size_t o = 0; o < n_outputs; ++o) {
                double acc = L.bias[o];
                for (size_t i = 0; i < current_layer_input.size(); ++i) {
                    acc += L.weights[o][i] * current_layer_input[i];
                }
                current_layer_output[o] = std::tanh(acc); // simple nonlinearity
            }
        }

        current_layer_input = current_layer_output; // feed into next layer
        active_layer++;
        mega_compute_ticks_remaining--;

        if (mega_compute_ticks_remaining <= 0) {
            // Apply chemical mute once, at the end, to the final layer output.
            double mute = muteFactor(env.levels, affectation);
            for (auto& v : current_layer_output) v = applyMute(v, mute);
            pending_output = n_outputs > 0 ? current_layer_output[0] : 0.0; // primary output slot
            final_outputs = current_layer_output;
            active_layer = 0;
            phase = Phase::Send;
        }
        // else: stay in Compute phase for another tick (Brain must not
        // advance phase past Compute while mega_compute_ticks_remaining > 0)
    }

    std::vector<double> final_outputs; // full output vector, for multi-output fanout

    bool readyToSend() const override {
        return mega_compute_ticks_remaining <= 0;
    }
};

// ============================================================================
// InputNeuron
// ============================================================================
// Same shape as MegaNeuron but its "inputs" are externally fed scalars
// (e.g. raycast distances) rather than messages from other neurons.
class InputNeuron : public MegaNeuron {
public:
    InputNeuron(std::string name_, size_t num_inputs, size_t num_outputs,
                size_t num_layers, size_t num_chemicals, std::mt19937& rng)
        : MegaNeuron(num_inputs, num_outputs, num_layers, num_chemicals, rng),
          name(std::move(name_)) {}

    std::string name;

    // Called by Brain once per tick (during Receive) with the latest external
    // reading for this named input. All instances sharing a name receive the
    // same external_values vector.
    void setExternalInput(const std::vector<double>& external_values) {
        pending_external = external_values;
        has_external = true;
    }

    void onReceiveEnter() override {
        // Ignore normal message-based inbox; use externally injected values instead.
        if (has_external) {
            inbox_this_tick = pending_external;
            has_external = false;
        }
    }

private:
    std::vector<double> pending_external;
    bool has_external = false;
};

// ============================================================================
// OutputNeuron
// ============================================================================
// Same shape as MegaNeuron, but its outputs are not wired to other neurons;
// an external interpreter reads final_outputs once ready and turns them into
// actions. Named instances resolve by magnitude across instances (done by Brain).
class OutputNeuron : public MegaNeuron {
public:
    OutputNeuron(std::string name_, size_t num_inputs, size_t num_outputs,
                 size_t num_layers, size_t num_chemicals, std::mt19937& rng)
        : MegaNeuron(num_inputs, num_outputs, num_layers, num_chemicals, rng),
          name(std::move(name_)) {}

    std::string name;
};

// ============================================================================
// Brain: owns all neurons, runs the tick loop, manages named groups
// ============================================================================
class Brain {
public:
    explicit Brain(size_t num_chemicals) : env(num_chemicals) {}

    ChemicalEnvironment env;
    std::vector<std::unique_ptr<Neuron>> neurons;

    // name -> list of neuron indices sharing that name (for input/output neurons)
    std::unordered_map<std::string, std::vector<NeuronId>> input_groups;
    std::unordered_map<std::string, std::vector<NeuronId>> output_groups;

    NeuronId addNeuron(std::unique_ptr<Neuron> n) {
        neurons.push_back(std::move(n));
        return neurons.size() - 1;
    }

    void registerInput(const std::string& name, NeuronId id) {
        input_groups[name].push_back(id);
    }
    void registerOutput(const std::string& name, NeuronId id) {
        output_groups[name].push_back(id);
    }

    void connect(NeuronId from, NeuronId to) {
        neurons[from]->outgoing.push_back(to);
    }

    // Feed the same external reading to every instance of a named input group.
    void feedInput(const std::string& name, const std::vector<double>& values) {
        auto it = input_groups.find(name);
        if (it == input_groups.end()) return;
        for (NeuronId id : it->second) {
            auto* in = dynamic_cast<InputNeuron*>(neurons[id].get());
            if (in) in->setExternalInput(values);
        }
    }

    // Resolve a named output group to a single action vector: across all
    // instances sharing the name, take the strongest-magnitude value at
    // each output slot independently.
    std::vector<double> resolveOutput(const std::string& name) const {
        auto it = output_groups.find(name);
        if (it == output_groups.end()) return {};

        std::vector<double> result;
        for (NeuronId id : it->second) {
            auto* out = dynamic_cast<OutputNeuron*>(neurons[id].get());
            if (!out) continue;
            const auto& fo = out->final_outputs;
            if (result.size() < fo.size()) result.resize(fo.size(), 0.0);
            for (size_t i = 0; i < fo.size(); ++i) {
                if (std::fabs(fo[i]) > std::fabs(result[i])) result[i] = fo[i];
            }
        }
        return result;
    }

    // One full tick: global Receive pass, global Compute pass, global Send pass.
    // Each neuron only actually advances phase when its own state says it's done
    // (this matters for MegaNeurons mid multi-tick compute).
    void tick() {
        // --- Receive phase: only neurons currently in Receive act ---
        for (auto& n : neurons) {
            if (n->phase == Phase::Receive) {
                n->onReceiveEnter();
                n->onReceiveExit(); // -> Compute
            }
        }

        // --- Compute phase: only neurons currently in Compute act ---
        for (auto& n : neurons) {
            if (n->phase == Phase::Compute) {
                n->onCompute(env);
                // onCompute internally decides whether to move to Send
                // (base neurons and finished mega-neurons do; in-progress
                // mega-neurons stay in Compute).
            }
        }

        // --- Send phase: only neurons currently in Send and ready act ---
        for (size_t id = 0; id < neurons.size(); ++id) {
            auto& n = neurons[id];
            if (n->phase == Phase::Send && n->readyToSend()) {
                // MegaNeuron/InputNeuron/OutputNeuron have full vector outputs
                // available via final_outputs when multi-output fanout matters;
                // base Neuron just uses pending_output.
                auto* mega = dynamic_cast<MegaNeuron*>(n.get());
                if (mega && !mega->final_outputs.empty() && mega->n_outputs > 1) {
                    // Fan out per-output-slot round robin across outgoing paths,
                    // or broadcast full vector if outgoing count matches. v1: broadcast
                    // the primary output scalar to all outgoing paths (simple case),
                    // real multi-output routing can be refined later.
                    for (NeuronId to : n->outgoing) {
                        neurons[to]->deliverMessage(mega->pending_output);
                    }
                } else {
                    for (NeuronId to : n->outgoing) {
                        neurons[to]->deliverMessage(n->pending_output);
                    }
                }
                n->onSendExit(); // -> Receive
            }
        }
    }
};
