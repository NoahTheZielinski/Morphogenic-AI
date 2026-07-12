#include "brain.hpp"
#include <iostream>
#include <iomanip>

// Small helper to print a labeled separator.
static void section(const std::string& title) {
    std::cout << "\n=== " << title << " ===\n";
}

int main() {
    std::mt19937 rng(42);

    // ------------------------------------------------------------------
    // Test 1: Basic neuron chain propagation + collision rule
    // ------------------------------------------------------------------
    // Chain: A -> B -> C
    // A has no inputs (we inject a message directly), tick-by-tick we should
    // see the signal reach B, then C, one hop per tick, respecting the
    // receive/compute/send phase timing.
    section("Test 1: basic chain propagation timing");
    {
        Brain brain(1); // 1 chemical, all neutral (0.0) so mute = 0 -> no distortion
        NeuronId a = brain.addNeuron(std::make_unique<Neuron>(1, rng));
        NeuronId b = brain.addNeuron(std::make_unique<Neuron>(1, rng));
        NeuronId c = brain.addNeuron(std::make_unique<Neuron>(1, rng));
        brain.connect(a, b);
        brain.connect(b, c);

        // Force A's affectation to 0 so mute=0 regardless of chemical env.
        brain.neurons[a]->affectation = {0.0};
        brain.neurons[b]->affectation = {0.0};
        brain.neurons[c]->affectation = {0.0};

        // Inject a message directly into A's inbox (simulating some upstream source).
        brain.neurons[a]->deliverMessage(5.0);

        for (int t = 0; t < 6; ++t) {
            std::cout << "tick " << t
                      << " | A.phase=" << static_cast<int>(brain.neurons[a]->phase)
                      << " A.out=" << brain.neurons[a]->pending_output
                      << " | B.phase=" << static_cast<int>(brain.neurons[b]->phase)
                      << " B.out=" << brain.neurons[b]->pending_output
                      << " | C.phase=" << static_cast<int>(brain.neurons[c]->phase)
                      << " C.out=" << brain.neurons[c]->pending_output
                      << "\n";
            brain.tick();
        }
        // Expect: signal visibly at B's output around tick 2-3, at C's around tick 4-5.
    }

    // ------------------------------------------------------------------
    // Test 2: Chemical muting formula
    // ------------------------------------------------------------------
    section("Test 2: chemical muting");
    {
        Brain brain(2); // 2 chemicals
        NeuronId n = brain.addNeuron(std::make_unique<Neuron>(2, rng));
        brain.neurons[n]->affectation = {1.0, -1.0}; // excite via chem0, suppress via chem1

        auto runOnce = [&](double chem0, double chem1, double input) {
            brain.env.levels = {chem0, chem1};
            brain.neurons[n]->phase = Phase::Receive;
            brain.neurons[n]->cached_inbox.clear();
            brain.neurons[n]->inbox_this_tick.clear();
            brain.neurons[n]->deliverMessage(input);
            brain.tick(); // receive+compute -> should land in Send with pending_output set
            std::cout << "chem=[" << chem0 << "," << chem1 << "] input=" << input
                      << " -> mute=" << Neuron::muteFactor(brain.env.levels, brain.neurons[n]->affectation)
                      << " output=" << brain.neurons[n]->pending_output << "\n";
            brain.tick(); // send -> back to receive
        };

        runOnce(0.0, 0.0, 10.0);   // mute=0 -> output=10
        runOnce(1.0, 0.0, 10.0);   // mute=+1 -> output=20
        runOnce(0.0, 1.0, 10.0);   // mute=-1 -> output=0 (floor)
        runOnce(0.5, 0.5, 10.0);   // mute=0 -> output=10
    }

    // ------------------------------------------------------------------
    // Test 3: MegaNeuron multi-tick compute (passthrough seed) + Input/Output
    // ------------------------------------------------------------------
    section("Test 3: MegaNeuron with InputNeuron -> OutputNeuron, 3 layers");
    {
        Brain brain(1);
        // "vision" input: 5 raycast values -> 3 outputs, 1 layer (pure passthrough)
        NeuronId vis = brain.addNeuron(
            std::make_unique<InputNeuron>("vision", 5, 3, 1, 1, rng));
        brain.registerInput("vision", vis);

        // A mega-neuron hub: 3 inputs -> 2 outputs, 3 layers (multi-tick compute)
        NeuronId hub = brain.addNeuron(
            std::make_unique<MegaNeuron>(3, 2, 3, 1, rng));

        // "move" output: 2 inputs -> 2 outputs, 1 layer
        NeuronId move = brain.addNeuron(
            std::make_unique<OutputNeuron>("move", 2, 2, 1, 1, rng));
        brain.registerOutput("move", move);

        brain.connect(vis, hub);
        brain.connect(hub, move);

        brain.feedInput("vision", {1.0, 2.0, -7.0, 3.0, 0.5}); // strongest magnitude = -7.0

        for (int t = 0; t < 12; ++t) {
            auto* visN = dynamic_cast<InputNeuron*>(brain.neurons[vis].get());
            auto* hubN = dynamic_cast<MegaNeuron*>(brain.neurons[hub].get());
            auto* movN = dynamic_cast<OutputNeuron*>(brain.neurons[move].get());
            std::cout << "tick " << t
                      << " | vision.phase=" << static_cast<int>(visN->phase)
                      << " vision.out=" << visN->pending_output
                      << " | hub.phase=" << static_cast<int>(hubN->phase)
                      << " hub.layer=" << hubN->active_layer
                      << " hub.ticks_left=" << hubN->mega_compute_ticks_remaining
                      << " hub.out=" << hubN->pending_output
                      << " | move.phase=" << static_cast<int>(movN->phase)
                      << " move.out=" << movN->pending_output
                      << "\n";
            brain.tick();
            // Re-feed vision every tick so it keeps producing (in a real sim the
            // environment would feed this continuously)
            brain.feedInput("vision", {1.0, 2.0, -7.0, 3.0, 0.5});
        }

        auto resolved = brain.resolveOutput("move");
        std::cout << "Resolved 'move' action vector: ";
        for (double v : resolved) std::cout << v << " ";
        std::cout << "\n";
    }

    // ------------------------------------------------------------------
    // Test 4: Named parallel output instances resolve by strongest magnitude
    // ------------------------------------------------------------------
    section("Test 4: parallel output instances, strongest-magnitude resolution");
    {
        Brain brain(1);
        NeuronId out1 = brain.addNeuron(std::make_unique<OutputNeuron>("thrust", 1, 1, 1, 1, rng));
        NeuronId out2 = brain.addNeuron(std::make_unique<OutputNeuron>("thrust", 1, 1, 1, 1, rng));
        brain.registerOutput("thrust", out1);
        brain.registerOutput("thrust", out2);

        // Manually force final_outputs to simulate two instances disagreeing.
        dynamic_cast<OutputNeuron*>(brain.neurons[out1].get())->final_outputs = {2.0};
        dynamic_cast<OutputNeuron*>(brain.neurons[out2].get())->final_outputs = {-9.0};

        auto resolved = brain.resolveOutput("thrust");
        std::cout << "thrust resolves to: " << resolved[0] << " (expect -9, strongest magnitude)\n";
    }

    std::cout << "\nAll tests ran without crashing.\n";
    return 0;
}
