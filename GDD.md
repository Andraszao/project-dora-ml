# Game Design Document: Artificial Life Ecosystem Simulator

**Status:** Draft v0.1 · **Author:** Zeyphrus @ Inchworm Games · **Date:** 2026-07-17

> **Design thesis:** *A game about creating the conditions for life, not scripting the behavior of life.*

This document proposes a modern artificial-life game in the spirit of *Creatures* — but built as a **simulation platform where emergent life is the gameplay** rather than a pet-raising toy. It deliberately builds on the neural-network foundation already established in **Project Dora** (this repository): a from-scratch feedforward network with backpropagation, the Adam optimizer, online human-in-the-loop learning, and data augmentation, all running in Godot with no black-box ML libraries. Where the outline below touches technical claims, it is anchored to what Dora already demonstrates, so the ambition stays honest.

---

## 1. High Concept

| Field | Value |
|---|---|
| **Working title** | *[Placeholder]* — codename **DORA-Life** internally |
| **Genre** | Artificial Life Simulation / God Game / Ecosystem Sandbox |
| **Platform target** | Desktop first (Godot 4.4+), the same engine Dora runs on |
| **Player count** | Single-player; optional shared-world / lineage-export later |

**Player fantasy:**

> *"Raise, observe, influence, and evolve a living species whose behaviors emerge from biology, learning, environment, and generations."*

The player does **not** control creatures directly. The player shapes the conditions that create life. A creature is temporary; **the species is the protagonist.**

---

## 2. Core Design Pillars

### Pillar 1 — Emergent Intelligence
Creatures must not execute authored behaviors. Behavior emerges from the interaction of internal state, perception, memory, and learned associations.

```
Avoid:                         Prefer:
IF hungry:                     internal state
    find food                    + environmental perception
                                 + memory
                                 + learned associations
                                 = behavior
```

**Dora grounding:** Dora already replaces hand-authored classification rules with a learned 784→128→10 network. The same principle — *outcomes from learned weights, not from `if` ladders* — is the seed of this pillar. A creature brain is a smaller version of the same machinery, run per-agent.

### Pillar 2 — Individual Identity
Two creatures born in identical conditions should diverge. Individuality comes from a genome that biases the brain, plus divergent lived experience.

| Category | Examples |
|---|---|
| Personality | Curious · cautious · aggressive |
| Learning | Fast / slow learner (≈ per-agent learning rate) |
| Memory | Strong / weak retention |
| Social | Solitary / group-oriented |
| Metabolism | Efficient / inefficient |
| Temperament | Calm / anxious |

**Dora grounding:** Dora's `learning_rate`, Xavier-scaled random init, and Adam momentum are exactly the knobs that make "fast learner vs. slow learner" a real, measurable difference rather than a cosmetic label.

### Pillar 3 — Generational Evolution
The player should care about lineage. Selection pressure over many generations, not any single creature, is where the story lives.

---

## 3. Creature Architecture

```
CREATURE
├── Genome        (potential — heritable, mutable)
├── Body          (physical form + capabilities)
├── Brain         (perception → decision → action)
├── Physiology    (energy, health, hormones, stress)
├── Memory        (episodic + associative)
└── Behavior      (emergent output — never stored, always computed)
```

The genome defines **potential**; the brain and body express it; experience and physiology modulate it moment to moment.

---

## 4. Genome System

The genome is a fixed-length vector of genes, split into three functional groups.

### Physical genes
Body size · shape · coloration · movement speed · strength · vision range · lifespan.

### Neural genes (the interesting part)
The genome does not store a brain — it stores the **hyperparameters that build one**:

| Gene | Analogue in Dora |
|---|---|
| Number of neurons | `hidden_size` (Dora fixes this at 128; here it is heritable) |
| Connection density | sparsity of the weight matrices |
| Learning rate | `learning_rate` / Adam step |
| Memory capacity | replay-buffer / memory-store size |
| Curiosity bias | exploration term in the decision layer |
| Fear response | gain on threat-sensor inputs |
| Social affinity | weight prior on conspecific-detection inputs |

### Mutation
- **Rate:** 0.1%–5% per gene per reproduction (tunable global parameter).
- **Types:** cosmetic · behavioral · physiological · neural.
- Neural mutations change the *shape and biases* of the brain that gets built at birth; the brain then still has to **learn** within that shape.

> **Design note:** Separating "inherited architecture" (genome → network shape/biases) from "learned weights" (lifetime experience) is the central technical bet. It is the difference between *evolution* and *learning*, and the game needs both to feel alive.

---

## 5. Body Simulation

### Energy system
Every action has a cost. Energy is the universal currency.

```
Energy  =  + food intake
           - movement
           - growth
           - reproduction
           - basal metabolism (constant drain)
```

### Biological systems
- **Minimum viable:** energy · health · age · temperature · stress · hunger · fatigue.
- **Advanced (later phases):** immune system · hormones · disease · adaptation.

Hormones and stress are the bridge between physiology and brain: they scale sensory gains and decision thresholds, so a starving creature *perceives and decides differently* without any special-case code.

---

## 6. Sensory System

Creatures perceive the world through signals — not necessarily human senses. Sensors are the **input vector** to the brain (the direct analogue of Dora's 784 pixel inputs).

### Vision
Parameters: range · resolution · field of view · object recognition · movement detection.
Outputs: food detected · threat detected · creature detected · object detected.

### Chemical sense (very important)
Food scent · danger scent · mate scent · territory markers. Diffusing scent fields make the world legible to simple brains without requiring vision.

### Internal sensors
Hunger · pain · fear · pleasure · energy · temperature · social need.

> **Dora grounding:** Dora normalizes and flattens a 28×28 grid into a 784-vector before the network sees it. The creature's perception layer does the same job — it turns a rich world into a fixed-width, normalized input vector the brain can consume every tick.

---

## 7. Brain Architecture

A **hybrid** model: a small learned network wrapped by hard instincts and a memory store. This keeps per-agent cost low enough to run thousands of creatures.

```
INPUTS (sensor vector)
        │
        ▼
  Neural Network Layer   ← learned, per-agent (Dora-style feedforward)
        │
        ▼
   Decision System       ← arbitration + instinct override
        │
        ▼
      ACTIONS
```

### Components
- **Instinct layer** — hard-coded survival floors: avoid death, seek energy, reproduce, protect offspring. These act as a safety net and as reward shaping.
- **Learning layer** — experience modifies behavior via reinforcement:
  ```
  action → outcome → reward → weight adjustment → future choice changes
  ```
  This is Dora's online-learning loop repurposed: instead of a human correcting a digit, the **environment** supplies the reward signal.
- **Memory layer** — stores locations, creatures, events, threats, food sources, and relationships; feeds back into the input vector as remembered context.

> **Feasibility check (from Dora):** Dora's network is ~101k parameters and runs real-time inference on CPU in GDScript. A creature brain would be *smaller* (tens of inputs, one modest hidden layer). Thousands of such brains are tractable; the design's job is to keep each brain small and let the **environment** carry the complexity.

---

## 8. World Simulation

### Environment
Size · climate · resources · terrain · weather · day/night · season.

### Ecosystem entities
Plants · herbivores · predators · parasites · resources.

### Resource dynamics (the core feedback loop)
```
Plants grow → creatures eat → population rises →
plants decline → population crashes → adaptation → …
```
These boom/bust cycles are the pressure that drives selection. The player tunes the loop; evolution responds.

---

## 9. Social System

### Relationships
Creatures remember family · friends · enemies · mates · teachers.

### Communication
- **Phase 1 — signals:** danger call · food signal · mate signal · territory signal (fixed vocabulary, learned *usage*).
- **Phase 2 — emergent language:** signal meanings drift and specialize through use.
- **Phase 3 (optional/experimental):** an LLM-assisted narration or translation layer that *describes* emergent behavior to the player — never a per-creature brain.

---

## 10. Player Interaction

The player is a **caretaker / researcher**, not a puppeteer.

- **Direct (deliberately limited):** feed · heal · move · name · breed.
- **Indirect (primary):** modify environment · introduce species · change climate · provide resources · create challenges.

The design intentionally under-powers direct control so that outcomes read as *the species adapting*, not *the player winning*.

---

## 11. Core Gameplay Loop

```
Observe → discover behavior → influence environment →
creatures adapt → population changes → new behaviors emerge → repeat
```

---

## 12. Success Metrics

Not "beat level 10." Instead, **emergence metrics**:

- Species survival (generations sustained)
- Behavioral diversity
- Genetic diversity
- Social complexity
- Adaptation success (measurable response to introduced pressure)

These double as the game's telemetry and its scoring, the same way Dora logs an accuracy trajectory rather than a pass/fail.

---

## 13. Example Player Stories

1. *"I accidentally created a timid species because predators killed the aggressive individuals for 50 generations."*
2. *"One creature learned that I provided food, and eventually the whole population became domesticated."*
3. *"A mutation improved vision, and that branch outcompeted everyone else."*

Each story is an outcome the systems *permit* but never script.

---

## 14. Technical Reality Check (2026)

A realistic modern stack:

```
Simulation engine (Godot)
  + ECS architecture
  + per-agent neural networks   ← Dora already proves this is feasible in-engine
  + reinforcement learning       ← Dora's online-correction loop, env-driven reward
  + genetic algorithms           ← genome → brain shape, mutation, selection
  + LLM-assisted narration layer (optional, world-level, not per-creature)
```

**The key constraint:** do **not** give every creature a large model. A million creatures with frontier-model brains is impossible and unnecessary. The magic comes from **thousands of simple agents** in **rich environments**, with **memory**, **evolution**, and **feedback loops**.

**Why this repo is the right starting point:** Project Dora already demonstrates, in Godot and without external ML dependencies, the three hardest primitives this game needs —
1. a working feedforward network with backpropagation,
2. an online learning loop that updates weights from feedback in real time, and
3. augmentation/robustness so learned behavior generalizes beyond memorized cases.

The GDD's job from here is to multiply that single brain into a population, replace human correction with environmental reward, and let genetics decide the shape of each brain.

---

## 15. Phased Roadmap

| Phase | Goal | Builds on |
|---|---|---|
| **P0 — Single brain** | *(done: Project Dora)* one learning network, online correction | this repo |
| **P1 — One creature** | one agent: sensors → Dora-style brain → actions in a toy world; energy + death | Dora net, new sensor/action I/O |
| **P2 — Population** | many agents, reproduction, genome → brain shape, mutation | P1 + genetic algorithm |
| **P3 — Ecosystem** | plants/predators, resource boom-bust, selection pressure | P2 + world sim |
| **P4 — Society** | memory of relationships, signals, emergent communication | P3 + social layer |
| **P5 — Platform** | player tools, metrics dashboard, lineage export, optional narration | all of the above |

---

## 16. Open Design Questions

- **Reward design:** environmental reward is notoriously easy to game. How much instinct scaffolding is too much before behavior stops feeling emergent?
- **Compute budget:** what is the real per-tick cost of N Dora-scale brains in GDScript, and where is the crossover point that forces batching, a compute shader, or a C++/GDExtension backend?
- **Determinism & reproducibility:** lineage stories require reproducible runs (seeded RNG). Dora already saves weights and a training-journey log; the sim needs the same for whole populations.
- **Legibility:** emergence is worthless if the player can't *read* it. How much of P5's job is visualization and narration versus simulation depth?

---

*Made for learners, by a learner. This is a design sketch, not a promise — but every technical claim in it is one step from something Project Dora already does.*
