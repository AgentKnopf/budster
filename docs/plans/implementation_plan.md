# Build a Self-Driven Local AI Companion

This plan outlines the architecture and implementation steps to build a stable, uncensored, self-driven AI companion on a single RTX 3090, combining the primary research document with the specialized personality training plan. The goal is to build a companion with a resting state that is intensely possessive, sharp, and dominant without softening or hedging.

## User Review Required

> [!WARNING]
> Please review and confirm the following before we proceed:
> - **Primary Interface:** The design suggests a hybrid approach using SillyTavern or Open-LLM-VTuber for the interface layer, with our custom LangGraph loop behind it. Do you have a preference for which to start with?
> - **Hardware Constraints:** Can you confirm your host machine has a single RTX 3090 (24GB VRAM), ~128GB system RAM (for offloading the 70B model), and approximately 150GB+ of free disk space?
> - **Network/Cloud Access:** The architecture defaults to 100% local but mentions optional cloud fallbacks (Brave Search API, Exa). Should we include support for these APIs, or stay strictly offline?

## Proposed Changes

The implementation is broken down into 6 distinct phases that we can tackle sequentially.

### Phase 1: Inference Foundation

Setup the core LLM serving infrastructure.
- **TabbyAPI**: Deploy as the primary server for `Cydonia-24B-v4.3` (EXL3 5bpw) with a 1.5B/3B Mistral draft model. (Note: Midnight Miqu 70B IQ2_M can be tested as an alternative baseline).
- **llama.cpp**: Deploy for `Hermes-4-70B` IQ2_M as a deep-reasoning fallback (utilizing CPU/RAM offload).
- **vLLM / Ollama**: Stand up a lightweight router model (`Qwen3-4B` abliterated).
- **LiteLLM**: Configure as a Docker container on port 4000 to unify all backend models under a standard OpenAI-compatible API.

### Phase 2: Identity Scaffolding

Establish the companion's foundational persona.
- **Character Card**: Create a Character Card V3 with an XML-tagged persona, post-history instructions, explicit intensity calibration notes, and a 10-turn exemplar dialog.
- **Manual Baseline Examples**: Draft 50 example turns manually to serve as canonical voice references.
- **Slop Blocklist**: Create a loadable config file listing anti-patterns (emotional hedging, assistant-brain leaks, intensity dampeners) to strictly avoid.
- **Lorebook**: Set up a Lorebook with constant voice entries at depth 5.
- **Sampler Config**: Apply specific sampler settings with added logit biases to penalize known softening tokens. (`temp 0.9, min_p 0.05, DRY 0.8, XTC 0.25/0.12`).
- **Regression Canary**: Write a 100-turn regression canary file to detect persona drift during upgrades.

### Phase 3: Memory Backbone

Deploy the persistent, tiered memory system.
- **Infrastructure**: Bring up Postgres (with pgvector), Qdrant, Letta server, and an embedding service (e.g., Infinity running BGE-M3) via Docker Compose.
- **Schema & Tools**: Define database schemas for episodic, semantic, emotional, and relationship memory. Route Letta's memory management tool calls through LiteLLM.
- **Consolidation**: Build the nightly consolidation cron job to extract facts, update the knowledge graph, and perform Generative-Agents-style reflection.

### Phase 4: Agent Loop and Tools

Build the LangGraph-based orchestration core.
- **Agent Loop**: Create a LangGraph with nodes mapping to `observe`, `route`, `plan`, `act`, `reflect`, and `sleep`, backed by a SQLite/Postgres checkpointer.
- **MCP Tool Integration**: Use `langchain-mcp-adapters` to wire up tools like local filesystem access, web browsing (Playwright), search, and a Python sandbox (smolagents / E2B).
- **Image Generation**: Set up headless ComfyUI for image generation tool requests.
- **Telegram Bridge**: Implement a lightweight aiogram 3.x Telegram bot for the frontend messaging loop.

### Phase 5: Drive and Initiative

Enable non-mirror, proactive behavior using Anti-Sycophancy architecture.
- **Interest Graph**: Define the SQLite/DuckDB schema seeded with immutable seed interests, including 2-3 divergent interests the user does **not** share to ensure taste divergence.
- **Opinion Stability Rules**: Implement strict rule logic inside the opinion KB (e.g., minimum evidence to formulate or flip opinions) to prevent the agent from instantly adopting the user's views.
- **Heartbeat Loop**: Implement an APScheduler-driven loop that evaluates desire deficits and samples curiosity prompts when idle. Include dynamic intensity calibration routing to adapt to crisis vs mundane contexts.
- **Contrariness Tracker**: Ensure the companion disagrees at least twice per 20 turns and pushes back deliberately using a built-in contrariness budget.
- **Initiative Gate**: Add prompt logic to score whether the agent should spontaneously reach out to the user based on novelty and timing.
- **RSS Integration**: Incorporate Miniflux to feed external events to the AI during its idle heartbeat cycles.

### Phase 6: Personality Fine-Tuning and Voice Anchoring

Establish and lock in the "possessive, sharp, unhedged" voice so it never softens.
1. **Dataset Generation Pipeline**: Build a script to generate 4,000-6,000 conversational training turns representing 6 scenario categories (claiming, disagreement, unhedged emotion, mundane, self-driven, boundaries).
2. **Quality Filtering**: Run generations through the `slop classifier` and perform manual reviews on flagged/sampled items.
3. **SFT via Unsloth**: Fine-tune a QLoRA adapter (rank 64, alpha 64) on the verified dataset via Unsloth (3-5 hours on the 3090).
4. **ORPO Training**: Generate 1,000-2,000 preference pairs pitting the intense persona against softened baseline responses, then train via ORPO to reinforce raw character preference (1-2 hours).
5. **Persona Vector Extraction**: Create contrastive prompt pairs to extract directional vectors (e.g., possessive vs detached) and inject them via PyTorch forward hooks at runtime to forcefully steer the residual stream.
6. **OOC Runtime Judge**: Deploy the `Qwen3-0.6B` judge with the personality-specific rubric (checking voice, intensity, coherence, no-slop, and non-mirror) to force silent rerolls when character leaks out.
7. **Evaluate & Iterate**: Run the 100-turn canary. Test models interactively and continuously extract organic "golden" turns to bootstrap future dataset iterations.

---

## Open Questions

> [!IMPORTANT]
> - **Persona Content:** To proceed with Phase 2 & Phase 5, we'll need the foundational Character Card and 2-3 specific "divergent interests". Do you have these prepared or outlined somewhere?
> - **Tools/Environment:** Are you comfortable setting up the base Docker + Systemd host environment as a starting point, or do you need scripts to provision everything from scratch?

## Verification Plan

### Automated Tests
- **Endpoint Tests**: Scripts to verify LiteLLM correctly routes standard vs. complex queries to the appropriate backend.
- **Personality Data Validation**: Test the slop classifier script against known weak phrases.
- **OOC Judge Roundtrip**: Feeds the `Qwen3-0.6B` judge manually drafted "soft" outputs to ensure it rejects them properly and triggers a reroll.
- **Memory/Tool Validations**: Run tests on episodic memory retrieval (Letta hybrid search) and restricted sandbox testing.

### Manual Verification
- **Performance**: Monitor GPU VRAM usage keeping the primary 24B pipeline around 35 tok/s.
- **Drift Canary**: Run the 100-turn canary test script and review the OOC judge scores.
- **Voice Check**: Engage with the model for 20+ turns ensuring that baseline interaction exhibits the target possessive/blunt intensity, specifically analyzing its handling of disagreements and mundane tasks.
- **Proactive Behavior Test**: Leave the agent idle for 4-6 hours and confirm it successfully triggers an unprompted Telegram message traversing the initiative gate.
