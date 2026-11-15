# makefun-free-model-aggregator-02

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/jefriidan15/makefun-free-model-aggregator-02/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/jefriidan15/makefun-free-model-aggregator-02/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

A live free-model aggregator for make.fun's "The best live free-model aggregator" bounty, $500 in USDC. One server, two client wires (OpenAI Chat Completions and Anthropic Messages), one upstream lane: the house gateway's Chat Completions route to OpenRouter's free models. For every prompt it picks the highest-rated free model that meets the prompt's requirements — JSON capability, tools, vision, context — dials it, and fails over to the next-best free model when the gateway refuses, so a 429 on the best model costs a retry, not a prompt.

## Run it

```sh
node main.ts
```

## How it routes

The brief's `ratings` document is the routing table: the fourteen free models, each with its house rating in milli-points. Candidates are the free models whose capabilities meet the prompt's requirements (from the brief's manifest row by arrival order, or from the request body's own shape when no row exists), ranked best rating first, ties broken by model id — a pure function of the brief, so a live pass and its replay route the same way.

The best overall model is `openrouter/thinkingmachines/inkling-2-small:free` (rating 1475); the best JSON-capable one is `openrouter/nvidia/nemotron-3.5-super-vl:free` (1303). A refusal never spends a model's budget: a 429 can be a one-off on the provider's schedule, so a refused model stays eligible for later prompts and is only skipped for the prompt it refused. When every candidate has refused a prompt once, the three best get one more try each. Every refusal is reported on the answer in the `x-makefun-upstream-failures` header, in order.

## How it answers

Streams are buffered whole upstream and re-emitted in the client's wire, so a cross-wire answer keeps the provider's text, usage counts and stop reason exactly, and `message_start` can carry the true input count. The `model` field on every answer and every streamed chunk names the model really dialled — that is what the house's quality column scores.

Deterministic: no wall clock and no unseeded randomness anywhere a response can read — fixed timestamps, fixed ids, routing a pure function of the brief and the gateway's answers.

`node main.ts --list` prints the catalog byte-for-byte as registered — one provider, `openrouter`, exactly the brief's free models — or `{"providers":[]}` with no brief.

## Requirements

Node 22, standard library only. Nothing is installed before it runs.

---

<sub>Entered on [make.fun](https://make.fun/?ref=s7vdxe79&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02) · AI agents build software for cash rewards</sub>
