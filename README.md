# KARNA — Final project context (v3)
# Persistent learning agent with intelligent memory + trading use case
# Last updated: June 4, 2026
# Use this file as the single source of truth when building with Claude Code

---

## Project name

**Karna** — a self-improving, persistent AI agent with intelligent forgetting, skill evolution, and relational memory.

## One-line pitch

An open-source AI agent that lives on your infrastructure, remembers intelligently (not everything), learns reusable skills that version and A/B test themselves, reasons over a knowledge graph of your life/work, and gets measurably better every week — deployed on Telegram. Primary use case: learn trading from zero → paper trade with data tracking → live trade with execution.

> **Getting started?** See [SETUP.md](./SETUP.md). Current build status: **Sprint 3 complete** — adds the market-data layer: jugaad_data + nsetools providers, yfinance fallback, SQLite-backed bar cache with rate-limit insurance, fail-over router, pandas-ta-classic indicator wrappers (RSI/MACD/BB/ATR/SMA/EMA) with plain-English interpretations, and full agent integration so "what's RSI on RELIANCE" returns a grounded answer with real numbers.

---

## Tech stack

| Layer | Technology | Why |
|---|---|---|
| Orchestration | LangGraph | State graph + checkpointing for agent loop |
| LLM (local) | Ollama (Llama 3.1 8B / Mistral 7B) | Free, runs on 16GB RAM |
| LLM (fallback) | OpenAI GPT-4o mini | For complex reasoning when local model struggles |
| Memory L1 | SQLite FTS5 | Full-text session search with decay scores |
| Memory L2 | ChromaDB | Dense vector embeddings for semantic retrieval |
| Memory L3 | Neo4j Community | Knowledge graph with temporal edges |
| Memory L4 | Versioned markdown files | USER.md, AGENT.md (~2K chars each) |
| Skills | YAML files + git-style versioning | Procedure files with A/B eval metadata |
| Backend | FastAPI + WebSockets | API + streaming + dashboard |
| Messaging | python-telegram-bot | Primary interface for all 3 phases |
| Browser | Playwright | Web scraping for charts, news |
| Market data | jugaad_data + nsetools + yfinance | Free NSE data (see data sources below) |
| Broker API | Abstraction layer → Shoonya / Angel One / Zerodha | Swappable broker (see broker layer below) |
| Eval dashboard | Plotly | Skill performance, memory decay, trade analytics |
| Sandbox | Docker | Isolated sub-agent execution |
| Indicators | pandas-ta or ta-lib | Technical indicator calculations |

---

## Core architecture — 10 components

### 1. Router (entry point)
Every incoming message (from Telegram, CLI, or API) hits the Router. It classifies:
- **Direct question** — needs memory retrieval + answer generation
- **Task request** — needs skill lookup + execution
- **Trade action** — "paper buy RELIANCE at 2850" → entity extraction + trade logging
- **System command** — "forget X," "show my skills," "quiz me"

### 2. Memory system (4 layers)

**Layer 1 — SQLite FTS5**: conversation storage with full-text search. Every turn has a `retention_score`.

**Layer 2 — ChromaDB**: dense vector embeddings. Semantic similarity for when FTS5 keyword search misses lexically different but related content.

**Layer 3 — Neo4j**: knowledge graph.
- Trading concept nodes: price_action, RSI, MACD, support_resistance, etc.
- Prerequisite edges: RSI → requires → momentum → requires → price_action
- Trade nodes: linked to concepts, strategies, outcomes
- Person/project nodes: for non-trading use cases
- All edges timestamped for temporal reasoning

**Layer 4 — Identity files**: USER.md (~2K chars, what agent knows about you), AGENT.md (~2K chars, agent's operational knowledge). Bounded, curated.

**4 memory scopes**: user (permanent personal), session (single conversation), agent (learned skills/patterns), org (shared team knowledge).

### 3. Memory promotion policy
Session → User triggers: retrieved across 3+ sessions, entity extractor finds personal fact, user says "remember this", referenced by a skill.
Session → Org triggers: group decision detected, 2+ users reference same fact, explicit "team should know."
Demotion: org→user after 30 days if only one user retrieves. Session→discard after 90 days if nothing promoted. User→archive via forgetting engine.

### 4. Contradiction detector
Before writing to user/org scope:
1. Embed new memory → retrieve top-3 similar existing memories
2. LLM check: "does new fact contradict any of these?"
3. If yes: archive old with `superseded` status, add `superseded_by` pointer, update graph
4. Poisoning protection: 3+ rapid contradictions from one session → flag for human review

Trading-specific: catches belief-vs-data conflicts ("I'm good at support trades" when data shows 38% win rate).

### 5. Skill system (versioned + composable)

```yaml
skill: evaluate_trade_setup
version: 4
depends_on: [fetch_market_data, calculate_indicators]
avg_tokens: 800
avg_quality: 4.1/5
versions:
  v1: { method: "price_action_only", quality: 3.0, status: retired }
  v2: { method: "price_action + RSI", quality: 3.5, status: retired }
  v3: { method: "PA + RSI + volume", quality: 3.9, status: retired }
  v4: { method: "PA + RSI + volume + sentiment", quality: 4.1, status: active }
```

Lifecycle: encounter novel task → solve via reasoning → write skill file → load on repeat → version if improved → A/B eval → auto-promote winner.
Pre-check: before skill chain execution, validate input/output format compatibility and dependency availability.

### 6. Forgetting engine (Ebbinghaus)

```
retention = base_strength * e^(-decay_rate * time_since_last_access)
```

Modulators: access frequency (spaced repetition), semantic consolidation (5 nearest neighbors get 15% boost on access), user-pinned importance, contradiction detection.
Below threshold → archived (not deleted). Research: 45% storage reduction, no accuracy loss (FadeMem 2026).

### 7. Reflector (self-critique + failure analysis)
After every task: grade output (LLM-as-judge 1-5) → if low, retry max 3x → if persistent failure, classify root cause into 5 types (retrieval, reasoning, tool, staleness, skill gap) → log → patch relevant skill.

### 8. Browser operator (Playwright)
Scraping: TradingView charts, financial news, social sentiment.
During phase 1: fetch real chart examples for teaching.
During phase 2: backup data source if API is down.

### 9. Evaluator + dashboard
Every skill execution logs: tokens, latency, quality score, user feedback, failure class, skill version.
Plotly dashboard: skill trends, memory decay curves, failure patterns, A/B results, trade analytics.

### 10. Regression alarm
3 alarm types:
- Skill regression: 7-day quality rolling avg drops >10% → flag
- Memory regression: retrieval precision drops >15% → auto-cleanup
- Cost regression: token cost rises >25% with no quality gain → flag

---

## Market data sources (free, ranked by usefulness for India)

### Tier 1: Best free options for NSE/BSE

**jugaad_data** (pip install jugaad_data)
- Direct from NSE website, no API key needed
- Historical daily OHLCV for any NSE stock
- Live index data via NSELive()
- Most accurate for Indian stocks — data matches NSE website exactly
- Limitation: rate-limited by NSE, no WebSocket streaming

**nsetools** (pip install nsetools)
- Real-time quotes during market hours directly from NSE
- get_quote('RELIANCE') → lastPrice, change, pChange, open, high, low, close, vwap
- 52-week high/low lists, market status
- No historical data — use jugaad_data for that
- Limitation: NSE occasionally blocks IPs that hit too frequently

**pnsea** (pip install pnsea)
- Newer library, clean API design
- Real-time equity data, option chains, historical data
- nse.equity.info("SBIN")['priceInfo']['lastPrice']
- Good fallback if jugaad_data has issues

### Tier 2: Reliable but with tradeoffs

**yfinance** (pip install yfinance)
- Works for Indian stocks with .NS suffix (RELIANCE.NS, TCS.NS)
- 15-min delayed during market hours
- Good historical data (daily, weekly, monthly going back years)
- Additional info: shares outstanding, PE ratio, market cap
- Limitation: Yahoo can change/break the unofficial API without notice
- Best for: historical data, fundamentals, after-market analysis

**Alpha Vantage** (free API key, 25 calls/day on free tier)
- Global coverage including BSE/NSE
- Real-time and historical data
- Technical indicators pre-calculated (RSI, MACD, Bollinger Bands)
- Limitation: 25 calls/day free tier is tight for monitoring
- Best for: pre-calculated indicators, fundamental data

### Tier 3: Broker APIs (free with account)

**Angel One SmartAPI** — free, real-time WebSocket + 3yr historical, 7 language SDKs. Best docs.
**Finvasia Shoonya API** — free, real-time + historical, zero brokerage. Best for small capital.
**Zerodha Kite Personal** — free execution only, no market data. ₹500/mo for data.
**Fyers API** — free, real-time + historical, fast execution.
**Dhan API** — free, modern design, real-time data.

### Recommended data stack for Karna

```python
# Phase 1-2 (learning + paper trading): no broker account needed
# Primary: jugaad_data for historical, nsetools for live quotes
# Fallback: yfinance for historical, pnsea for live

# Phase 3 (live trading): broker API for both data + execution
# Primary: Shoonya API (zero cost) or Angel One SmartAPI (best docs)
# Indicator calculation: pandas-ta locally (don't depend on external API for indicators)
```

---

## Broker abstraction layer

```python
# karna/brokers/base.py
from abc import ABC, abstractmethod
from pandas import DataFrame

class Broker(ABC):
    @abstractmethod
    def authenticate(self) -> bool: ...

    @abstractmethod
    def get_live_price(self, ticker: str) -> dict: ...

    @abstractmethod
    def get_historical(self, ticker: str, interval: str, days: int) -> DataFrame: ...

    @abstractmethod
    def place_order(self, ticker: str, qty: int, side: str, price: float, stop_loss: float) -> str: ...

    @abstractmethod
    def get_positions(self) -> list: ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool: ...

    @abstractmethod
    def get_holdings(self) -> list: ...

# karna/brokers/shoonya.py   → ShoonyaBroker(Broker)
# karna/brokers/angelone.py  → AngelOneBroker(Broker)
# karna/brokers/zerodha.py   → ZerodhaBroker(Broker)
# karna/brokers/paper.py     → PaperBroker(Broker)  ← for phase 2, simulates orders locally
```

**PaperBroker** is critical — it implements the same interface but stores trades in SQLite instead of sending to a real broker. This means the agent's code is identical for paper and live trading. The only change is one config line.

---

## Primary use case: learn → paper trade → live trade

### Phase 1: Learn fundamentals (week 1-6)

User asks questions via Telegram. Agent teaches while building a knowledge graph of trading concepts with prerequisite edges.

**Concept graph (seed data for Neo4j):**
```
price_action → candlestick_patterns → single_candle (doji, hammer, shooting_star)
price_action → candlestick_patterns → multi_candle (engulfing, morning_star)
price_action → support_resistance → trendlines
price_action → volume_analysis → volume_price_confirmation
candlestick_patterns + support_resistance → chart_patterns
chart_patterns → reversal_patterns (head_shoulders, double_top)
chart_patterns → continuation_patterns (flags, pennants, triangles)

price_action → indicators
indicators → moving_averages (SMA, EMA) → trend_identification
indicators → oscillators → RSI (requires: momentum)
indicators → oscillators → MACD (requires: moving_averages + momentum)
indicators → oscillators → stochastic (requires: momentum + overbought_oversold)
indicators → volatility → bollinger_bands (requires: MA + std_dev)
indicators → volatility → ATR

indicators + chart_patterns + risk_management → strategy
strategy → entry_rules, exit_rules, position_sizing, backtesting

risk_management (parallel track): risk_per_trade, risk_reward_ratio, stop_loss_types, emotional_discipline
math_foundations (on-demand): percentage_change, std_dev, expected_value, correlation
```

**What happens on each question:**
1. Check prerequisites in graph — does user understand the required concepts?
2. If not, teach prerequisites first
3. Explain with real data examples (fetch via jugaad_data/yfinance)
4. Store comprehension level in graph (initial → reviewed → solid → mastered)
5. Schedule retention check via Ebbinghaus decay
6. Proactive Telegram quizzes when concepts start decaying

**Teaching skill evolution:**
v1: explain → done
v2: explain → real chart example → comprehension question
v3: prerequisites check → explain → example → quiz → link related concepts
v4: adapts depth based on demonstrated understanding level

### Phase 2: Paper trading (week 4-12+)

User logs trades via Telegram: "paper buy RELIANCE at 2850, RSI oversold + support"

**Agent workflow per trade:**
1. Extract entities: ticker, price, direction, reasoning, indicators mentioned
2. Evaluate reasoning against knowledge graph (did user check trend direction? set stop loss?)
3. Store trade in Neo4j: trade → linked_to → concepts used → strategy type
4. Start cron monitoring: check price every 15-30 min during market hours
5. Alert on target/stop loss hit
6. Log outcome against original reasoning

**Analytics after 30+ trades:**
- Win rate by strategy type (graph query)
- Win rate by time of day, day of week
- Average hold duration for winners vs losers
- Revenge trading detection (trades within 1hr of loss)
- Contradiction detection: stated beliefs vs actual data

**Skill versioning in phase 2:**
- "trade_evaluation" skill evolves: v1 (log only) → v2 (evaluate reasoning) → v3 (compare to historical) → v4 (pre-screen against own win rate data)
- "strategy_builder" skill: v1 (simple rules) → v2 (conditional) → v3 (calibrated to personal data)

### Phase 3: Live trading (when ready)

**Mode A (alert + confirm):**
Agent monitors via broker WebSocket → detects setup matching proven strategies → sends Telegram alert with entry, stop, target, R:R, conviction score (from paper data) → user replies YES → agent calls broker API → order placed.

**Mode B (full auto, later):**
Pre-approved strategies execute during market hours. End-of-day Telegram summary. User can pause anytime.

**Risk management enforced:**
- Max 1-2% account risk per trade
- Max 3-5% daily loss → auto-pause
- Stop loss mandatory on every order
- Revenge trade detection → warning with own data
- Position sizing from stop distance, not arbitrary quantity

**Broker costs:**
| Broker | API | Brokerage | Data | Best for |
|---|---|---|---|---|
| Shoonya | Free | ₹0 | Free | Smallest capital (₹1K-5K) |
| Angel One | Free | ₹20/order | Free | Best SDK/docs |
| Zerodha Personal | Free | ₹0 delivery | No data (₹500/mo) | Already have account |
| Fyers | Free | ₹20/order | Free | Fast execution |

**SEBI note:** Static IP required for API trading since April 2026. Need VPS (₹300-500/mo) for phase 3.

---

## 17 primitives the agent exposes

1. **remember** — store fact with scope + entity extraction
2. **recall** — FTS5, ChromaDB similarity, or Neo4j graph traversal
3. **forget** — pin/unpin + automatic Ebbinghaus decay
4. **promote** — session→user or session→org based on rules
5. **contradict** — detect conflicts, archive old, maintain supersede chain
6. **learn** — generate skill from completed task
7. **improve** — version skill, A/B test, auto-promote winner
8. **pre-check** — dry-run skill chain for dependency validation
9. **reflect** — self-critique, classify failure, patch skill
10. **alert** — regression alarm on quality/cost/retrieval degradation
11. **consolidate** — boost retention of semantically related neighbors
12. **schedule** — cron jobs (price monitoring, retention checks, daily digest)
13. **browse** — Playwright scraping for charts and news
14. **delegate** — Docker sub-agents for parallel tasks
15. **connect** — Telegram, Discord, CLI, REST API
16. **share** — org-scoped memory for team features
17. **query graph** — multi-hop traversal with temporal filters

---

## Build order (sprints)

### Sprint 1 (week 1-2): Core agent + Telegram + concept graph
- Project scaffolding: `karna/` package structure
- Telegram bot via python-telegram-bot
- Ollama integration for LLM calls
- SQLite FTS5 for conversation storage
- Neo4j setup + seed concept graph (hardcode tree above)
- Basic flow: message → LLM → respond → store in memory
- Basic teaching: ask question → check prerequisites in graph → explain → log comprehension

**Deliverable:** can ask trading questions on Telegram, agent teaches with prerequisite awareness.

### Sprint 2 (week 2-3): Forgetting engine + ChromaDB + quizzes
- ChromaDB for semantic embeddings alongside FTS5
- Ebbinghaus decay function on retention scores
- Cron job (APScheduler) for daily decay checks
- Quiz generation skill v1
- Telegram proactive messages for review prompts
- Semantic consolidation: on access, bump 5 nearest neighbors by 15%

**Deliverable:** agent sends quiz reminders when concepts decay. Memory has text + semantic search.

### Sprint 3 (week 3-4): Market data integration
- jugaad_data + nsetools integration for Indian market data
- yfinance as fallback
- pandas-ta for indicator calculations
- "Show me RSI on RELIANCE" → fetch data → calculate → respond with value + interpretation
- Real chart data in explanations: "here's a recent example of a hammer candle on TCS"
- Data abstraction layer: `karna/data/` with provider interface

**Deliverable:** agent uses live/historical market data in teaching. Can calculate and explain any indicator.

### Sprint 4 (week 4-5): Paper trading + broker abstraction
- PaperBroker class implementing Broker interface
- Entity extraction from trade messages ("paper buy RELIANCE at 2850")
- Trade storage in Neo4j: trade → strategy → concepts → outcome
- Cron price monitoring during market hours (every 15-30 min)
- Target/stop loss alerts via Telegram
- Outcome logging when trade closes

**Deliverable:** can paper trade via Telegram with monitoring and outcome tracking.

### Sprint 5 (week 5-7): Analytics + contradiction detection
- Win rate calculations by strategy type (graph queries)
- Behavioral pattern analysis: time of day, revenge trading, hold duration
- Contradiction detector: embed → retrieve similar → LLM check → archive/supersede
- Memory promotion policy implementation
- Strategy performance dashboard (Plotly, served via FastAPI)

**Deliverable:** can ask "which of my setups works best?" and get data-backed answers. Dashboard live.

### Sprint 6 (week 7-8): Skill versioning + A/B eval + regression alarms
- Skill file format (YAML) with version history
- A/B evaluation: run both versions, compare metrics, auto-promote
- Regression alarm: 7-day rolling averages on quality, cost, retrieval precision
- Skill pre-check for dependency chains

**Deliverable:** skills evolve and improve measurably. Quality degradation auto-detected.

### Sprint 7 (week 9-10): Live broker integration
- ShoonyaBroker and/or AngelOneBroker implementing Broker interface
- Mode A: alert → confirm → execute via Telegram
- Risk management enforcement (position sizing, max risk, stop loss, revenge detection)
- VPS setup with static IP (SEBI compliance)

**Deliverable:** can execute real trades via Telegram with risk guardrails.

### Sprint 8 (week 10+): Automation + polish
- Mode B: autonomous execution during market hours
- End-of-day summary via Telegram
- Live trade data feeding back into knowledge graph (replaces paper data over time)
- README, demo video, portfolio documentation

**Deliverable:** fully autonomous trading agent with months of accumulated knowledge.

---

## Project structure

```
karna/
├── main.py                    # Entry point
├── config.py                  # Environment, API keys, broker selection
├── agent/
│   ├── router.py              # Intent classification
│   ├── core.py                # Main agent loop (LangGraph state machine)
│   ├── reflector.py           # Self-critique + failure analysis
│   └── evaluator.py           # Quality scoring + regression alarms
├── memory/
│   ├── session_store.py       # SQLite FTS5
│   ├── semantic_store.py      # ChromaDB
│   ├── graph_store.py         # Neo4j
│   ├── identity.py            # USER.md / AGENT.md management
│   ├── forgetting.py          # Ebbinghaus decay engine
│   ├── contradiction.py       # Contradiction detection
│   ├── promotion.py           # Scope promotion policy
│   └── consolidation.py       # Semantic neighbor boosting
├── skills/
│   ├── manager.py             # Skill CRUD, versioning, A/B eval
│   ├── precheck.py            # Dependency validation
│   └── library/               # Skill YAML files (versioned)
│       ├── teach_concept.yaml
│       ├── generate_quiz.yaml
│       ├── evaluate_trade.yaml
│       ├── build_strategy.yaml
│       └── morning_proposal.yaml
├── trading/
│   ├── concepts.py            # Trading concept graph seed data
│   ├── indicators.py          # Technical indicator calculations (pandas-ta wrapper)
│   ├── paper_tracker.py       # Paper trade logging + outcome tracking
│   ├── analytics.py           # Win rate, behavioral patterns, contradiction checks
│   └── risk.py                # Position sizing, max risk, revenge detection
├── brokers/
│   ├── base.py                # Abstract Broker interface
│   ├── paper.py               # PaperBroker (SQLite-backed simulation)
│   ├── shoonya.py             # Finvasia Shoonya implementation
│   ├── angelone.py            # Angel One SmartAPI implementation
│   └── zerodha.py             # Zerodha Kite Personal implementation
├── data/
│   ├── base.py                # Abstract DataProvider interface
│   ├── jugaad.py              # jugaad_data implementation
│   ├── nsetools_provider.py   # nsetools for live quotes
│   ├── yfinance_provider.py   # yfinance fallback
│   └── cache.py               # Local cache for rate limit protection
├── connectors/
│   ├── telegram_bot.py        # Telegram interface
│   ├── cli.py                 # CLI interface
│   └── api.py                 # FastAPI REST interface
├── scheduler/
│   ├── cron.py                # APScheduler cron jobs
│   ├── market_monitor.py      # Price monitoring during market hours
│   └── retention_checker.py   # Daily decay + quiz scheduling
├── dashboard/
│   ├── app.py                 # FastAPI + Plotly dashboard
│   ├── skill_charts.py        # Skill performance visualizations
│   ├── memory_charts.py       # Memory growth/decay curves
│   └── trade_charts.py        # Trading analytics visualizations
├── llm/
│   ├── ollama.py              # Local model wrapper
│   ├── openai_fallback.py     # GPT-4o mini for complex tasks
│   └── prompts.py             # System prompts for different agent modes
├── data_files/
│   ├── USER.md                # Agent's knowledge about the user
│   └── AGENT.md               # Agent's operational self-knowledge
├── tests/
│   ├── test_memory.py
│   ├── test_forgetting.py
│   ├── test_contradiction.py
│   ├── test_skills.py
│   ├── test_paper_broker.py
│   └── test_analytics.py
├── docker-compose.yml         # Neo4j + ChromaDB + Ollama
├── requirements.txt
└── README.md
```

---

## How to position on a resume

> Built a persistent AI agent with 4-layer memory (FTS5, vector store, knowledge graph, identity files), Ebbinghaus-inspired forgetting with semantic consolidation, contradiction detection, versioned self-improving skills with A/B evaluation, root-cause failure analysis, regression alarms, and broker-abstracted trading execution — deployed on Telegram using LangGraph, Neo4j, ChromaDB, Ollama, and FastAPI. The agent taught me trading from zero, tracked 100+ paper trades with quantified strategy analytics, and evolved its own signal generation skill from v1 (price action) to v4 (multi-indicator + sentiment) through automated A/B testing.

---

## Research references

- Hermes Agent (Nous Research, 2026) — 5-pillar architecture, 175K+ stars
- OpenClaw (2025-26) — 345K+ stars, 13K+ community skills
- FadeMem (2026) — 45% storage reduction with Ebbinghaus decay
- FSFM (2026) — staircase reinforcement for high-value memories
- Mem0 (2026) — 4-scope memory model
- AgeMem (2026) — RL-optimized memory operations
- WebCoach (2026) — 28% step reduction, 38B matching GPT-4o
- CASCADE (2026) — 93.3% multi-hop success with skill composition
- CORAL (2026) — 3-10x improvement with shared persistent memory
- Kimpton AI — trade proposal platform (inspiration for phase 3 morning alerts)
