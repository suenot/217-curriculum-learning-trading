# Chapter 289: Curriculum Learning for Trading

## Introduction

Curriculum learning is a training strategy inspired by human education: we learn arithmetic before calculus, simple melodies before symphonies. In the context of machine learning for trading, curriculum learning means presenting a model with progressively harder market conditions during training rather than exposing it to the full complexity of financial markets from the start.

Financial markets exhibit a wide spectrum of regimes -- from calm, low-volatility trending periods to chaotic, high-volatility crisis episodes. A naive approach trains models on randomly shuffled historical data, forcing the learner to simultaneously cope with gentle trends and violent regime changes. Curriculum learning instead organizes training data by difficulty: the agent first masters low-volatility trending markets, then graduates to moderate volatility and range-bound conditions, and finally tackles extreme events such as flash crashes and liquidity crises.

Bengio et al. (2009) formalized curriculum learning, showing that networks trained on easy-to-hard sequences converge faster and often reach better optima. In trading, the benefits are threefold: (1) faster convergence because the agent builds foundational strategies on simple data, (2) improved generalization because the agent gradually encounters harder edge cases, and (3) better risk management because the agent learns stable behavior before encountering tail events.

This chapter develops the mathematical framework for curriculum learning in trading, implements a complete Rust-based curriculum training pipeline, and demonstrates integration with live Bybit market data.

---

## Mathematical Framework

### 1. Task Difficulty Ordering

Let $\mathcal{D} = \{d_1, d_2, \ldots, d_N\}$ be a set of market periods (e.g., daily or weekly windows). We define a difficulty scoring function:

$$s: \mathcal{D} \rightarrow \mathbb{R}^+$$

A natural choice for trading is volatility-based difficulty:

$$s(d_i) = \sigma(d_i) = \sqrt{\frac{1}{T-1} \sum_{t=1}^{T} (r_t - \bar{r})^2}$$

where $r_t$ are the log-returns within period $d_i$ and $\bar{r}$ is the mean return.

We can enrich this with additional factors:

$$s(d_i) = w_1 \cdot \hat{\sigma}(d_i) + w_2 \cdot \text{DrawdownMax}(d_i) + w_3 \cdot \text{GapFreq}(d_i) + w_4 \cdot \text{SpreadMean}(d_i)$$

where each component captures a different dimension of market difficulty: volatility, drawdown severity, price gap frequency, and bid-ask spread.

### 2. Curriculum Scheduler

Given the scored periods, we sort them by difficulty and partition into $K$ difficulty buckets:

$$\mathcal{B}_k = \{d_i \mid q_{k-1} \leq s(d_i) < q_k\}, \quad k = 1, \ldots, K$$

where $q_k$ are quantile thresholds. A standard curriculum with $K = 3$ yields:

- **Easy** ($\mathcal{B}_1$): Low-volatility trending periods
- **Medium** ($\mathcal{B}_2$): Moderate volatility, mixed trends
- **Hard** ($\mathcal{B}_3$): High volatility, crises, whipsaws

The curriculum scheduler defines a training sequence $\mathcal{C} = (\mathcal{B}_1^{(e_1)}, \mathcal{B}_2^{(e_2)}, \mathcal{B}_3^{(e_3)})$ where $e_k$ denotes the number of epochs spent on bucket $k$.

A linear curriculum schedules the transition at epoch $t$:

$$p_k(t) = \begin{cases}
1 & \text{if } t < t_1 \text{ and } k = 1 \\
1 & \text{if } t_1 \leq t < t_2 \text{ and } k \leq 2 \\
1 & \text{if } t \geq t_2
\end{cases}$$

where $t_1, t_2$ are transition epochs and $p_k(t)$ indicates which buckets are available at time $t$.

### 3. Self-Paced Learning

Self-paced learning (SPL) extends curriculum learning by letting the model itself decide when to advance. We introduce a competence function:

$$c(t) = \min\left(1, \sqrt{\frac{t}{T} \cdot \left(1 + \frac{\text{PnL}(t)}{\text{PnL}_{\text{target}}}\right)}\right)$$

where $\text{PnL}(t)$ is the agent's cumulative profit at epoch $t$ and $\text{PnL}_{\text{target}}$ is a performance threshold. The competence score determines the maximum difficulty level the agent can access:

$$\mathcal{D}_{\text{available}}(t) = \{d_i \in \mathcal{D} \mid s(d_i) \leq Q(c(t))\}$$

where $Q(c)$ is the quantile function of the difficulty distribution at competence level $c$.

The self-paced weight for each sample is:

$$v_i^* = \begin{cases} 1 & \text{if } \ell_i < \lambda(t) \\ 0 & \text{otherwise} \end{cases}$$

where $\ell_i$ is the loss on sample $i$ and $\lambda(t)$ is a threshold that increases over training, progressively including harder samples.

### 4. Competence-Based Curriculum

The competence-based curriculum combines the scheduler with performance monitoring:

$$\lambda(t+1) = \lambda(t) + \alpha \cdot \mathbb{1}[\text{Sharpe}(t) > \text{Sharpe}_{\min}]$$

The pace parameter $\lambda$ only increases when the agent demonstrates sufficient performance (measured by Sharpe ratio) on the current difficulty level. This prevents premature advancement and ensures mastery at each stage.

---

## Applications in Trading

### Progressive Market Difficulty

A typical curriculum for a trading agent proceeds through three stages:

**Stage 1 -- Low Volatility Trending Markets:**
The agent learns basic trend-following on calm, directional markets. Annualized volatility below the 33rd percentile. The agent builds core position-sizing and entry/exit logic.

**Stage 2 -- Moderate Volatility Mixed Markets:**
The agent encounters range-bound and moderately volatile conditions. It learns to adapt between trending and mean-reverting regimes, handle false breakouts, and manage larger drawdowns.

**Stage 3 -- High Volatility Crisis Periods:**
The agent faces flash crashes, liquidity gaps, and extreme tail events. It must learn defensive strategies, rapid de-leveraging, and survival-first behavior.

### Multi-Asset Curriculum

Curriculum learning can also be applied across assets. Start training on the most liquid, well-behaved assets (e.g., BTC/USDT on major exchanges) and progressively introduce less liquid altcoins with wider spreads and more erratic behavior.

### Transfer Across Timeframes

Another dimension of difficulty is timeframe. Daily bars are smoother and easier to learn from; minute bars contain more noise. A curriculum can train first on daily data, then 4-hour, then hourly, and finally minute-level data.

---

## Rust Implementation

The implementation in `rust/src/lib.rs` provides:

1. **`DifficultyScorer`** -- Scores market periods by volatility and other metrics
2. **`CurriculumScheduler`** -- Manages progression through easy/medium/hard buckets
3. **`SelfPacedLearner`** -- Adjusts curriculum based on agent performance
4. **`TradingAgent`** -- Simple momentum-based agent trained with curriculum
5. **`BybitClient`** -- Fetches historical kline data from Bybit API

### Key Design Decisions

- **Volatility as primary difficulty metric**: Realized volatility is the most intuitive and robust measure of market difficulty for a trading agent.
- **Three-bucket curriculum**: Provides sufficient granularity without over-complicating the scheduler.
- **Competence gating**: The agent must achieve a minimum Sharpe ratio before advancing, preventing catastrophic exposure to hard markets.

---

## Bybit Data Integration

The implementation fetches OHLCV data from Bybit's public API (`/v5/market/kline`). Historical kline data is split into periods, each scored for difficulty. The curriculum scheduler then organizes these periods for progressive training.

```
GET https://api.bybit.com/v5/market/kline?category=linear&symbol=BTCUSDT&interval=60&limit=200
```

Each kline provides open, high, low, close, and volume -- sufficient to compute volatility, drawdowns, and other difficulty metrics.

---

## Key Takeaways

1. **Curriculum learning organizes training data by difficulty**, presenting easy market conditions first and progressively harder ones, mirroring how human traders learn.

2. **Volatility is a natural difficulty metric** for financial markets. Low-volatility trending periods are easy; high-volatility crisis periods are hard.

3. **Self-paced learning lets the agent control its own curriculum**, advancing only when it demonstrates competence on the current difficulty level.

4. **Competence gating prevents catastrophic failures** by ensuring the agent has mastered simpler conditions before facing extreme market events.

5. **Curriculum-trained agents consistently outperform randomly-trained agents** in terms of Sharpe ratio, maximum drawdown, and final PnL, particularly during out-of-sample crisis periods.

6. **The approach generalizes across assets and timeframes**, enabling multi-dimensional curricula that progressively increase complexity along multiple axes.

7. **Implementation in Rust** provides the performance needed for large-scale backtesting across many historical periods and curriculum configurations.

8. **Integration with Bybit** enables real-time difficulty scoring and curriculum construction on live crypto market data, bridging the gap between research and production trading systems.
