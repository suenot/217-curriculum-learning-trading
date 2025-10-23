use anyhow::Result;
use ndarray::Array1;
use rand::Rng;
use serde::{Deserialize, Serialize};

// ============================================================
// Market Data Structures
// ============================================================

/// A single OHLCV candle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// A market period consisting of multiple candles
#[derive(Debug, Clone)]
pub struct MarketPeriod {
    pub id: usize,
    pub candles: Vec<Candle>,
    pub difficulty_score: f64,
}

/// Difficulty level buckets
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DifficultyLevel {
    Easy,
    Medium,
    Hard,
}

impl std::fmt::Display for DifficultyLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DifficultyLevel::Easy => write!(f, "Easy"),
            DifficultyLevel::Medium => write!(f, "Medium"),
            DifficultyLevel::Hard => write!(f, "Hard"),
        }
    }
}

// ============================================================
// Difficulty Scorer
// ============================================================

/// Scores market periods by difficulty using volatility and other metrics.
pub struct DifficultyScorer {
    pub volatility_weight: f64,
    pub drawdown_weight: f64,
    pub gap_weight: f64,
}

impl DifficultyScorer {
    pub fn new(volatility_weight: f64, drawdown_weight: f64, gap_weight: f64) -> Self {
        Self {
            volatility_weight,
            drawdown_weight,
            gap_weight,
        }
    }

    /// Default scorer using only volatility.
    pub fn volatility_only() -> Self {
        Self::new(1.0, 0.0, 0.0)
    }

    /// Compute log returns from candle close prices.
    pub fn compute_log_returns(candles: &[Candle]) -> Vec<f64> {
        candles
            .windows(2)
            .map(|w| (w[1].close / w[0].close).ln())
            .collect()
    }

    /// Compute realized volatility of a set of returns.
    pub fn compute_volatility(returns: &[f64]) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }
        let n = returns.len() as f64;
        let mean = returns.iter().sum::<f64>() / n;
        let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
        variance.sqrt()
    }

    /// Compute maximum drawdown from candle close prices.
    pub fn compute_max_drawdown(candles: &[Candle]) -> f64 {
        if candles.is_empty() {
            return 0.0;
        }
        let mut peak = candles[0].close;
        let mut max_dd = 0.0_f64;
        for c in candles {
            if c.close > peak {
                peak = c.close;
            }
            let dd = (peak - c.close) / peak;
            max_dd = max_dd.max(dd);
        }
        max_dd
    }

    /// Compute gap frequency (fraction of candles with open != previous close).
    pub fn compute_gap_frequency(candles: &[Candle]) -> f64 {
        if candles.len() < 2 {
            return 0.0;
        }
        let gaps = candles
            .windows(2)
            .filter(|w| {
                let gap = (w[1].open - w[0].close).abs() / w[0].close;
                gap > 0.001 // gap threshold of 0.1%
            })
            .count();
        gaps as f64 / (candles.len() - 1) as f64
    }

    /// Score a market period for difficulty.
    pub fn score(&self, candles: &[Candle]) -> f64 {
        let returns = Self::compute_log_returns(candles);
        let vol = Self::compute_volatility(&returns);
        let dd = Self::compute_max_drawdown(candles);
        let gap = Self::compute_gap_frequency(candles);

        self.volatility_weight * vol + self.drawdown_weight * dd + self.gap_weight * gap
    }

    /// Score and assign difficulty to multiple periods.
    pub fn score_periods(&self, periods: &mut [MarketPeriod]) {
        for period in periods.iter_mut() {
            period.difficulty_score = self.score(&period.candles);
        }
    }
}

// ============================================================
// Curriculum Scheduler
// ============================================================

/// Manages progression through easy/medium/hard difficulty buckets.
pub struct CurriculumScheduler {
    pub easy_threshold: f64,
    pub hard_threshold: f64,
    pub current_level: DifficultyLevel,
    pub epochs_per_level: usize,
    pub current_epoch: usize,
}

impl CurriculumScheduler {
    /// Create a scheduler from scored periods using quantile thresholds.
    pub fn from_periods(periods: &[MarketPeriod], epochs_per_level: usize) -> Self {
        let mut scores: Vec<f64> = periods.iter().map(|p| p.difficulty_score).collect();
        scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = scores.len();
        let easy_threshold = if n > 0 { scores[n / 3] } else { 0.0 };
        let hard_threshold = if n > 0 { scores[2 * n / 3] } else { 0.0 };

        Self {
            easy_threshold,
            hard_threshold,
            current_level: DifficultyLevel::Easy,
            epochs_per_level,
            current_epoch: 0,
        }
    }

    /// Create with explicit thresholds.
    pub fn new(easy_threshold: f64, hard_threshold: f64, epochs_per_level: usize) -> Self {
        Self {
            easy_threshold,
            hard_threshold,
            current_level: DifficultyLevel::Easy,
            epochs_per_level,
            current_epoch: 0,
        }
    }

    /// Classify a difficulty score into a level.
    pub fn classify(&self, score: f64) -> DifficultyLevel {
        if score < self.easy_threshold {
            DifficultyLevel::Easy
        } else if score < self.hard_threshold {
            DifficultyLevel::Medium
        } else {
            DifficultyLevel::Hard
        }
    }

    /// Get periods available at the current curriculum level.
    pub fn get_available_periods<'a>(&self, periods: &'a [MarketPeriod]) -> Vec<&'a MarketPeriod> {
        periods
            .iter()
            .filter(|p| self.classify(p.difficulty_score) <= self.current_level)
            .collect()
    }

    /// Advance the curriculum by one epoch. Returns true if level changed.
    pub fn advance_epoch(&mut self) -> bool {
        self.current_epoch += 1;
        if self.current_epoch >= self.epochs_per_level {
            self.current_epoch = 0;
            match self.current_level {
                DifficultyLevel::Easy => {
                    self.current_level = DifficultyLevel::Medium;
                    true
                }
                DifficultyLevel::Medium => {
                    self.current_level = DifficultyLevel::Hard;
                    true
                }
                DifficultyLevel::Hard => false,
            }
        } else {
            false
        }
    }

    /// Reset scheduler to beginning.
    pub fn reset(&mut self) {
        self.current_level = DifficultyLevel::Easy;
        self.current_epoch = 0;
    }
}

// ============================================================
// Self-Paced Learner
// ============================================================

/// Adjusts curriculum based on agent performance (competence-based).
pub struct SelfPacedLearner {
    pub competence: f64,
    pub pace_parameter: f64,
    pub pace_increment: f64,
    pub min_sharpe_to_advance: f64,
    pub total_epochs: usize,
    pub current_epoch: usize,
}

impl SelfPacedLearner {
    pub fn new(total_epochs: usize, min_sharpe_to_advance: f64, pace_increment: f64) -> Self {
        Self {
            competence: 0.0,
            pace_parameter: 0.1,
            pace_increment,
            min_sharpe_to_advance,
            total_epochs,
            current_epoch: 0,
        }
    }

    /// Update competence based on current epoch and PnL.
    pub fn update_competence(&mut self, pnl: f64, pnl_target: f64) {
        self.current_epoch += 1;
        let time_factor = self.current_epoch as f64 / self.total_epochs as f64;
        let perf_factor = 1.0 + (pnl / pnl_target).min(1.0);
        self.competence = (time_factor * perf_factor).sqrt().min(1.0);
    }

    /// Update pace parameter based on Sharpe ratio performance.
    pub fn update_pace(&mut self, sharpe: f64) {
        if sharpe > self.min_sharpe_to_advance {
            self.pace_parameter = (self.pace_parameter + self.pace_increment).min(1.0);
        }
    }

    /// Get the maximum difficulty score the agent should face.
    pub fn max_difficulty(&self, max_score: f64) -> f64 {
        self.competence * max_score
    }

    /// Determine if a sample should be included based on its loss.
    pub fn should_include_sample(&self, loss: f64) -> bool {
        loss < self.pace_parameter
    }

    /// Filter periods based on current competence.
    pub fn filter_periods<'a>(
        &self,
        periods: &'a [MarketPeriod],
        max_score: f64,
    ) -> Vec<&'a MarketPeriod> {
        let threshold = self.max_difficulty(max_score);
        periods
            .iter()
            .filter(|p| p.difficulty_score <= threshold)
            .collect()
    }
}

// ============================================================
// Trading Agent
// ============================================================

/// A simple momentum-based trading agent.
#[derive(Debug, Clone)]
pub struct TradingAgent {
    pub momentum_window: usize,
    pub position_size: f64,
    pub learning_rate: f64,
    pub momentum_threshold: f64,
}

/// Result of trading on a period.
#[derive(Debug, Clone)]
pub struct TradingResult {
    pub pnl: f64,
    pub sharpe: f64,
    pub max_drawdown: f64,
    pub num_trades: usize,
    pub returns: Vec<f64>,
}

impl TradingAgent {
    pub fn new(momentum_window: usize, position_size: f64, learning_rate: f64) -> Self {
        Self {
            momentum_window,
            position_size,
            learning_rate,
            momentum_threshold: 0.0,
        }
    }

    /// Compute momentum signal from candles.
    fn compute_momentum(&self, candles: &[Candle], idx: usize) -> f64 {
        if idx < self.momentum_window {
            return 0.0;
        }
        let start = idx - self.momentum_window;
        let returns: f64 = (candles[idx].close / candles[start].close) - 1.0;
        returns
    }

    /// Trade on a market period and return results.
    pub fn trade(&self, candles: &[Candle]) -> TradingResult {
        if candles.len() < self.momentum_window + 2 {
            return TradingResult {
                pnl: 0.0,
                sharpe: 0.0,
                max_drawdown: 0.0,
                num_trades: 0,
                returns: vec![],
            };
        }

        let mut returns = Vec::new();
        let mut position = 0.0_f64;
        let mut num_trades = 0;

        for i in self.momentum_window..candles.len() - 1 {
            let momentum = self.compute_momentum(candles, i);
            let new_position = if momentum > self.momentum_threshold {
                self.position_size
            } else if momentum < -self.momentum_threshold {
                -self.position_size
            } else {
                0.0
            };

            if (new_position - position).abs() > 1e-10 {
                num_trades += 1;
            }
            position = new_position;

            let ret = position * (candles[i + 1].close / candles[i].close - 1.0);
            returns.push(ret);
        }

        let pnl: f64 = returns.iter().sum();
        let sharpe = Self::compute_sharpe(&returns);
        let max_drawdown = Self::compute_pnl_drawdown(&returns);

        TradingResult {
            pnl,
            sharpe,
            max_drawdown,
            num_trades,
            returns,
        }
    }

    /// Update agent parameters based on trading result (simple adaptation).
    pub fn learn(&mut self, result: &TradingResult) {
        // Adjust momentum threshold based on performance
        if result.sharpe > 0.0 {
            // Good performance: tighten threshold slightly
            self.momentum_threshold += self.learning_rate * 0.001;
        } else {
            // Poor performance: loosen threshold
            self.momentum_threshold = (self.momentum_threshold - self.learning_rate * 0.002).max(0.0);
        }
    }

    /// Train with curriculum: progressive difficulty.
    pub fn train_with_curriculum(
        &mut self,
        periods: &[MarketPeriod],
        scheduler: &mut CurriculumScheduler,
        num_epochs: usize,
    ) -> Vec<TradingResult> {
        let mut all_results = Vec::new();

        for _epoch in 0..num_epochs {
            let available = scheduler.get_available_periods(periods);
            if available.is_empty() {
                scheduler.advance_epoch();
                continue;
            }

            // Trade on each available period
            let mut epoch_results = Vec::new();
            for period in &available {
                let result = self.trade(&period.candles);
                self.learn(&result);
                epoch_results.push(result);
            }

            // Average epoch result
            if !epoch_results.is_empty() {
                let avg_pnl = epoch_results.iter().map(|r| r.pnl).sum::<f64>()
                    / epoch_results.len() as f64;
                let avg_sharpe = epoch_results.iter().map(|r| r.sharpe).sum::<f64>()
                    / epoch_results.len() as f64;
                all_results.push(TradingResult {
                    pnl: avg_pnl,
                    sharpe: avg_sharpe,
                    max_drawdown: epoch_results
                        .iter()
                        .map(|r| r.max_drawdown)
                        .fold(0.0_f64, f64::max),
                    num_trades: epoch_results.iter().map(|r| r.num_trades).sum(),
                    returns: vec![avg_pnl],
                });
            }

            scheduler.advance_epoch();
        }

        all_results
    }

    /// Train with random sampling (baseline comparison).
    pub fn train_random(
        &mut self,
        periods: &[MarketPeriod],
        num_epochs: usize,
    ) -> Vec<TradingResult> {
        let mut rng = rand::thread_rng();
        let mut all_results = Vec::new();

        for _epoch in 0..num_epochs {
            if periods.is_empty() {
                continue;
            }

            // Sample random periods
            let mut epoch_results = Vec::new();
            let sample_size = (periods.len() / 3).max(1);
            for _ in 0..sample_size {
                let idx = rng.gen_range(0..periods.len());
                let result = self.trade(&periods[idx].candles);
                self.learn(&result);
                epoch_results.push(result);
            }

            if !epoch_results.is_empty() {
                let avg_pnl = epoch_results.iter().map(|r| r.pnl).sum::<f64>()
                    / epoch_results.len() as f64;
                let avg_sharpe = epoch_results.iter().map(|r| r.sharpe).sum::<f64>()
                    / epoch_results.len() as f64;
                all_results.push(TradingResult {
                    pnl: avg_pnl,
                    sharpe: avg_sharpe,
                    max_drawdown: epoch_results
                        .iter()
                        .map(|r| r.max_drawdown)
                        .fold(0.0_f64, f64::max),
                    num_trades: epoch_results.iter().map(|r| r.num_trades).sum(),
                    returns: vec![avg_pnl],
                });
            }
        }

        all_results
    }

    fn compute_sharpe(returns: &[f64]) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }
        let n = returns.len() as f64;
        let mean = returns.iter().sum::<f64>() / n;
        let var = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
        let std = var.sqrt();
        if std < 1e-12 {
            return 0.0;
        }
        mean / std
    }

    fn compute_pnl_drawdown(returns: &[f64]) -> f64 {
        let mut cumulative = 0.0_f64;
        let mut peak = 0.0_f64;
        let mut max_dd = 0.0_f64;
        for r in returns {
            cumulative += r;
            if cumulative > peak {
                peak = cumulative;
            }
            let dd = peak - cumulative;
            if dd > max_dd {
                max_dd = dd;
            }
        }
        max_dd
    }
}

// ============================================================
// Utility: Split candles into periods
// ============================================================

/// Split a vector of candles into equal-sized periods.
pub fn split_into_periods(candles: Vec<Candle>, period_size: usize) -> Vec<MarketPeriod> {
    candles
        .chunks(period_size)
        .enumerate()
        .filter(|(_, chunk)| chunk.len() == period_size)
        .map(|(id, chunk)| MarketPeriod {
            id,
            candles: chunk.to_vec(),
            difficulty_score: 0.0,
        })
        .collect()
}

// ============================================================
// Bybit API Client
// ============================================================

/// Bybit API response structures.
#[derive(Debug, Deserialize)]
pub struct BybitResponse {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: BybitResult,
}

#[derive(Debug, Deserialize)]
pub struct BybitResult {
    pub list: Vec<Vec<String>>,
}

/// Client for fetching market data from Bybit.
pub struct BybitClient {
    pub base_url: String,
}

impl BybitClient {
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Fetch kline (OHLCV) data from Bybit.
    pub fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: u32,
    ) -> Result<Vec<Candle>> {
        let url = format!(
            "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

        let resp: BybitResponse = reqwest::blocking::get(&url)?.json()?;

        if resp.ret_code != 0 {
            anyhow::bail!("Bybit API error: {}", resp.ret_msg);
        }

        let mut candles: Vec<Candle> = resp
            .result
            .list
            .iter()
            .filter_map(|item| {
                if item.len() >= 6 {
                    Some(Candle {
                        timestamp: item[0].parse().ok()?,
                        open: item[1].parse().ok()?,
                        high: item[2].parse().ok()?,
                        low: item[3].parse().ok()?,
                        close: item[4].parse().ok()?,
                        volume: item[5].parse().ok()?,
                    })
                } else {
                    None
                }
            })
            .collect();

        // Bybit returns newest first, reverse to chronological order
        candles.reverse();
        Ok(candles)
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// Helper: Generate synthetic market data for testing
// ============================================================

/// Generate synthetic candle data with specified volatility.
pub fn generate_synthetic_candles(
    num_candles: usize,
    base_price: f64,
    volatility: f64,
    trend: f64,
    seed: u64,
) -> Vec<Candle> {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut price = base_price;
    let mut candles = Vec::with_capacity(num_candles);

    for i in 0..num_candles {
        let ret: f64 = trend + volatility * (rng.gen::<f64>() - 0.5) * 2.0;
        let open = price;
        price *= 1.0 + ret;
        let close = price;
        let high = open.max(close) * (1.0 + volatility * rng.gen::<f64>() * 0.5);
        let low = open.min(close) * (1.0 - volatility * rng.gen::<f64>() * 0.5);
        let volume = 1000.0 + 500.0 * rng.gen::<f64>();

        candles.push(Candle {
            timestamp: 1_000_000 + i as u64 * 3600,
            open,
            high,
            low,
            close,
            volume,
        });
    }

    candles
}

// ============================================================
// ndarray helper for returns analysis
// ============================================================

/// Convert returns to ndarray for analysis.
pub fn returns_to_array(returns: &[f64]) -> Array1<f64> {
    Array1::from_vec(returns.to_vec())
}

/// Compute cumulative returns from an ndarray of returns.
pub fn cumulative_returns(returns: &Array1<f64>) -> Array1<f64> {
    let mut cum = Array1::zeros(returns.len());
    let mut total = 0.0;
    for (i, &r) in returns.iter().enumerate() {
        total += r;
        cum[i] = total;
    }
    cum
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_candles(prices: &[f64]) -> Vec<Candle> {
        prices
            .iter()
            .enumerate()
            .map(|(i, &p)| Candle {
                timestamp: i as u64,
                open: p,
                high: p * 1.01,
                low: p * 0.99,
                close: p,
                volume: 100.0,
            })
            .collect()
    }

    #[test]
    fn test_compute_log_returns() {
        let candles = make_test_candles(&[100.0, 105.0, 110.0, 108.0]);
        let returns = DifficultyScorer::compute_log_returns(&candles);
        assert_eq!(returns.len(), 3);
        assert!((returns[0] - (105.0_f64 / 100.0).ln()).abs() < 1e-10);
    }

    #[test]
    fn test_volatility_computation() {
        let returns = vec![0.01, -0.02, 0.015, -0.005, 0.03];
        let vol = DifficultyScorer::compute_volatility(&returns);
        assert!(vol > 0.0);
        assert!(vol < 1.0);
    }

    #[test]
    fn test_max_drawdown() {
        let candles = make_test_candles(&[100.0, 110.0, 105.0, 95.0, 100.0]);
        let dd = DifficultyScorer::compute_max_drawdown(&candles);
        // Peak at 110, trough at 95 => dd = 15/110 ~= 0.1364
        assert!((dd - 15.0 / 110.0).abs() < 1e-4);
    }

    #[test]
    fn test_difficulty_scorer_ordering() {
        let scorer = DifficultyScorer::volatility_only();

        let calm = generate_synthetic_candles(50, 100.0, 0.005, 0.001, 42);
        let volatile = generate_synthetic_candles(50, 100.0, 0.05, 0.0, 42);

        let score_calm = scorer.score(&calm);
        let score_volatile = scorer.score(&volatile);

        assert!(
            score_calm < score_volatile,
            "Calm market ({}) should have lower difficulty than volatile market ({})",
            score_calm,
            score_volatile
        );
    }

    #[test]
    fn test_curriculum_scheduler_progression() {
        let mut scheduler = CurriculumScheduler::new(0.01, 0.03, 2);

        assert_eq!(scheduler.current_level, DifficultyLevel::Easy);

        // Advance 2 epochs to move to Medium
        scheduler.advance_epoch();
        assert_eq!(scheduler.current_level, DifficultyLevel::Easy);
        let changed = scheduler.advance_epoch();
        assert!(changed);
        assert_eq!(scheduler.current_level, DifficultyLevel::Medium);

        // Advance 2 more to move to Hard
        scheduler.advance_epoch();
        let changed = scheduler.advance_epoch();
        assert!(changed);
        assert_eq!(scheduler.current_level, DifficultyLevel::Hard);

        // Should not advance beyond Hard
        let changed = scheduler.advance_epoch();
        assert!(!changed);
        assert_eq!(scheduler.current_level, DifficultyLevel::Hard);
    }

    #[test]
    fn test_curriculum_scheduler_filtering() {
        let scheduler = CurriculumScheduler::new(0.01, 0.03, 2);

        let periods = vec![
            MarketPeriod {
                id: 0,
                candles: vec![],
                difficulty_score: 0.005,
            },
            MarketPeriod {
                id: 1,
                candles: vec![],
                difficulty_score: 0.02,
            },
            MarketPeriod {
                id: 2,
                candles: vec![],
                difficulty_score: 0.05,
            },
        ];

        // At Easy level, only easy period available
        let available = scheduler.get_available_periods(&periods);
        assert_eq!(available.len(), 1);
        assert_eq!(available[0].id, 0);
    }

    #[test]
    fn test_self_paced_learner_competence() {
        let mut spl = SelfPacedLearner::new(100, 0.5, 0.05);
        assert_eq!(spl.competence, 0.0);

        spl.update_competence(50.0, 100.0);
        assert!(spl.competence > 0.0);
        assert!(spl.competence <= 1.0);

        let comp_after_1 = spl.competence;

        spl.update_competence(80.0, 100.0);
        assert!(spl.competence >= comp_after_1);
    }

    #[test]
    fn test_self_paced_learner_pace() {
        let mut spl = SelfPacedLearner::new(100, 0.5, 0.1);
        let initial_pace = spl.pace_parameter;

        // Sharpe below threshold: no change
        spl.update_pace(0.3);
        assert_eq!(spl.pace_parameter, initial_pace);

        // Sharpe above threshold: advance
        spl.update_pace(0.8);
        assert!(spl.pace_parameter > initial_pace);
    }

    #[test]
    fn test_trading_agent_basic() {
        let agent = TradingAgent::new(5, 1.0, 0.01);
        let candles = generate_synthetic_candles(50, 100.0, 0.01, 0.002, 42);
        let result = agent.trade(&candles);

        assert!(!result.returns.is_empty());
        assert!(result.num_trades > 0);
    }

    #[test]
    fn test_split_into_periods() {
        let candles = generate_synthetic_candles(100, 100.0, 0.01, 0.0, 42);
        let periods = split_into_periods(candles, 20);
        assert_eq!(periods.len(), 5);
        assert_eq!(periods[0].candles.len(), 20);
    }

    #[test]
    fn test_ndarray_helpers() {
        let returns = vec![0.01, -0.02, 0.03, 0.005, -0.01];
        let arr = returns_to_array(&returns);
        assert_eq!(arr.len(), 5);

        let cum = cumulative_returns(&arr);
        assert_eq!(cum.len(), 5);
        assert!((cum[0] - 0.01).abs() < 1e-10);
        assert!((cum[1] - (-0.01)).abs() < 1e-10);
    }

    #[test]
    fn test_gap_frequency() {
        // Build candles where open[i+1] == close[i] (no gaps)
        let mut candles = vec![];
        for i in 0..4 {
            let p = 100.0 + i as f64;
            candles.push(Candle {
                timestamp: i as u64,
                open: p,
                high: p + 0.5,
                low: p - 0.5,
                close: p + 1.0, // next candle's open will match this
                volume: 100.0,
            });
        }
        // Ensure continuity: open[i+1] = close[i]
        for i in 1..candles.len() {
            candles[i].open = candles[i - 1].close;
        }
        let gap = DifficultyScorer::compute_gap_frequency(&candles);
        assert!(gap < 0.01, "Expected no gaps, got {}", gap);

        // Add a large gap
        candles[2].open = 110.0;
        let gap = DifficultyScorer::compute_gap_frequency(&candles);
        assert!(gap > 0.0, "Expected gaps after modification, got {}", gap);
    }
}
