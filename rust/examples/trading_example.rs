use curriculum_learning_trading::*;

fn main() {
    println!("=== Curriculum Learning for Trading ===\n");

    // ---------------------------------------------------------
    // Step 1: Fetch data from Bybit (or use synthetic fallback)
    // ---------------------------------------------------------
    println!("Step 1: Fetching BTCUSDT data from Bybit...");
    let candles = match BybitClient::new().fetch_klines("BTCUSDT", "60", 200) {
        Ok(c) => {
            println!("  Fetched {} candles from Bybit", c.len());
            c
        }
        Err(e) => {
            println!("  Bybit fetch failed ({}), using synthetic data", e);
            let mut all = Vec::new();
            // Easy: low volatility, uptrend
            all.extend(generate_synthetic_candles(80, 40000.0, 0.005, 0.001, 1));
            // Medium: moderate volatility
            all.extend(generate_synthetic_candles(80, 42000.0, 0.02, 0.0, 2));
            // Hard: high volatility, downtrend
            all.extend(generate_synthetic_candles(80, 41000.0, 0.06, -0.002, 3));
            all
        }
    };

    // ---------------------------------------------------------
    // Step 2: Split into periods and score difficulty
    // ---------------------------------------------------------
    println!("\nStep 2: Scoring historical periods by difficulty...");
    let period_size = 20;
    let mut periods = split_into_periods(candles, period_size);

    let scorer = DifficultyScorer::new(0.7, 0.2, 0.1);
    scorer.score_periods(&mut periods);

    // Sort by difficulty for display
    periods.sort_by(|a, b| {
        a.difficulty_score
            .partial_cmp(&b.difficulty_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    println!("  {} periods scored:", periods.len());
    for p in &periods {
        let level = if p.difficulty_score < 0.01 {
            "Easy"
        } else if p.difficulty_score < 0.03 {
            "Medium"
        } else {
            "Hard"
        };
        println!(
            "    Period {:2}: difficulty = {:.6} [{}]",
            p.id, p.difficulty_score, level
        );
    }

    // ---------------------------------------------------------
    // Step 3: Train agent WITH curriculum
    // ---------------------------------------------------------
    println!("\nStep 3: Training agent WITH curriculum (easy -> medium -> hard)...");
    let mut curriculum_agent = TradingAgent::new(5, 1.0, 0.01);
    let mut scheduler = CurriculumScheduler::from_periods(&periods, 3);

    let curriculum_results = curriculum_agent.train_with_curriculum(&periods, &mut scheduler, 9);

    let curriculum_total_pnl: f64 = curriculum_results.iter().map(|r| r.pnl).sum();
    let curriculum_avg_sharpe: f64 = if !curriculum_results.is_empty() {
        curriculum_results.iter().map(|r| r.sharpe).sum::<f64>() / curriculum_results.len() as f64
    } else {
        0.0
    };

    println!("  Curriculum training results:");
    for (i, r) in curriculum_results.iter().enumerate() {
        println!(
            "    Epoch {:2}: PnL = {:+.6}, Sharpe = {:+.4}, MaxDD = {:.6}, Trades = {}",
            i + 1,
            r.pnl,
            r.sharpe,
            r.max_drawdown,
            r.num_trades
        );
    }
    println!("  Total PnL: {:+.6}", curriculum_total_pnl);
    println!("  Avg Sharpe: {:+.4}", curriculum_avg_sharpe);

    // ---------------------------------------------------------
    // Step 4: Train agent WITHOUT curriculum (random baseline)
    // ---------------------------------------------------------
    println!("\nStep 4: Training agent WITHOUT curriculum (random sampling)...");
    let mut random_agent = TradingAgent::new(5, 1.0, 0.01);

    let random_results = random_agent.train_random(&periods, 9);

    let random_total_pnl: f64 = random_results.iter().map(|r| r.pnl).sum();
    let random_avg_sharpe: f64 = if !random_results.is_empty() {
        random_results.iter().map(|r| r.sharpe).sum::<f64>() / random_results.len() as f64
    } else {
        0.0
    };

    println!("  Random training results:");
    for (i, r) in random_results.iter().enumerate() {
        println!(
            "    Epoch {:2}: PnL = {:+.6}, Sharpe = {:+.4}, MaxDD = {:.6}, Trades = {}",
            i + 1,
            r.pnl,
            r.sharpe,
            r.max_drawdown,
            r.num_trades
        );
    }
    println!("  Total PnL: {:+.6}", random_total_pnl);
    println!("  Avg Sharpe: {:+.4}", random_avg_sharpe);

    // ---------------------------------------------------------
    // Step 5: Compare results
    // ---------------------------------------------------------
    println!("\n=== Performance Comparison ===");
    println!(
        "  Curriculum: PnL = {:+.6}, Sharpe = {:+.4}",
        curriculum_total_pnl, curriculum_avg_sharpe
    );
    println!(
        "  Random:     PnL = {:+.6}, Sharpe = {:+.4}",
        random_total_pnl, random_avg_sharpe
    );
    println!(
        "  PnL difference:    {:+.6}",
        curriculum_total_pnl - random_total_pnl
    );
    println!(
        "  Sharpe difference: {:+.4}",
        curriculum_avg_sharpe - random_avg_sharpe
    );

    // ---------------------------------------------------------
    // Step 6: Demonstrate self-paced learning
    // ---------------------------------------------------------
    println!("\n=== Self-Paced Learning Demo ===");
    let mut spl = SelfPacedLearner::new(9, 0.3, 0.1);
    let max_score = periods
        .iter()
        .map(|p| p.difficulty_score)
        .fold(0.0_f64, f64::max);

    let mut spl_agent = TradingAgent::new(5, 1.0, 0.01);

    for epoch in 0..9 {
        let available = spl.filter_periods(&periods, max_score);
        println!(
            "  Epoch {}: competence = {:.3}, pace = {:.3}, available periods = {}",
            epoch + 1,
            spl.competence,
            spl.pace_parameter,
            available.len()
        );

        let mut epoch_pnl = 0.0;
        let mut epoch_sharpe = 0.0;
        let mut count = 0;
        for p in &available {
            let result = spl_agent.trade(&p.candles);
            spl_agent.learn(&result);
            epoch_pnl += result.pnl;
            epoch_sharpe += result.sharpe;
            count += 1;
        }

        if count > 0 {
            epoch_pnl /= count as f64;
            epoch_sharpe /= count as f64;
        }

        spl.update_competence(epoch_pnl * 100.0, 1.0);
        spl.update_pace(epoch_sharpe);
    }

    println!("\nDone! Curriculum learning helps agents build skills progressively.");
}
