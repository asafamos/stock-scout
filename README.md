# Stock Scout

AI-driven stock analysis system combining technical & fundamental scoring with **advanced multi-layer filters**.

## Features

### Core Analysis
- **Technical Scoring**: MA alignment, Momentum (1/3/6m), RSI bands, Near-High detection, Overextension, Pullback windows, ATR/Price, MACD/ADX
- **Fundamental Analysis**: Growth metrics (Rev/EPS YoY), Quality indicators (ROE/Margin), Valuation ratios (P/E, P/S), Debt penalties

### 🆕 Advanced Filters (NEW!)
- **Relative Strength Analysis**: Compare stock performance vs market (SPY/QQQ) across multiple timeframes
- **Volume Surge Detection**: Identify institutional buying with abnormal volume spikes and price-volume correlation
- **Price Consolidation**: Detect volatility squeeze patterns (tight ranges before breakouts)
- **MA Alignment Check**: Verify bullish trend with proper moving average order (10>20>50>200)
- **Support/Resistance Levels**: Calculate distance to key levels for optimal entry timing
- **Momentum Quality**: Assess consistency and acceleration of price momentum
- **Risk/Reward Optimization**: Enhanced risk/reward ratios based on support/resistance and ATR
- **High Confidence Signals**: Multi-factor confirmation for highest probability setups

### Risk Management
- Earnings blackout window
- Beta filtering vs benchmark
- Sector diversification caps
- Price verification across multiple data providers (Alpha Vantage, Finnhub, Polygon, Tiingo)
- ATR-based position sizing

## Run
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r [requirements.txt](http://_vscodecontentref_/4)
streamlit run [stock_scout.py](http://_vscodecontentref_/5)
```
