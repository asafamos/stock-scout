#!/usr/bin/env python3
"""Validate current stock recommendations - deep dive analysis.

Downloads real-time data and fundamentals for current picks to verify quality.
"""

import sys
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np

def analyze_stock(ticker: str, recommendation_type: str):
    """Deep analysis of a recommended stock."""
    print(f"\n{'='*70}")
    print(f"📊 ANALYZING: {ticker} ({recommendation_type})")
    print(f"{'='*70}")
    
    try:
        stock = yf.Ticker(ticker)
        
        # Get 6 months of data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        hist = stock.history(start=start_date, end=end_date)
        
        if hist.empty:
            print(f"❌ No data for {ticker}")
            return None
        
        # Current price and momentum
        current_price = hist['Close'].iloc[-1]
        price_5d_ago = hist['Close'].iloc[-6] if len(hist) >= 6 else current_price
        price_20d_ago = hist['Close'].iloc[-21] if len(hist) >= 21 else current_price
        price_60d_ago = hist['Close'].iloc[-61] if len(hist) >= 61 else current_price
        
        ret_5d = ((current_price / price_5d_ago) - 1) * 100
        ret_20d = ((current_price / price_20d_ago) - 1) * 100
        ret_60d = ((current_price / price_60d_ago) - 1) * 100
        
        # Volatility
        returns = hist['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100  # Annualized
        
        # Support/Resistance
        high_52w = hist['High'].tail(252).max()
        low_52w = hist['Low'].tail(252).min()
        price_range_pct = ((current_price - low_52w) / (high_52w - low_52w)) * 100
        
        # Volume analysis
        avg_volume = hist['Volume'].tail(20).mean()
        recent_volume = hist['Volume'].tail(5).mean()
        volume_surge = (recent_volume / avg_volume) if avg_volume > 0 else 1.0
        
        # Moving averages
        ma20 = hist['Close'].tail(20).mean()
        ma50 = hist['Close'].tail(50).mean()
        ma_alignment = current_price > ma20 and ma20 > ma50
        
        # Get fundamentals
        info = stock.info
        market_cap = info.get('marketCap', 0) / 1e9 if info.get('marketCap') else None
        pe_ratio = info.get('trailingPE')
        forward_pe = info.get('forwardPE')
        peg_ratio = info.get('pegRatio')
        beta = info.get('beta')
        sector = info.get('sector', 'Unknown')
        industry = info.get('industry', 'Unknown')
        
        # Print analysis
        print(f"\n📈 PRICE & MOMENTUM")
        print(f"  Current Price: ${current_price:.2f}")
        print(f"  5-day return:  {ret_5d:+.2f}%")
        print(f"  20-day return: {ret_20d:+.2f}%")
        print(f"  60-day return: {ret_60d:+.2f}%")
        print(f"  52w Range:     {price_range_pct:.1f}% (${low_52w:.2f} - ${high_52w:.2f})")
        
        print(f"\n📊 TECHNICAL")
        print(f"  Volatility (annual): {volatility:.1f}%")
        print(f"  MA Alignment (20>50): {'✅ YES' if ma_alignment else '❌ NO'}")
        print(f"  Price vs MA20: {((current_price/ma20 - 1)*100):+.2f}%")
        print(f"  Price vs MA50: {((current_price/ma50 - 1)*100):+.2f}%")
        print(f"  Volume Surge: {volume_surge:.2f}x")
        
        print(f"\n💼 FUNDAMENTALS")
        print(f"  Sector: {sector}")
        print(f"  Industry: {industry}")
        if market_cap:
            cap_category = "Large" if market_cap > 200 else "Mid" if market_cap > 10 else "Small"
            print(f"  Market Cap: ${market_cap:.1f}B ({cap_category})")
        if pe_ratio:
            print(f"  P/E Ratio: {pe_ratio:.1f}")
        if forward_pe:
            print(f"  Forward P/E: {forward_pe:.1f}")
        if peg_ratio:
            print(f"  PEG Ratio: {peg_ratio:.2f}")
        if beta:
            print(f"  Beta: {beta:.2f}")
        
        # Risk assessment
        print(f"\n⚠️  RISK ASSESSMENT")
        risks = []
        if volatility > 40:
            risks.append(f"High volatility ({volatility:.0f}%)")
        if price_range_pct > 90:
            risks.append("Near 52-week high (potential resistance)")
        if price_range_pct < 10:
            risks.append("Near 52-week low (potential support but bearish)")
        if not ma_alignment:
            risks.append("Moving averages not aligned (weak trend)")
        if ret_20d < -10:
            risks.append(f"Recent 20d decline ({ret_20d:.1f}%)")
        if pe_ratio and pe_ratio > 50:
            risks.append(f"High P/E ({pe_ratio:.0f}) - expensive valuation")
        
        if risks:
            for risk in risks:
                print(f"  ⚠️  {risk}")
        else:
            print(f"  ✅ No major red flags detected")
        
        # Buy recommendation
        print(f"\n🎯 VERDICT")
        score = 0
        if ret_20d > 0:
            score += 2
        if ret_60d > 0:
            score += 1
        if ma_alignment:
            score += 2
        if volume_surge > 1.2:
            score += 1
        if price_range_pct > 30 and price_range_pct < 85:
            score += 1
        if volatility < 35:
            score += 1
        if len(risks) == 0:
            score += 1
        
        if score >= 7:
            verdict = "🟢 STRONG BUY - Excellent setup"
        elif score >= 5:
            verdict = "🟡 MODERATE BUY - Good opportunity with some caution"
        elif score >= 3:
            verdict = "🟠 HOLD - Wait for better entry or more confirmation"
        else:
            verdict = "🔴 AVOID - Too many risk factors"
        
        print(f"  Score: {score}/9")
        print(f"  {verdict}")
        
        return {
            'Ticker': ticker,
            'Type': recommendation_type,
            'Price': current_price,
            'Ret_5d': ret_5d,
            'Ret_20d': ret_20d,
            'Ret_60d': ret_60d,
            'Volatility': volatility,
            'MA_Aligned': ma_alignment,
            'Volume_Surge': volume_surge,
            'Risk_Count': len(risks),
            'Score': score,
            'Verdict': verdict.split(' - ')[0],
        }
        
    except Exception as e:
        print(f"❌ Error analyzing {ticker}: {e}")
        return None


if __name__ == "__main__":
    # Get tickers from command line or use examples
    if len(sys.argv) > 1:
        tickers = sys.argv[1:]
        types = ["Unknown"] * len(tickers)
    else:
        # Default: analyze one from each category
        print("Usage: python validate_current_picks.py TICKER1 [TICKER2 ...]")
        print("Example: python validate_current_picks.py NVDA AAPL TSLA")
        sys.exit(1)
    
    results = []
    for ticker in tickers:
        result = analyze_stock(ticker.upper(), "User Input")
        if result:
            results.append(result)
    
    if results:
        df = pd.DataFrame(results)
        print(f"\n{'='*70}")
        print("📋 SUMMARY")
        print(f"{'='*70}")
        print(df.to_string(index=False))
        print(f"\n✅ Analysis complete for {len(results)} stocks")
