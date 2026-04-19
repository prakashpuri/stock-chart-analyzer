import os
import groq
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

# API Key load karo
api_key = None
try:
    with open(".env") as f:
        for line in f:
            if "ANTHROPIC_API_KEY" in line:
                api_key = line.strip().split("=")[1]
except:
    pass

client = groq.Groq(api_key=api_key)

def fetch_stock_data(symbol, period="3mo"):
    print(f"\n📊 {symbol} ka data fetch ho raha hai...")
    ticker_symbol = symbol + ".NS"
    stock = yf.Ticker(ticker_symbol)
    df = stock.history(period=period)
    if df.empty:
        stock = yf.Ticker(symbol)
        df = stock.history(period=period)
    return df, stock.info

def calculate_indicators(df):
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['Support'] = df['Low'].rolling(window=20).min()
    df['Resistance'] = df['High'].rolling(window=20).max()
    return df

def create_chart(df, symbol):
    os.makedirs("charts", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                    gridspec_kw={'height_ratios': [3, 1]})
    fig.patch.set_facecolor('#1a1a2e')
    ax1.set_facecolor('#16213e')
    ax1.plot(df.index, df['Close'], color='#00d4ff', linewidth=2, label='Price')
    ax1.plot(df.index, df['MA20'], color='#ffd700', linewidth=1.5,
             linestyle='--', label='MA20')
    ax1.plot(df.index, df['MA50'], color='#ff6b6b', linewidth=1.5,
             linestyle='--', label='MA50')
    ax1.fill_between(df.index, df['Support'], df['Resistance'],
                     alpha=0.1, color='green', label='Support-Resistance Zone')
    ax1.set_title(f'{symbol} - Stock Chart Analysis',
                  color='white', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Price', color='white')
    ax1.tick_params(colors='white')
    ax1.legend(loc='upper left', facecolor='#1a1a2e',
               labelcolor='white', fontsize=10)
    ax1.grid(True, alpha=0.2, color='gray')
    for spine in ax1.spines.values():
        spine.set_color('gray')
    ax2.set_facecolor('#16213e')
    ax2.plot(df.index, df['RSI'], color='#a855f7', linewidth=2, label='RSI')
    ax2.axhline(y=70, color='#ff6b6b', linestyle='--', alpha=0.7, label='Overbought (70)')
    ax2.axhline(y=30, color='#00ff88', linestyle='--', alpha=0.7, label='Oversold (30)')
    ax2.fill_between(df.index, df['RSI'], 70,
                     where=(df['RSI'] >= 70), color='red', alpha=0.3)
    ax2.fill_between(df.index, df['RSI'], 30,
                     where=(df['RSI'] <= 30), color='green', alpha=0.3)
    ax2.set_ylabel('RSI', color='white')
    ax2.set_ylim(0, 100)
    ax2.tick_params(colors='white')
    ax2.legend(loc='upper left', facecolor='#1a1a2e',
               labelcolor='white', fontsize=9)
    ax2.grid(True, alpha=0.2, color='gray')
    for spine in ax2.spines.values():
        spine.set_color('gray')
    plt.tight_layout()
    chart_path = f"charts/{symbol}_analysis.png"
    plt.savefig(chart_path, dpi=150, bbox_inches='tight',
                facecolor='#1a1a2e')
    plt.close()
    print(f"✅ Chart save ho gaya: {chart_path}")
    return chart_path

def analyze_with_ai(df, symbol):
    latest = df.tail(1).iloc[0]
    prev = df.tail(2).iloc[0]
    current_price = round(latest['Close'], 2)
    price_change = round(((latest['Close'] - prev['Close']) / prev['Close']) * 100, 2)
    rsi = round(latest['RSI'], 2)
    ma20 = round(latest['MA20'], 2)
    ma50 = round(latest['MA50'], 2)
    support = round(latest['Support'], 2)
    resistance = round(latest['Resistance'], 2)
    month_ago = df.tail(22).iloc[0]['Close']
    monthly_return = round(((current_price - month_ago) / month_ago) * 100, 2)

    prompt = f"""
Tu ek expert stock market analyst hai jo Indian retail investors ko
simple Hindi-English (Hinglish) mein samjhata hai.

Stock: {symbol}
Current Price: ₹{current_price}
Aaj ka Change: {price_change}%
1 Month Return: {monthly_return}%

Technical Indicators:
- RSI: {rsi}
- MA20: ₹{ma20}
- MA50: ₹{ma50}
- Support Level: ₹{support}
- Resistance Level: ₹{resistance}

Yeh analysis de simple aur clear tarike se:

1. 📈 TREND KYA HAI?
2. 🎯 SUPPORT & RESISTANCE
3. 📊 RSI SIGNAL
4. ⚡ MA SIGNAL
5. 🚦 OVERALL SIGNAL (Strong Buy/Buy/Hold/Sell/Strong Sell)
6. ⚠️ RISK WARNING

Hinglish mein likho. Yeh sirf educational analysis hai.
"""
    print("🤖 AI analysis kar raha hai...")
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def main():
    print("=" * 50)
    print("🚀 STOCK CHART ANALYZER AGENT")
    print("=" * 50)
    while True:
        symbol = input("\n📌 Stock symbol daalo (jaise RELIANCE, TCS, INFY) ya 'quit': ").upper().strip()
        if symbol == 'QUIT':
            print("👋 Agent band ho raha hai!")
            break
        if not symbol:
            continue
        try:
            df, info = fetch_stock_data(symbol)
            if df.empty or len(df) < 20:
                print("❌ Data nahi mila. Correct symbol daalo.")
                continue
            df = calculate_indicators(df)
            chart_path = create_chart(df, symbol)
            analysis = analyze_with_ai(df, symbol)
            print("\n" + "=" * 50)
            print(f"📊 {symbol} - AI ANALYSIS")
            print("=" * 50)
            print(analysis)
            print("=" * 50)
            print(f"\n💾 Chart yahan save hua: {chart_path}")
            os.system(f"start {chart_path}")
        except Exception as e:
            print(f"❌ Error aaya: {e}")

if __name__ == "__main__":
    main()