import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from typing import List, Dict, Optional
import yfinance as yf
from datetime import datetime, timedelta
import praw
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import requests

from config import MarketFeatures, FeatureConfig

# ---------------------------------------------------------------------------
# Fetch Data
# ---------------------------------------------------------------------------

def fetch_market_data(symbol: str, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
    return yf.Ticker(symbol).history(start=start_date, end=end_date or datetime.now().strftime('%Y-%m-%d'), interval="1d")

def fetch_news_data(date: datetime, symbol: str, newsapi_key: str) -> List[str]:
    date_str = date.strftime('%Y-%m-%d')
    url = ('https://newsapi.org/v2/everything?'f'q={symbol}&'f'from={date_str}&'f'to={date_str}&''language=en&''sortBy=popularity&'f'apiKey={newsapi_key}')
    response = requests.get(url)
    if response.status_code == 200:
        articles = response.json().get('articles', [])
        return [f"{article['title']} {article['description']}"
                for article in articles if article['description']]
    return []

def fetch_reddit_data(date: datetime, symbol: str, reddit_client: praw.Reddit, subreddits: List[str] = ['wallstreetbets', 'stocks', 'investing']) -> List[str]:
    posts = []
    start_timestamp = datetime.combine(date, datetime.min.time()).timestamp()
    end_timestamp = datetime.combine(date + timedelta(days=1), datetime.min.time()).timestamp()
    for subreddit_name in subreddits:
        subreddit = reddit_client.subreddit(subreddit_name)
        for post in subreddit.search(f"{symbol}", limit=100):
            if start_timestamp <= post.created_utc < end_timestamp:
                posts.append(f"{post.title} {post.selftext}")
    return posts

# ---------------------------------------------------------------------------
# Calendar Features
# ---------------------------------------------------------------------------

def calculate_calendar_features(dates: pd.DatetimeIndex) -> Dict[str, torch.Tensor]:
    days = dates.dayofweek.values + 1
    months = dates.month.values

    return {
        'day'   :   torch.tensor(days, dtype=torch.float32).unsqueeze(-1),
        'month' :   torch.tensor(months, dtype=torch.float32).unsqueeze(-1)
    }

# ---------------------------------------------------------------------------
# Technical Indicators
# ---------------------------------------------------------------------------

def calculate_ohlcv(df: pd.DataFrame) -> torch.Tensor:
    ohlcv = df[['Open', 'High', 'Low', 'Close', 'Volume']].values
    return torch.tensor(ohlcv, dtype=torch.float32)

def calculate_returns(df: pd.DataFrame, use_log: bool = True) -> torch.Tensor:
    close_prices = torch.tensor(df['Close'].values, dtype=torch.float32)
    if use_log:
        returns = torch.log(close_prices[1:] / close_prices[:-1])
        returns = torch.cat([torch.tensor([0.0]), returns])
    else:
        pct_change = (close_prices[1:] - close_prices[:-1]) / close_prices[:-1]
        returns = torch.cat([torch.tensor([0.0]), pct_change])
    return returns.unsqueeze(-1)

def calculate_sma(df: pd.DataFrame, periods: List[int]) -> torch.Tensor:
    sma_features = []
    for period in periods:
        sma = df['Close'].rolling(window=period).mean().fillna(0).values
        sma_features.append(torch.tensor(sma))
    sma_tensor = torch.stack(sma_features, dim=1)

    return sma_tensor.float()

def calculate_ema(df: pd.DataFrame, periods: List[int]) -> torch.Tensor:
    ema_features = []
    for period in periods:
        ema = df['Close'].ewm(span=period, adjust=False).mean().values
        ema_features.append(torch.tensor(ema, dtype=torch.float32))

    return torch.stack(ema_features, dim=1)

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> torch.Tensor:
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rsi = (100 - (100 / (1 + gain / loss))).fillna(50).values

    return torch.tensor(rsi, dtype=torch.float32).unsqueeze(-1)

def calculate_macd(df: pd.DataFrame) -> torch.Tensor:
    macd = (df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()).values

    return torch.tensor(macd, dtype=torch.float32).unsqueeze(-1)

# ---------------------------------------------------------------------------
# Sentiment Analysis
# ---------------------------------------------------------------------------

def calculate_daily_sentiment(reddit_posts: List[str], news_articles: List[str], model, tokenizer, reddit_weight: float = 0.5) -> Dict[str, float]:
    def get_sentiment(texts: List[str]) -> float:
        if not texts:
            return 0.0
        combined_text = " ".join(texts[:5])
        inputs = tokenizer(combined_text, return_tensors="pt", truncation=True, max_length=512)
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return (probs[0][2] - probs[0][0]).item()

    reddit_score = get_sentiment(reddit_posts)
    news_score = get_sentiment(news_articles)
    combined_score = reddit_weight * reddit_score + (1 - reddit_weight) * news_score

    return {
        'reddit': reddit_score,
        'news': news_score,
        'combined': combined_score
    }

def calculate_sentiment_features(dates: pd.DatetimeIndex, symbol: str, reddit_credentials: Dict[str, str], newsapi_key: str, reddit_weight: float = 0.5) -> Dict[str, torch.Tensor]:
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    reddit_client = praw.Reddit(client_id=reddit_credentials['client_id'], client_secret=reddit_credentials['client_secret'], user_agent=reddit_credentials['user_agent'])

    seq_len = len(dates)
    reddit_sentiment = np.zeros(seq_len)
    news_sentiment = np.zeros(seq_len)
    combined_sentiment = np.zeros(seq_len)

    for i, date in enumerate(tqdm(dates, desc="Calculating sentiment")):
        reddit_posts = fetch_reddit_data(date, symbol, reddit_client)
        news_articles = fetch_news_data(date, symbol, newsapi_key)
        daily_sentiment = calculate_daily_sentiment(reddit_posts, news_articles, model, tokenizer, reddit_weight)
        reddit_sentiment[i] = daily_sentiment['reddit']
        news_sentiment[i] = daily_sentiment['news']
        combined_sentiment[i] = daily_sentiment['combined']
        time.sleep(1)

    return {
        'reddit': torch.tensor(reddit_sentiment, dtype=torch.float32).unsqueeze(-1),
        'news': torch.tensor(news_sentiment, dtype=torch.float32).unsqueeze(-1),
        'combined': torch.tensor(combined_sentiment, dtype=torch.float32).unsqueeze(-1)
    }

# ---------------------------------------------------------------------------
# Data Pipeline
# ---------------------------------------------------------------------------

def pipeline(symbol: str, start_date: str, end_date: Optional[str] = None, config: Optional[FeatureConfig] = None, reddit_credentials: Optional[Dict[str, str]] = None, newsapi_key: Optional[str] = None) -> MarketFeatures:
    config = FeatureConfig() if config is None
    df = yf.Ticker(symbol).history(start=start_date, end=end_date or datetime.now().strftime('%Y-%m-%d'), interval="1d")

    ohlcv = calculate_ohlcv(df)
    returns = calculate_returns(df, config.use_log_returns)
    sma = calculate_sma(df, config.sma_periods)
    ema = calculate_ema(df, config.ema_periods)
    rsi = calculate_rsi(df, config.rsi_period)
    macd = calculate_macd(df)
    calendar = calculate_calendar_features(df.index)

    if reddit_credentials and newsapi_key:
        sentiment = calculate_sentiment_features(df.index, symbol, reddit_credentials, newsapi_key, config.reddit_weight)
    else:
        seq_len = len(df)
        sentiment = { 'reddit': torch.zeros(seq_len, 1), 'news': torch.zeros(seq_len, 1), 'combined': torch.zeros(seq_len, 1) }

    return MarketFeatures(ohlvc=ohlcv, returns=returns, sma=sma, ema=ema, rsi=rsi, macd=macd, reddit_sentiment=sentiment['reddit'], news_sentiment=sentiment['news'], combined_sentiment=sentiment['combined'], day=calendar['day'], month=calendar['month'])
