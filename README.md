# Crypto Market Analysis Bot

An automated cryptocurrency market analysis tool that monitors market data, generates predictions, and posts insights to X.

## Overview

This agent continuously monitors cryptocurrency market data for multiple tokens, analyzes price movements, volume patterns, and market correlations, and then posts insights and predictions to X. It features smart money detection, trend analysis, and prediction accuracy tracking.

## Features

- **Real-time Market Monitoring**: Tracks price movements, volume changes, and correlations across multiple tokens
- **Smart Money Detection**: Identifies potential institutional activity through volume pattern analysis
- **Automated Twitter Posts**: Posts market analyses and predictions with configurable triggers
- **Prediction Engine**: Generates price predictions using multiple models:
  - Technical analysis
  - Statistical models (ARIMA)
  - Machine learning (RandomForest)
  - Claude AI enhanced predictions
- **Accuracy Tracking**: Evaluates prediction accuracy and maintains performance metrics
- **Duplicate Detection**: Smart time-based duplicate post prevention
- **Market Correlation Analysis**: Monitors inter-token relationships and market trends

## Setup

### Prerequisites

- Python 3.8+
- Chrome browser and ChromeDriver
- X account
- Anthropic Claude API key
- CoinGecko API access

### Installation

1. Clone the repository:
   ```
   git clone Agent-Yugo
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with the following variables:
   ```
   # API Keys
   CLAUDE_API_KEY=your_claude_api_key
   
   # Twitter Credentials
   TWITTER_USERNAME=your_twitter_username
   TWITTER_PASSWORD=your_twitter_password
   
   # Chrome Configuration
   CHROME_DRIVER_PATH=/path/to/chromedriver
   
   # Optional Google Sheets Configuration
   GOOGLE_SHEETS_PROJECT_ID=your_project_id
   GOOGLE_SHEETS_PRIVATE_KEY=your_private_key
   GOOGLE_SHEETS_CLIENT_EMAIL=your_client_email
   GOOGLE_SHEET_ID=your_sheet_id
   ```

4. Initialize the database:
   The database will be automatically created on first run in the `data/` directory.

## Usage

### Running the Agent

To start the agent:

```
python src/bot.py
```

The agent will:
1. Initialize the database (if not already created)
2. Log in to X
3. Begin monitoring market data
4. Generate and post analyses based on configured triggers

### Configuration

Main configuration settings can be adjusted in `src/config.py`:

- `BASE_INTERVAL`: Time between market checks (in seconds)
- `PRICE_CHANGE_THRESHOLD`: Minimum price change to trigger post (percentage)
- `VOLUME_CHANGE_THRESHOLD`: Minimum volume change to trigger post (percentage)
- `VOLUME_WINDOW_MINUTES`: Rolling window for volume trend analysis
- `TRACKED_CRYPTO`: Dictionary of tokens to track

### Prediction Configuration

Prediction settings are managed through the `PREDICTION_CONFIG` in `config.py`:

- `enabled_timeframes`: Supported prediction timeframes
- `confidence_threshold`: Minimum confidence level for predictions
- `prediction_frequency`: Minutes between predictions for the same token
- `fomo_factor`: Multiplier for interest-generating predictions

## Troubleshooting

### Common Issues

- **"NoneType is not iterable" Error**: Usually occurs when the agent can't retrieve token data or market information. Check your internet connection and CoinGecko API status.
- **Twitter Login Failures**: Can happen due to:
  - Incorrect credentials in `.env`
  - Twitter's anti-automation measures
  - Changes in Twitter's login page structure
- **Claude API Errors**: Verify your API key is correct and hasn't expired

### Logs

Logs are stored in the `logs/` directory:
- `eth_btc_correlation.log`: Main application log
- `coingecko.log`: CoinGecko API interactions
- `claude.log`: Claude API interactions

## License

MIT

## Disclaimer

This software is for informational purposes only. It is not financial advice. Always do your own research before making investment decisions.
