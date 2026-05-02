# Amazon-Comment-Sentiment-Risk-Monitor

## 📖 Overview
This project analyzes Amazon product reviews and Reddit discussions to monitor sentiment and identify potential risks. It includes data cleaning, sentiment analysis, and an alarm system to detect negative trends or potential brand risks.

##  Features
- **Data Collection**: Automated scraping of Reddit discussions
- **Data Cleaning**: Preprocessing and cleaning of Amazon and Reddit data
- **Sentiment Analysis**: Analyze user comment sentiment (positive/negative/neutral)
- **Risk Monitoring**: Identify negative sentiment spikes and risk keywords
- **Alarm System**: Automated alerts when high-risk indicators are detected

## Project Structure

| File | Description |
|------|-------------|
| `Reddit_data get.py` | **Data Scraper**: Fetches relevant discussion data from Reddit |
| `Reddit-data cleaning.py` | **Data Cleaning**: Processes and cleans raw Reddit data |
| `Amazon data cleaning code.py` | **Data Cleaning**: Cleans and standardizes Amazon review data |
| `Amazon-analysis.py` | **Core Analysis**: Performs sentiment analysis on Amazon data |
| `Amazon-analysis2.py` | **Advanced Analysis**: Additional analysis or visualization |
| `Alarm system demo.py` | **Alarm Demo**: (Latest) Demonstrates risk alert functionality |

##  Getting Started

### Prerequisites
- Python 3.x
- Required libraries (pandas, numpy, requests, etc.)
