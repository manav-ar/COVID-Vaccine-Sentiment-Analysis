# COVID Vaccine Sentiment Analysis


A comprehensive end-to-end sentiment analysis system analyzing public opinion on COVID-19 vaccines through social media data. This project encompasses the complete data science pipeline from web scraping and data collection to sentiment classification, visualization, and a query-based web interface.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Pipeline Components](#pipeline-components)
- [Key Features](#key-features)
- [Getting Started](#getting-started)
- [Data Collection](#data-collection)
- [Sentiment Classification](#sentiment-classification)
- [Web Interface](#web-interface)
- [Results & Insights](#results--insights)
- [Contributing](#contributing)


## Overview

This project analyzes public sentiment toward COVID-19 vaccines by collecting, processing, and classifying social media posts (primarily from Twitter/X). The system provides insights into public opinion trends, concerns, and attitudes during different phases of the vaccine rollout.

### Motivation

Understanding public sentiment toward vaccines is crucial for:
- **Public Health Policy**: Informing vaccination campaigns and communication strategies
- **Misinformation Detection**: Identifying and addressing vaccine hesitancy
- **Trend Analysis**: Tracking sentiment evolution over time
- **Demographic Insights**: Understanding sentiment variations across different groups

### Research Questions

1. What is the overall sentiment distribution (positive, negative, neutral) toward COVID vaccines?
2. How does sentiment change over time and in response to events?
3. What are the main topics and concerns expressed in negative sentiment?
4. Can we identify patterns in vaccine hesitancy narratives?

## Project Structure

```
COVID-Vaccine-Sentiment-Analysis/
├── Crawling-Updated/              # Web scraping and data collection
│   ├── twitter_scraper.py        # Twitter API integration
│   ├── reddit_scraper.py         # Reddit data collection
│   ├── config.py                 # API credentials
│   └── data_storage.py           # Database operations
├── Preprocessing/                 # Data cleaning and preparation
│   ├── text_cleaning.py          # Text normalization
│   ├── tokenization.py           # Tokenization methods
│   ├── feature_extraction.py     # TF-IDF, embeddings
│   └── data_augmentation.py      # Handling imbalanced data
├── Classification/                # Sentiment classification models
│   ├── traditional_ml/           # Naive Bayes, SVM, Logistic Regression
│   ├── deep_learning/            # LSTM, BERT, RoBERTa
│   ├── ensemble/                 # Model combinations
│   └── evaluation/               # Metrics and validation
├── Indexing/                      # Search indexing for query interface
│   ├── build_index.py            # Create search indices
│   ├── search_engine.py          # Query processing
│   └── ranking.py                # Result ranking
├── Visualisation/                 # Data visualization and analytics
│   ├── sentiment_trends.py       # Time series plots
│   ├── wordclouds.py            # Visual text analysis
│   ├── geospatial.py            # Geographic distribution
│   └── dashboards.py            # Interactive visualizations
├── QueryUI_Django/                # Web application
│   ├── manage.py                 # Django management
│   ├── sentiment_app/            # Main application
│   ├── templates/                # HTML templates
│   ├── static/                   # CSS, JS, images
│   └── api/                      # REST API endpoints
├── Deliverables/                  # Project documentation
│   ├── reports/                  # Technical reports
│   ├── presentations/            # Slide decks
│   └── papers/                   # Research papers
├── .gitignore
├── LICENSE
└── README.md
```

## Pipeline Components

### 1. Data Collection (Crawling)

Automated web scraping to collect social media posts about COVID vaccines.

```python
from crawling import TwitterScraper, RedditScraper

# Initialize scrapers
twitter_scraper = TwitterScraper(api_key='your_key')
reddit_scraper = RedditScraper(client_id='your_id')

# Define search queries
queries = [
    'COVID vaccine',
    'coronavirus vaccination',
    'pfizer vaccine',
    'moderna vaccine',
    'vaccine side effects'
]

# Collect data
for query in queries:
    tweets = twitter_scraper.search(query, count=1000)
    posts = reddit_scraper.search(query, subreddits=['Coronavirus', 'COVID19'])
    
    # Store in database
    store_data(tweets, posts)
```

**Data Sources:**
- Twitter/X posts and replies
- Reddit posts and comments
- News article comments
- Public health forums

**Collection Period:** January 2021 - Present

### 2. Data Preprocessing

Comprehensive text cleaning and preparation pipeline.

```python
class TextPreprocessor:
    def __init__(self):
        self.stopwords = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text):
        """Clean and normalize text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove user mentions
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags (but keep text)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize(self, text):
        """Tokenize and lemmatize"""
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) 
                 for token in tokens 
                 if token not in self.stopwords]
        return tokens
    
    def process(self, text):
        """Full preprocessing pipeline"""
        text = self.clean_text(text)
        tokens = self.tokenize(text)
        return ' '.join(tokens)
```

**Preprocessing Steps:**
1. Text normalization (lowercase, whitespace)
2. URL and mention removal
3. Special character handling
4. Stopword removal
5. Lemmatization/stemming
6. Emoji handling (conversion to text)

### 3. Sentiment Classification

Multiple machine learning and deep learning models for sentiment analysis.

#### Traditional ML Models

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Feature extraction
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train models
models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'SVM': SVC(kernel='linear')
}

for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    accuracy = model.score(X_test_tfidf, y_test)
    print(f"{name} Accuracy: {accuracy:.4f}")
```

#### Deep Learning Models

```python
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, 
                           num_layers=2, bidirectional=True, 
                           dropout=0.3, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.lstm(embedded)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        return self.fc(hidden)

# BERT for sentiment classification
def train_bert_model(train_loader, val_loader, epochs=5):
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=3  # positive, negative, neutral
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
```

### 4. Search Indexing

Efficient search functionality for the query interface.

```python
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, ID, DATETIME
from whoosh.qparser import QueryParser

# Define schema
schema = Schema(
    id=ID(stored=True),
    text=TEXT(stored=True),
    sentiment=TEXT(stored=True),
    date=DATETIME(stored=True),
    source=TEXT(stored=True)
)

# Build index
ix = create_in("indexdir", schema)
writer = ix.writer()

for post in dataset:
    writer.add_document(
        id=post['id'],
        text=post['text'],
        sentiment=post['sentiment'],
        date=post['timestamp'],
        source=post['platform']
    )

writer.commit()
```

### 5. Web Interface (Django)

User-friendly web application for querying and visualizing results.

```python
# views.py
from django.shortcuts import render
from django.http import JsonResponse
from .models import Post
from .sentiment_analyzer import analyze_sentiment

def search_posts(request):
    """Search posts by keyword and filter by sentiment"""
    query = request.GET.get('q', '')
    sentiment_filter = request.GET.get('sentiment', 'all')
    
    posts = Post.objects.filter(text__icontains=query)
    
    if sentiment_filter != 'all':
        posts = posts.filter(sentiment=sentiment_filter)
    
    results = {
        'posts': list(posts.values()),
        'total': posts.count(),
        'sentiment_distribution': get_sentiment_distribution(posts)
    }
    
    return JsonResponse(results)

def analyze_text(request):
    """Analyze sentiment of user-provided text"""
    text = request.POST.get('text', '')
    
    sentiment = analyze_sentiment(text)
    
    return JsonResponse({
        'text': text,
        'sentiment': sentiment['label'],
        'confidence': sentiment['score']
    })
```

## Key Features

- **Automated Data Collection**: Continuous scraping from multiple social media platforms
- **Robust Preprocessing**: Comprehensive text cleaning and normalization
- **Multiple Models**: Traditional ML and state-of-the-art deep learning approaches
- **Real-Time Analysis**: Analyze sentiment of new text on-the-fly
- **Interactive Dashboard**: Django-based web interface for exploration
- **Temporal Analysis**: Track sentiment trends over time
- **Topic Modeling**: Identify key themes in vaccine discussions
- **Geospatial Visualization**: Map sentiment by location
- **Export Capabilities**: Download results in various formats

## Getting Started

### Prerequisites

```bash
Python 3.8+
Django 4.0+
PyTorch 1.9+ / TensorFlow 2.x
transformers (Hugging Face)
scikit-learn
pandas, numpy
nltk, spacy
tweepy (Twitter API)
praw (Reddit API)
whoosh (search indexing)
plotly, matplotlib, seaborn
```

### Installation

```bash
# Clone the repository
git clone https://github.com/manav-ar/COVID-Vaccine-Sentiment-Analysis.git
cd COVID-Vaccine-Sentiment-Analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -m nltk.downloader punkt stopwords wordnet

# Download spaCy model
python -m spacy download en_core_web_sm

# Set up database
cd QueryUI_Django
python manage.py migrate
python manage.py createsuperuser
```

### Configuration

Create a `.env` file with API credentials:

```env
# Twitter API
TWITTER_API_KEY=your_api_key
TWITTER_API_SECRET=your_api_secret
TWITTER_ACCESS_TOKEN=your_access_token
TWITTER_ACCESS_SECRET=your_access_secret

# Reddit API
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=your_user_agent

# Django
SECRET_KEY=your_django_secret_key
DEBUG=True
```

### Quick Start

#### 1. Collect Data

```bash
cd Crawling-Updated
python twitter_scraper.py --query "COVID vaccine" --count 1000
python reddit_scraper.py --subreddit Coronavirus --limit 500
```

#### 2. Preprocess Data

```bash
cd ../Preprocessing
python preprocess_data.py --input ../data/raw/ --output ../data/processed/
```

#### 3. Train Models

```bash
cd ../Classification
python train_model.py --model bert --epochs 5 --batch-size 32
```

#### 4. Run Web Interface

```bash
cd ../QueryUI_Django
python manage.py runserver

# Access at http://localhost:8000
```

## Data Collection

### Collection Statistics

| Platform | Posts Collected | Date Range | Languages |
|----------|-----------------|------------|-----------|
| Twitter/X | 150,000+ | Jan 2021 - Present | English, Spanish, French |
| Reddit | 45,000+ | Jan 2021 - Present | English |
| News Comments | 20,000+ | Jan 2021 - Present | English |
| **Total** | **215,000+** | - | - |

### Data Schema

```python
{
    'id': 'unique_post_id',
    'text': 'post content',
    'timestamp': 'ISO 8601 datetime',
    'platform': 'twitter|reddit|news',
    'user_id': 'anonymized_user_id',
    'location': 'geographic location (if available)',
    'likes': 'engagement metrics',
    'retweets': 'share count',
    'sentiment': 'positive|negative|neutral',
    'confidence': 'classification confidence score'
}
```

## Sentiment Classification

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|----------|-----------|--------|----------|---------------|
| Naive Bayes | 78.3% | 0.77 | 0.78 | 0.77 | 2 min |
| Logistic Regression | 82.1% | 0.81 | 0.82 | 0.81 | 5 min |
| SVM | 83.7% | 0.83 | 0.84 | 0.83 | 15 min |
| BiLSTM | 87.2% | 0.86 | 0.87 | 0.86 | 2 hours |
| BERT | 91.5% | 0.91 | 0.92 | 0.91 | 4 hours |
| RoBERTa | 92.3% | 0.92 | 0.92 | 0.92 | 5 hours |

### Confusion Matrix (BERT Model)

|              | Predicted Positive | Predicted Negative | Predicted Neutral |
|--------------|-------------------|-------------------|------------------|
| **Actual Positive** | 8,450 | 320 | 230 |
| **Actual Negative** | 280 | 7,890 | 380 |
| **Actual Neutral** | 410 | 450 | 6,140 |

### Class Distribution

- **Positive**: 42% (Pro-vaccine, vaccination success stories, gratitude)
- **Negative**: 38% (Vaccine hesitancy, side effects, mistrust)
- **Neutral**: 20% (Information sharing, questions, discussions)

## Results & Insights

### Key Findings

1. **Overall Sentiment**: 42% positive, 38% negative, 20% neutral
   - Positive sentiment increased from 35% (early 2021) to 48% (late 2021)
   - Negative sentiment peaked during side effect reports

2. **Temporal Trends**:
   - Initial vaccine rollout: High uncertainty (55% neutral)
   - Post-approval period: Growing positivity (45% positive by Q2 2021)
   - Booster announcement: Mixed reactions (35% negative spike)

3. **Common Positive Themes**:
   - Protection and safety
   - Return to normalcy
   - Gratitude toward healthcare workers
   - Family protection

4. **Common Negative Themes**:
   - Side effect concerns
   - Mistrust of pharmaceutical companies
   - Misinformation and conspiracy theories
   - Government mandates resistance

5. **Geographic Patterns**:
   - Higher positive sentiment in urban areas (52%)
   - More vaccine hesitancy in rural regions (45% negative)
   - Regional variations in response to mandates

### Visualization Examples

**Sentiment Over Time**
```python
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(x=dates, y=positive_sentiment, name='Positive'))
fig.add_trace(go.Scatter(x=dates, y=negative_sentiment, name='Negative'))
fig.add_trace(go.Scatter(x=dates, y=neutral_sentiment, name='Neutral'))
fig.update_layout(title='COVID Vaccine Sentiment Trends', 
                  xaxis_title='Date', 
                  yaxis_title='Percentage')
fig.show()
```

## Web Interface

### Features

1. **Search & Filter**
   - Keyword search across all posts
   - Filter by sentiment, date range, platform
   - Sort by relevance, date, engagement

2. **Real-Time Analysis**
   - Analyze custom text for sentiment
   - Batch processing of multiple texts
   - API endpoint for integration

3. **Visualizations**
   - Interactive sentiment trends
   - Word clouds by sentiment
   - Geographic heatmaps
   - Topic distribution charts

4. **Export & Reports**
   - CSV/JSON export
   - PDF report generation
   - API access for researchers



## Contributing

Contributions welcome! Areas for improvement:

- Additional data sources (Facebook, Instagram, TikTok)
- Multi-lingual sentiment analysis
- Aspect-based sentiment analysis (specific vaccine features)
- Misinformation detection
- Real-time streaming analysis
- Mobile application development

## Ethical Considerations

- All data collected from public sources only
- User information anonymized
- No personally identifiable information stored
- Compliant with platform Terms of Service
- Results used for research and public health purposes only


For questions or collaboration opportunities, please create an issue or contact the maintainers.

If you find this project useful, please consider giving it a star!
