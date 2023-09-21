#  IGN Games Recommendation System

In this project, using IGN Games dataset, a Recommendation System has been built using the cosine similarity metric, and the approach employed is Content-Based Recommendation. This recommendation system analyzes and suggests game recommendations to users based on the content and features of the games themselves. The content features used include game genres, score, score phrase, release date, platform, and URL. By calculating the cosine similarity between game content vectors represented using TF-IDF (Term Frequency-Inverse Document Frequency) features, the system identifies games that are similar in terms of their content and provides personalized recommendations to users.

## Dataset
- In 20 years, the gaming industry has grown sophisticated. By exploring this dataset, one will be able to find trends about industries, compare consoles against each other, search through the most popular genres and more.
- It contains 18625 data points with various features such as release dates with different platform along with IGN scores.

## Column Description
There are 11 columns in this dataset.

Unnamed: 0,'score_phrase', 'title', 'url', 'platform', 'score',
       'genre', 'editors_choice', 'release_year', 'release_month',
       'release_day

## SKills

- Python
- Pandas
- Numpy
- Matplotlib
- scikit-learn
- Data visualization
- Data Preprocessing
- Content Vectorization
- Cosine Similarity Calculation
- Function to get Game Recommendations

## Installation
```bash
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
```
## Source    
https://www.kaggle.com/datasets/joebeachcapital/ign-games

