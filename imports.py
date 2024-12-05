import os
import re
import nltk
import nbimporter
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from utility import *
import plotly.io as pio
import tensorflow as tf
import plotly.express as px
import plotly.offline as pyo
nltk.download('vader_lexicon')
import matplotlib.pyplot as plt
from scipy.special import softmax
import plotly.graph_objects as go
from mlxtend.preprocessing import TransactionEncoder
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from mlxtend.frequent_patterns import apriori, association_rules
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, XLNetTokenizer, TFXLNetForSequenceClassification