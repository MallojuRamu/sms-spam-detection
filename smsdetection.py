import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

nltk.download('punkt')
nltk.download('punkt_tab')  # Added this line
nltk.download('stopwords')

data = pd.read_csv('spam.csv', encoding='latin-1')
data = data[['v1', 'v2']]
data.columns = ['label', 'text']

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)
data['clean_text'] = data['text'].apply(preprocess_text)

data['label'] = data['label'].map({'spam': 1, 'ham': 0})

vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(data['clean_text']).toarray()
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data preprocessed and split. Training set shape:", X_train.shape)