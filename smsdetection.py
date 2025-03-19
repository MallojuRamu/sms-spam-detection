import pandas as pd

# Activate virtual environment first (if using): venv\Scripts\activate
# Load the dataset
data = pd.read_csv('spam.csv', encoding='latin-1')

# Drop unnamed columns and rename
data = data[['v1', 'v2']]
data.columns = ['label', 'text']

# Explore
print(data.head())
print(data['label'].value_counts())