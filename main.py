import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime

# Load dataset
df = pd.read_csv('data.csv')

# Text Preprocessing
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_title = vectorizer.fit_transform(df['Title'])
X_body = vectorizer.fit_transform(df['Body'])

# Convert TF-IDF matrices to DataFrames and set column names as strings
X_title_df = pd.DataFrame(X_title.toarray())
X_body_df = pd.DataFrame(X_body.toarray())

# Ensure column names are strings
X_title_df.columns = X_title_df.columns.astype(str)
X_body_df.columns = X_body_df.columns.astype(str)

# Encode Tags using One-Hot Encoding
tags_encoder = OneHotEncoder()
X_tags = tags_encoder.fit_transform(df[['Tags']]).toarray()
X_tags_df = pd.DataFrame(X_tags, columns=tags_encoder.get_feature_names_out(['Tags']))

# Convert CreationDate to datetime and extract features
df['CreationDate'] = pd.to_datetime(df['CreationDate'])
df['Year'] = df['CreationDate'].dt.year
df['Month'] = df['CreationDate'].dt.month
df['Day'] = df['CreationDate'].dt.day
df['Hour'] = df['CreationDate'].dt.hour

# Combine all features
X = pd.concat([X_title_df, X_body_df, X_tags_df, df[['Year', 'Month', 'Day', 'Hour']]], axis=1)

# Encode target
y = df['Y'].apply(lambda x: 1 if x == 'LQ_CLOSE' else 0)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Prediction and Evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
