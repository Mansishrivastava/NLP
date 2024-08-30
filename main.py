import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report



# Load dataset
df = pd.read_csv('train.csv')

# Text Preprocessing
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_title = vectorizer.fit_transform(df['Title'])
X_body = vectorizer.fit_transform(df['Body'])

# Combine features
X = pd.concat([pd.DataFrame(X_title.toarray()), pd.DataFrame(X_body.toarray()), df[['Tags', 'CreationDate']]], axis=1)

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

