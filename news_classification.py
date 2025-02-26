import pandas as pd

# Specify the full path to your CSV file
file_path = r"C:\Users\bafan\news_classification_project\news_classification\news_classification_en_zu.csv"

# Load the dataset into a DataFrame
df = pd.read_csv(file_path)

# Display the first few rows of the dataset to inspect its structure
print(df.head())

# Print the column names to see the available columns
print(df.columns)
# Select the relevant columns (category and zulu_news)
df_zulu = df[['category', 'zulu_news']]

# Save the extracted data to a new CSV file
df_zulu.to_csv("zulu_news_classification.csv", index=False)

# Display the first few rows of the new DataFrame to confirm
print(df_zulu.head())
import matplotlib.pyplot as plt
import seaborn as sns

# Count the number of articles in each category
category_counts = df_zulu['category'].value_counts()

# Display the class distribution
print(category_counts)

# Plot the class distribution
plt.figure(figsize=(10, 6))
sns.countplot(y='category', data=df_zulu, order=category_counts.index, palette='viridis')
plt.title('Distribution of News Categories')
plt.xlabel('Number of Articles')
plt.ylabel('Category')
plt.show()

from sklearn.feature_extraction.text import CountVectorizer

# Initialize CountVectorizer to tokenize the words
vectorizer = CountVectorizer(stop_words='english', max_features=20)

# Fit and transform the Zulu news headlines
X = vectorizer.fit_transform(df_zulu['zulu_news'])

# Get the word frequencies
word_freq = X.toarray().sum(axis=0)

# Create a DataFrame for the word frequencies
word_freq_df = pd.DataFrame(word_freq, index=vectorizer.get_feature_names_out(), columns=["Frequency"])

# Sort the word frequencies
word_freq_df = word_freq_df.sort_values(by="Frequency", ascending=False)

# Display the top 10 most frequent words
print(word_freq_df.head(10))

# Plot the word frequencies
plt.figure(figsize=(10, 6))
word_freq_df.head(10).plot(kind='bar', legend=False, color='skyblue')
plt.title('Top 10 Most Frequent Words in Zulu News Headlines')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Preprocessing - Vectorization with TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

# Vectorize the Zulu news headlines
X = vectorizer.fit_transform(df_zulu['zulu_news'])

# Step 2: Prepare the target variable (category) and split the data
y = df_zulu['category']

# Split data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Step 4: Make predictions
y_pred = clf.predict(X_test)

# Step 5: Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


#Alternative Models
from sklearn.linear_model import LogisticRegression

# Train a Logistic Regression classifier
lr_clf = LogisticRegression(max_iter=1000, random_state=42)
lr_clf.fit(X_train, y_train)

# Make predictions
y_pred_lr = lr_clf.predict(X_test)

# Evaluate the model
print("Accuracy (Logistic Regression):", accuracy_score(y_test, y_pred_lr))
print("Classification Report (Logistic Regression):\n", classification_report(y_test, y_pred_lr))


from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Print the best parameters and best score
print("Best parameters found:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# Evaluate the best model on the test data
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

print("Accuracy (Best Model):", accuracy_score(y_test, y_pred_best))
print("Classification Report (Best Model):\n", classification_report(y_test, y_pred_best))


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# Define Stratified K-Fold Cross Validator
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracies = []

for train_index, test_index in cv.split(X, y):
    X_train_cv, X_test_cv = X[train_index], X[test_index]
    y_train_cv, y_test_cv = y[train_index], y[test_index]
    
    # Train the model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_cv, y_train_cv)
    
    # Evaluate the model
    y_pred_cv = clf.predict(X_test_cv)
    accuracies.append(accuracy_score(y_test_cv, y_pred_cv))

print("Cross-Validation Accuracy: ", sum(accuracies)/len(accuracies))


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# Step 1: Label Encoding the target variable (category)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Step 2: Padding the sequences
max_len = 100  # Maximum length of news headlines
X_padded = pad_sequences(X.toarray(), maxlen=max_len, padding='post', truncating='post')

# Step 3: Building the LSTM model
model = Sequential()
model.add(Embedding(input_dim=X_padded.shape[1], output_dim=128, input_length=max_len))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(64))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

# Step 4: Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Step 5: Train the model
history = model.fit(X_padded, y_encoded, epochs=5, batch_size=32, validation_split=0.2)

# Step 6: Evaluate the model
loss, accuracy = model.evaluate(X_padded, y_encoded)
print(f"Accuracy: {accuracy*100:.2f}%")


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Get the confusion matrix
cm = confusion_matrix(y_test, y_pred_best)

# Plot the confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
