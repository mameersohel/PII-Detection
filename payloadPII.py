import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

nlp = spacy.load("en_core_web_md")

EMRdataset = pd.read_csv('/Users/ameersohel/Desktop/Natural Lang/NLP project/payload-normal.csv')

# Specify the columns containing text data
column_names = ['FIRST', 'LAST', 'RACE', 'BIRTHPLACE', 'ADDRESS']  # Add your column names here

# Initialize lists to store entities and labels
pii = []
label_list = []

for column in column_names:
    docs = list(nlp.pipe(EMRdataset[column]))

    for doc in docs:
        # Extract pii and labels using NER
        text = [ent.text_with_ws for ent in doc.ents if ent.label_ == 'PERSON']
        label = int(bool(text))
        # Add pii-names and labels to the lists
        pii.append(' '.join(text))
        label_list.append(label)

##print(pii + label_list)

# Create DataFrame
df = pd.DataFrame({'Text': pii, 'Label': label_list})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['Label'], test_size=0.3, random_state=5)

# TF-IDF Vectorization using CountVectorizer and TfidfTransformer
count_vectorizer = CountVectorizer()
X_train_counts = count_vectorizer.fit_transform(X_train.astype('U'))  # Convert to Unicode string
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts).toarray()  # Convert to dense array

X_test_counts = count_vectorizer.transform(X_test.astype('U'))  # Convert to Unicode string
X_test_tfidf = tfidf_transformer.transform(X_test_counts).toarray()  # Convert to dense array

#run Naive Bayes classifier
naivebayes = MultinomialNB()
naivebayes.fit(X_train_tfidf, y_train)

#run Logistic Regression classifier
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train_tfidf, y_train)

# Predictions for Naive Bayes
y_pred_nb = naivebayes.predict(X_test_tfidf)

# Predictions for Logistic Regression
y_pred_lr = logistic_regression.predict(X_test_tfidf)

#calculation of different evaluations for Naive Bayes
accuracy_nb = accuracy_score(y_test, y_pred_nb)
precision_nb = precision_score(y_test, y_pred_nb)
recall_nb = recall_score(y_test, y_pred_nb)
f1_nb = f1_score(y_test, y_pred_nb)

#calculation of different evaluation for Logistic Regression
accuracy_lr = accuracy_score(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)

#print evaluation metrics for Naive Bayes
print("Naive Bayes:")
print(f"Accuracy: {accuracy_nb}")
print(f"Precision: {precision_nb}")
print(f"Recall: {recall_nb}")
print(f"F1 Score: {f1_nb}")

#print evaluation metrics for Logistic Regression
print("Logistic Regression:")
print(f"Accuracy: {accuracy_lr}")
print(f"Precision: {precision_lr}")
print(f"Recall: {recall_lr}")
print(f"F1 Score: {f1_lr}")
