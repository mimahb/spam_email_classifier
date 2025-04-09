import nltk
import pandas as pd # type: ignore
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import FreqDist, NaiveBayesClassifier
from nltk.classify import accuracy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

sample_email_text = "michael pobega wrote i'm not sure if it's the mpl or mozilla that didn't allow the distribution of their images or the patching of programs without their knowledge but i think that is not dfsg free last time i looked the mozilla images were in an other licenses directory so not under the mpl and not licensed to others at all hope that helps mjr slef my opinion only see http people debian org mjr please follow"


def clean_text(text):
    stop_words = set(stopwords.words('english'))  # Set of stopwords to remove

    """Preprocess the input text."""
    if not isinstance(text, str):  # Ensure the text is a string
        text = ""
    words = word_tokenize(text.lower())  # Tokenize and convert to lowercase
    return [word for word in words if word.isalpha() and word not in stop_words]  # Keep only alphabetic words



def main_ai(request, email_content):
    # Download necessary NLTK data files
    nltk.download('punkt')
    nltk.download('stopwords')

    # Load dataset
    spam_df = pd.read_csv('/content/spam data.csv')

    # Preprocessing the dataset
    spam_df = spam_df[['text', 'label']].sample(frac=1).reset_index(drop=True)  # Shuffle dataset

    # Apply text cleaning to the 'text' column
    spam_df['processed_text'] = spam_df['text'].apply(clean_text)

    # Extract most common words (2000 most frequent)
    all_words = [word for words in spam_df['processed_text'] for word in words]
    word_features = list(FreqDist(all_words).keys())[:2000]

    # Function to extract features for each document (email)
    def document_features(words):
        """Convert text into a feature set (binary features for each word)."""
        words = set(words)
        return {word: (word in words) for word in word_features}
    
    
    # Convert the dataset into feature sets for model training
    featuresets = [(document_features(words), label) for words, label in zip(spam_df['processed_text'], spam_df['label'])]

    # Split dataset into train and test sets (80% train, 20% test)
    train_set, test_set = train_test_split(featuresets, test_size=0.2, random_state=42)

    # Train Naive Bayes classifier
    classifier = NaiveBayesClassifier.train(train_set)

    # Evaluate the classifier's accuracy
    accuracy_score = accuracy(classifier, test_set)
    print(f"Accuracy: {accuracy_score * 100:.2f}%")

    # Print first 10 word features
    print("First 10 Word Features:", word_features[:10])

    # Process the email
    processed_email = clean_text(sample_email[0])
    email_features = document_features(processed_email)

    # Make a prediction
    prediction = classifier.classify(email_features)
    results = ("Prediction:", "Spam" if prediction == 1 else "Ham")


    
    # # Save model and vectorizer for future use
    # joblib.dump(classifier, 'spam_classifier.pkl')

    # # Create and save the TF-IDF vectorizer
    # vectorizer = TfidfVectorizer(max_features=2000)  # Adjust max_features as needed
    # vectorizer.fit(spam_df["text"])  # Fit vectorizer on the text data
    # joblib.dump(vectorizer, 'vectorizer.pkl')

    return results




