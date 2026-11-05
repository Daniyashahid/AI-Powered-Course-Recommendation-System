import pandas as pd
import spacy
import re

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    text = str(text)

    # remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # lowercase
    text = text.lower()

    # lemmatization + stopwords removal
    doc = nlp(text)
    cleaned = " ".join(
        token.lemma_ for token in doc
        if not token.is_stop and not token.is_punct
    )

    return cleaned


# Load dataset
df = pd.read_csv("data/courses.csv")

# IMPORTANT: change column names if needed
TEXT_COLUMN = "description"

# Remove empty rows
df = df.dropna(subset=[TEXT_COLUMN])

# Apply cleaning
df["clean_description"] = df[TEXT_COLUMN].apply(clean_text)

# Save cleaned dataset
df.to_csv("data/clean_courses.csv", index=False)

print("✅ Dataset preprocessing completed successfully!")
