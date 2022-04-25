import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import * # porter stemmer was developed for the english language
from nltk.stem.cistem import * # Cistem seems to be the best stemmer for the german language

"""
Cleaning:
    clean_dataframe:
        remove empty values
        removed standard-filled rows
    
    text_cleaning:
        no punctuation
        lowercase only
        remove stop words -> language dependency
        stemming
"""

def remove_punctuation(text):
    no_punct = text.translate(str.maketrans('', '', string.punctuation))
    return no_punct

def remove_stop_words_en(text):
    # Stop words - standard dictionary
    en_stopwords = stopwords.words('english')

    # Ingredients list- specific additions
    en_stopwords.extend(['ingredients'])

    filtered_words = [word for word in text if word not in en_stopwords]

    return filtered_words

def remove_stop_words_de(text):
    # Stop words - standard dictionary
    de_stopwords = stopwords.words('german')

    # Ingredients list- specific additions
    de_stopwords.extend(['Zutaten'])

    filtered_words = [word for word in text if word not in de_stopwords]

    return filtered_words

def word_stemmer_de(text):

    stemmer = Cistem()
    words_stem = " ".join([stemmer.stem(word) for word in text])

    return words_stem

def word_stemmer_en(text):

    # Use Porter stemmer
    stemmer = PorterStemmer()   
    words_stem = " ".join([stemmer.stem(word) for word in text])

    return words_stem

def clean_dataframe(df, language):
    # Drop empty values
    df.dropna(inplace=True)

    # Other deletions:
    df = df[df.text != '-']

    if language == 'en':
        df = df[df.text != 'Product information’s are not available in English']

    return df

def text_cleaning(df, language):
    
    # remove punctuation
    df['text'] = df['text'].apply(lambda x: remove_punctuation(x))

    # Tokenization an lower case only
    df['text']= df['text'].apply(lambda x: word_tokenize(x.lower()))

    # Filter out stop words
    if language == 'en':
        df['text']= df['text'].apply(lambda x: remove_stop_words_en(x))
    
    if language == 'de':
        df['text']= df['text'].apply(lambda x: remove_stop_words_de(x))
    
    #Stemming
    if language == 'en':
        df['text']=df['text'].apply(lambda x: word_stemmer_en(x))
    
    if language == 'de':
        df['text']=df['text'].apply(lambda x: word_stemmer_de(x))

    return df

if __name__ == "__main__":
    
    # TEST
    random_text = "I love veggies, but I can't imagine to not eat meat - it's just too good! I: ?just try ; to eat^ less of it."

    clean_text = remove_punctuation(random_text)
    print(clean_text)
    clean_text=word_tokenize(clean_text.lower())
    print(clean_text)
    clean_text=remove_stop_words_en(clean_text)
    print(clean_text)
    clean_text=word_stemmer_en(clean_text)
    print(clean_text)