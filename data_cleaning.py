import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import * # porter stemmer was developed for the english language
from nltk.stem.cistem import * # Cistem seems to be the best stemmer for the german language
from nltk.stem.snowball import FrenchStemmer 

# nltk.download('stopwords')
"""
Cleaning:
    clean_dataframe:
        remove empty values
        removed standard-filled rows
    
    text_cleaning:
        no punctuation
        lowercase only
        remove stop words -> language dependency: en, de, fr
        stemming -> language dependency: en, de, fr
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
    de_stopwords.extend(['Zutaten', 'Bezeichnung', 'Herkunft', 'gefangen'])

    filtered_words = [word for word in text if word not in de_stopwords]

    return filtered_words

def remove_stop_words_fr(text):
    # Stop words - standard dictionary
    fr_stopwords = stopwords.words('french')

    # Ingredients list- specific additions
    fr_stopwords.extend(['Ingrédients', 'ingrédient', 'Produkt',])

    filtered_words = [word for word in text if word not in fr_stopwords]

    return filtered_words

def word_stemmer_de(text):

    stemmer = Cistem()
    words_stem = " ".join([stemmer.stem(word) for word in text])

    return words_stem

def word_stemmer_en(text):

    stemmer = PorterStemmer()   
    words_stem = " ".join([stemmer.stem(word) for word in text])

    return words_stem

def word_stemmer_fr(text):

    stemmer = FrenchStemmer()   
    words_stem = " ".join([stemmer.stem(word) for word in text])

    return words_stem

def clean_dataframe(df, language):
    # Drop empty values
    df.dropna(inplace=True)

    # Other deletions:
    df = df[df.text != '-']

    if language == 'en':
        df = df[df.text != 'Product information’s are not available in English']

    if language == 'fr':
        df = df[df.text != 'Produkt'] # add here all prod_id with the text in german

    return df

def clean_OFF_dataframe(df, language):
    # Drop empty values
    df.dropna(subset=['ingredients_text_{}'.format(language), 'ecoscore_grade'],inplace=True)
    
    # Other deletions
    df = df[df.ecoscore_grade != 'unknown'] 
    df = df[df.ecoscore_grade != 'not-applicable']
    return df

def text_cleaning(df, language, column):
    
    # remove punctuation
    df[column] = df[column].apply(lambda x: remove_punctuation(x))

    # Tokenization an lower case only
    df[column]= df[column].apply(lambda x: word_tokenize(x.lower()))

    if language == 'en':
        # Filter out stop words
        df[column]= df[column].apply(lambda x: remove_stop_words_en(x))
        #Stemming
        df[column]=df[column].apply(lambda x: word_stemmer_en(x))
    
    if language == 'de':
        df[column]= df[column].apply(lambda x: remove_stop_words_de(x))
        df[column]=df[column].apply(lambda x: word_stemmer_de(x))
    
    if language == 'fr':
        df[column]= df[column].apply(lambda x: remove_stop_words_fr(x))
        df[column]=df[column].apply(lambda x: word_stemmer_fr(x))

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