import pandas as pd
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import * # porter stemmer was developed for the english language
from nltk.stem.cistem import * # Cistem seems to be the best stemmer for the german language
from nltk.stem.snowball import FrenchStemmer 

# Own imports
from data_handler.bags_of_words import *
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
        bag of words methodology --> language dependent
        stemming -> language dependency: en, de, fr
"""

# ---------- Dataframe Cleaning ----------------

def clean_dataframe(df, db, language):
    std_text = {
        'en': ['-', 'Product information’s are not available in English'],
        'de': ['-', 'Produkt', 'Keine Zutatenliste', 'Keine Zutatenliste.', 'KEINE ZUTATENLISTE.', 'Keine.'],
        'fr': ['-', 'Produkt']
    }

    if db =='Eatfit':
        # Drop empty values
        df.dropna(inplace=True)
        # Other deletions:
        for x in std_text[language]:
            df = df[df.text != x]
    else:
        # Drop empty rows
        df.dropna(subset=['ingredients_text_{}'.format(language), 'ecoscore_grade'],inplace=True)
        # Other deletions
        std_text = ['unknown', 'not-applicable']
        for x in std_text:
            df = df[df.ecoscore_grade != x]
    
    return df

# ---------- Text Cleaning functions ----------------

def remove_numbers(text):
    text = re.sub("\d+", "", text)
    return text

def replace_brackets(text):
    text = str(text).replace("(", ", ").replace(")", " ,")
    return text

def remove_superscript(text):
    text = ''.join([i for i in text if ord(i) < 128])
    return text

def remove_punctuation(text):
    no_punct = text.translate(str.maketrans('', '', string.punctuation))
    return no_punct

def remove_punct_subset(text):
    to_delete = string.punctuation.replace(",", "¹").replace("()", "")
    cleaned_text = text.translate(str.maketrans('', '', to_delete))
    return cleaned_text

def merge_ing(text):
    text = re.sub("(?<=\w) +(?=\w+)", "", text)
    return text

def decimal_with_point(text):
    text = re.sub('(?<=\d),(?=\d)', '.', text)
    return text

# ---------- Grouping, stemming, ... -----------------

def group_ing(text):
    """
    Group synonyms using reverse dictionary mapping
    """
    swapped_word_list = {
        word: replacement
        for replacement, words in grouped_words_de.items()
        for word in words
    }

    new_text = ' '.join([
            swapped_word_list.get(word, word)
            for word in text.split()
                ])
    
    return new_text

def remove_stop_words(text, language):
    if language == 'de':
        # Stop words - standard dictionary
        std_stopwords = stopwords.words('german')
        # Additional stopwords
        ing_stopwords = country_list_de + country_codes_de + add_stopwords_de + units
    elif language == 'en':
        std_stopwords = stopwords.words('english')
        ing_stopwords = country_list_en + country_codes_en + add_stopwords_en + units
    else:
        std_stopwords = stopwords.words('french')
        ing_stopwords = ['Ingrédients', 'ingrédient', 'Produkt',]

    std_stopwords.extend(ing_stopwords)
    ing_list = [word for word in text if word not in std_stopwords]

    return ing_list

def bag_of_words_de(text):
    """
    Delete all the words not included in the key_words list
    """
    # Add the key_words list in the bag_of_words script
    filtered_words = [word for word in text if word in key_words]
    ing_list = ' '.join([str(elem) for elem in filtered_words])

    return ing_list

def word_stemmer(text, language):
    stemmers = {
        'en': PorterStemmer(),
        'de': Cistem(),
        'fr': FrenchStemmer()
    }

    stemmer = stemmers[language]
    words_stem = " ".join([stemmer.stem(word) for word in text])

    return words_stem

# ---------- Other functions ----------------

def first_ing(text, splits):
    """
    Input: string of the ingredients text separated by comma
    Output: string with only the fist x ingredients
    """
    text = ",".join(text.split(",",splits)[:-1])
    return text

# ------------- Pre-processing pipeline ------------

def text_cleaning(df, language, column, model_modifications):
    """
    Function to call the pre-processing steps in a successive order
    """

    if model_modifications['first_x_ing'] is True:
        df[column]= df[column].apply(lambda x: first_ing(x, model_modifications['x']))

    df[column] = df[column].apply(lambda x: replace_brackets(x))
    df[column] = df[column].apply(lambda x: remove_punct_subset(x)) # remove punctuation except ','
    df[column] = df[column].apply(lambda x: remove_numbers(x))  # remove numbers
    df[column]= df[column].apply(lambda x: word_tokenize(x.lower())) # tokenization and lower case only
    df[column]= df[column].apply(lambda x: remove_stop_words(x, language)) # stopwords
    # df[column]= df[column].apply(lambda x: bag_of_words_de(x)) # bag of words
    df[column]=df[column].apply(lambda x: word_stemmer(x, language)) # stemming
    # df[column] = df[column].apply(lambda x: merge_ing(x)) # merge
    df[column] = df[column].apply(lambda x: remove_punctuation(x)) # comma
    # df[column]=df[column].apply(lambda x: group_ing(x)) # grouping

    return df

if __name__ == "__main__":
    
    # TEST
    # random_text = " sugar, palm oil, hazelnuts 13%, skimmed milk powder 8,7%, lean cocoa is 7,4%, emulsifiers : lecithins [soy] , vanillin,"
    # clean_text = remove_punctuation(random_text)
    # print(clean_text)
    # clean_text=word_tokenize(clean_text.lower())
    # print(clean_text)
    # clean_text=remove_stop_words(clean_text, 'en')
    # print(clean_text)
    # clean_text=word_stemmer(clean_text, 'en')
    # print(clean_text)

    # Model/text Modifications:
    # clean= decimal_with_point(rnd_text)
    # print(first_ing(clean))

    clean = group_ing('biovollmilch pasteurisier lacto schweiz bergmilch biorahm schweiz salz jodfluorfrei schweiz sauerung')
    print(clean)