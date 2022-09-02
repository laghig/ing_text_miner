import pandas as pd
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import * # porter stemmer was developed for the english language
from nltk.stem.cistem import * # Cistem seems to be the best stemmer for the german language
from nltk.stem.snowball import FrenchStemmer 

# Own imports
from bags_of_words import country_list_de, country_codes_de, add_stopwords_de, units
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

# -------- Cleaning functions ----------------

def remove_numbers(text):
    text = re.sub("\d+", "", text)
    return text

def replace_brackets(text):
    text = str(text).replace("(", ", ").replace(")", " ,")
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

def group_ing(text):
    swapped_word_list = {
        word: replacement
        for replacement, words in grouped_words.items()
        for word in words
    }

    new_text = ' '.join([
            swapped_word_list.get(word, word)
            for word in text.split()
                ])
    
    return new_text

def remove_stop_words_en(text):
    # Stop words - standard dictionary
    en_stopwords = stopwords.words('english')

    # Ingredients list- specific additions
    en_stopwords.extend(['ingredients', 'emulsifiers'])

    filtered_words = [word for word in text if word not in en_stopwords]

    return filtered_words

def remove_stop_words_de(text):
    # Stop words - standard dictionary
    de_stopwords = stopwords.words('german')
    # Additional stopwords
    extra_stopwords = country_list_de + country_codes_de + add_stopwords_de + units
    de_stopwords.extend(extra_stopwords)
    
    ing_list = [word for word in text if word not in de_stopwords]
    # ing_list = ' '.join([str(elem) for elem in filtered_words])
    return ing_list

def bag_of_words_de(text):

    filtered_words = [word for word in text if word in key_words]
    ing_list = ' '.join([str(elem) for elem in filtered_words])
    return ing_list

def remove_superscript(text):
    text = ''.join([i for i in text if ord(i) < 128])
    return text

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

    if language == 'de':
        df = df[df.text != 'Produkt']
        df = df[df.text != 'Keine Zutatenliste']
        df = df[df.text != 'Keine Zutatenliste.'] 
        df = df[df.text != 'KEINE ZUTATENLISTE.']
        df = df[df.text != 'Keine.']
        return df

def clean_OFF_dataframe(df, language):
    # Drop empty values
    df.dropna(subset=['ingredients_text_{}'.format(language), 'ecoscore_grade'],inplace=True)
    
    # Other deletions
    df = df[df.ecoscore_grade != 'unknown'] 
    df = df[df.ecoscore_grade != 'not-applicable']
    return df

def first_five_ing(text):
    """
    Input: string of the ingredients text separated by comma
    Output: string with only the fist five ingredients
    """
    text = ",".join(text.split(",",5)[:-1])
    return text

def text_cleaning(df, language, column, model_modifications):

    if model_modifications['OnlyFive'] is True:
        df[column]= df[column].apply(lambda x: first_five_ing(x))

    df[column] = df[column].apply(lambda x: replace_brackets(x))
    
    # remove punctuation
    df[column] = df[column].apply(lambda x: remove_punct_subset(x))
    
    # remove numbers
    df[column] = df[column].apply(lambda x: remove_numbers(x))

    # Tokenization an lower case only
    df[column]= df[column].apply(lambda x: word_tokenize(x.lower()))

    if language == 'en':
        # Filter out stop words
        df[column]= df[column].apply(lambda x: remove_stop_words_en(x))
        #Stemming
        df[column]=df[column].apply(lambda x: word_stemmer_en(x))
    
    if language == 'de':
        df[column]= df[column].apply(lambda x: remove_stop_words_de(x))
        # df[column]= df[column].apply(lambda x: bag_of_words_de(x))

        df[column]=df[column].apply(lambda x: word_stemmer_de(x))


        # remove the remaining punctuation
        df[column] = df[column].apply(lambda x: remove_punctuation(x))
        # df[column]=df[column].apply(lambda x: group_ing(x))
         
    
    if language == 'fr':
        df[column]= df[column].apply(lambda x: remove_stop_words_fr(x))
        df[column]=df[column].apply(lambda x: word_stemmer_fr(x))

    # Merge ingredients comoosed by multiple words
    # df[column] = df[column].apply(lambda x: merge_ing(x))

    return df

if __name__ == "__main__":
    
    # TEST
    # random_text = " sugar, palm oil, hazelnuts 13%, skimmed milk powder 8,7%, lean cocoa is 7,4%, emulsifiers : lecithins [soy] , vanillin,"

    # clean_text = remove_punctuation(random_text)
    # print(clean_text)
    # clean_text=word_tokenize(clean_text.lower())
    # print(clean_text)
    # clean_text=remove_stop_words_en(clean_text)
    # print(clean_text)
    # clean_text=word_stemmer_en(clean_text)
    # print(clean_text)

    # Model/text Modifications:
    # clean= decimal_with_point(rnd_text)
    # print(first_five_ing(clean))

    clean = group_ing('biovollmilch pasteurisier lacto schweiz bergmilch biorahm schweiz salz jodfluorfrei schweiz sauerung')
    print(clean)