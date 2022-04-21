import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import * # porter stemmer was developed for the english language
from nltk.stem.cistem import * # Cistem seems to be the best stemmer for the german language

def data_cleaning(df, language):

    # Stop words
    en_stopwords = stopwords.words('english')
    de_stopwords = stopwords.words('german')
    # add few words to the stop words dictionary
    en_stopwords.extend(['ingredients'])
    de_stopwords.extend(['Zutaten'])

    if language=='en':
        # Use Porter stemmer
        stemmer = PorterStemmer()
    
    elif language == 'de':
        # Use Cistem stemmer
        stemmer = Cistem()

    df1= pd.DataFrame(columns = ['text', 'prod_id', 'ubp_score'])

    for index, row in df.iterrows():
        ingr_ls = row['text'].translate(str.maketrans('', '', string.punctuation))
        ingr_ls = word_tokenize(ingr_ls)
        filtered_words = [word.lower() for word in ingr_ls if word not in de_stopwords if language == 'de' else en_stopwords]
        ingr_ls_stem = [stemmer.stem(word) for word in filtered_words]

        ingr_clean = [' '.join( ingr for ingr in ingr_ls_stem)]

        ingr_clean.append(row['prod_id'])
        ingr_clean.append(row['ubp_score'])

        new_row = pd.Series(ingr_clean, index = df1.columns)
        df1 = df1.append(new_row, ignore_index=True)

    return df1