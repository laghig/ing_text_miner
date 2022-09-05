from collections import Counter
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split

def merge(list1, list2):
      
    merged_list = tuple(zip(list1, list2)) 
    return merged_list

def data_balancing_summary(y, sampling_strategy):
    init_class_distribution = Counter(y)
    print(init_class_distribution)
    min_class_label, _ = init_class_distribution.most_common()[-4]
    sampling_strategy = sampling_strategy
    expected_n_samples = sampling_strategy.get(min_class_label)
    print(f'Expected number of generated samples: {expected_n_samples}')

def get_hyperparameters(algo):
    # KNN
    leaf_size = list(range(1,10)),
    n_neighbors = list(range(1,10)),
    p=[1,2]
    KNN = dict(clf__leaf_size=leaf_size, clf__n_neighbors=n_neighbors, clf__p=p)

    # Ridge and lasso regression
    alphas_l = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]
    alphas_r = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.90, 1]
    t_f = [True, False]
    lasso = dict(clf__alpha = alphas_l)
    ridge = dict(clf__alpha = alphas_r) # clf__positive=t_f

    # Naive Bayes
    alpha_nb = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
    nb = dict(clf__alpha = alpha_nb)

    hyperparameters = {'ridge':ridge,'lasso':lasso,'KNN': KNN, 'NaiveBayes': nb}
    return hyperparameters[algo]


def get_correlation(data, threshold):
    corr_col = set()
    cormat = data.corr()
    for i in range(len(cormat.columns)):
        for j in range(i):
            if abs(cormat.iloc[i,j]) > threshold:
                colname = cormat.columns[i]
                corr_col.add(colname)
    return corr_col

def delete_corr_features(vect_X, y):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(vect_X)
    df= pd.DataFrame(X.todense(),columns=vectorizer.get_feature_names_out())

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 11, stratify = y)

    constant_filter = VarianceThreshold(threshold = 0.0002)
    constant_filter.fit(x_train)

    feature_list = x_train[x_train.columns[constant_filter.get_support(indices=True)]]
    print('Number of selected features: ' ,len(list(feature_list)),'\n')
    # print('List of selected features: \n' ,list(feature_list))
    x_train_filter = constant_filter.transform(x_train)
    x_test_filter = constant_filter.transform(x_test)
    x_train_filter.shape, x_test_filter.shape, x_train.shape
    x_train_filter = pd.DataFrame(x_train_filter)
    x_test_filter = pd.DataFrame(x_test_filter)
    
    corr_features = get_correlation(x_train_filter, 0.70)
    x_train_uncorr = x_train_filter.drop(labels= corr_features, axis = 1)
    x_test_uncorr = x_test_filter.drop(labels= corr_features, axis = 1)
    x_train_uncorr = pd.DataFrame(x_train_uncorr)
    x_test_uncorr = pd.DataFrame(x_test_uncorr)
    x_train_uncorr.shape, x_test_uncorr.shape, x_train_filter.shape


if __name__ == "__main__":
    cleaned_dt = pd.read_pickle(r"C:\Users\Giorgio\Desktop\ETH\Code\interim_results\cleaned_data.pkl")
    X = cleaned_dt['text']
    y = cleaned_dt['co2_score']
   
    delete_corr_features(X, y)