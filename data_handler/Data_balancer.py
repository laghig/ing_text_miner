from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE

def random_upsampler(vectorized_X,y): # ,random_state
    ros = RandomOverSampler() # random_state
    X_resampled, y_resampled = ros.fit_resample(vectorized_X, y)

    return X_resampled, y_resampled

def smote_oversampler(vectorized_X,y):
    smote = SMOTE(random_state=777,k_neighbors=5)
    X_smote,y_smote = smote.fit_resample(vectorized_X,y)

    return X_smote, y_smote