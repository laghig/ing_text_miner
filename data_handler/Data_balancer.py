from imblearn.over_sampling import RandomOverSampler

def random_upsampler(X,y): # ,random_state
    ros = RandomOverSampler() # random_state
    X_resampled, y_resampled = ros.fit_resample(X, y)

    return X_resampled, y_resampled