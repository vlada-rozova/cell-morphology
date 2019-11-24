import os
import pandas as pd
import numpy as np
import preprocessing as proc
import randomforest as rf
from pandas.api.types import CategoricalDtype
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
import joblib

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor 
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, confusion_matrix

def initial_param_grid():
    """
    Parameter values for the first run 
    of random forest tuning.
    """
    
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

    # Number of features to consider at every split
    max_features = [int(x) for x in np.linspace(10, 100, num = 10)]
    max_features.append('auto')

    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)

    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]

    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]

    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    # Create the random grid
    param_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    
    return param_grid

def random_search_param(model, X_train, y_train, param_grid=None):
    """
    Randomised search of best parameter values 
    using K-fold cross-validation. 
    """
    
    if param_grid is None:
        param_grid = initial_param_grid()
        
    random_search = RandomizedSearchCV(estimator = model, 
                                       param_distributions = param_grid, 
                                       n_iter = 100, cv = 3, 
                                       iid=False,
                                       verbose=2, random_state=42, n_jobs = -1)
    
    random_search.fit(X_train, y_train)
    print("Best parameters:\n", random_search.best_params_)
    
    return random_search.best_estimator_

def test_rf_model(model, reg, X_train, X_test, y_train, y_test):
    """
    Evaluate the best model.
    Set parameter `reg` to 1 for regression
    or 0 for classificcation.
    """
    
    if reg:
        # Baseline
        print("Null RMSE:", np.sqrt(mean_squared_error(y_test, y_test.apply(lambda x: np.mean(y_train)))))
    
        # Make predictions
        y_pred = model.predict(X_test)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
        print("Testing RMSE:", rmse_test)
        
    else:   
        # Null accuracy
        print("\nNumber of observations of each class in the training set:\n", y_train.value_counts())
        print("\nNumber of observations of each class in the test set:\n", y_test.value_counts())
        most_freq_class = y_train.value_counts().index[0]
        n_most_freq = y_test.value_counts()[most_freq_class]
        print("\nNull accuraccy:", n_most_freq / len(y_test))
    
        # Make predictions
        y_pred = model.predict(X_test)
        print("Testing accuraccy:", accuracy_score(y_test, y_pred))
        print("Testing f-score:", f1_score(y_test, y_pred, average='macro'))
    
        # Confusion matrix
        labels = y_test.unique()
        confusion = confusion_matrix(y_test, y_pred, labels=labels)
        confusion_df = pd.DataFrame(confusion, columns=labels, index=labels)
        print(confusion_df)

def get_important_features(rf_model, feature_names):
    """
    Extract and plot the most important features
    identified by a random forest model.
    """
    
    # Create a dataframe with feature names
    feature_importance = pd.DataFrame(feature_names, columns=['feature'])
    
    # Add weight and std
    feature_importance['weight'] = rf_model.feature_importances_
    # feature_importance['weight_sd'] = np.std([tree.feature_importances_ for tree in rf_clf.estimators_], axis=0)

    # Sort features by weight
    feature_importance.sort_values(by='weight', ascending=False, inplace=True)

    # Top 10 percentile
    threshold = feature_importance.weight.quantile(0.9)
    important_features = feature_importance[feature_importance.weight > threshold]
    print("Important features\n", important_features)
    
    # Plot feature importance
    plt.rcParams['figure.figsize'] = (15, 6)
    sns.barplot(x='feature', y='weight', data=feature_importance,  color='orangered')
    plt.plot([9, 9], [-0.001, feature_importance.weight.max()], linewidth=5, color='royalblue')
    plt.xticks([]);
    plt.xlabel("Features");
    plt.ylabel("Weight");
    plt.title("Feature importance");
    plt.savefig('../results/Feature importance.png', bbox_inches='tight', dpi=300);

    return important_features