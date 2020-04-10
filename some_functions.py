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
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, confusion_matrix

plt.style.use('seaborn-ticks')
sns.set_style('ticks')
plt.rcParams['figure.figsize'] = (8, 8)
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['axes.labelsize'] = 22
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20

pd.options.display.max_columns = 1000

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def plot_corr(correlations, mask=False, threshold=0.9, annot=False):
    """
    Plot a correlation matrix as a heatmap
    masking coefficients below the threshold.
    """
    if mask:
        mask = np.zeros_like(correlations)
        mask[abs(correlations) < threshold] = 1

    plt.rcParams['figure.figsize'] = (10, 10)

    sns.heatmap(correlations, 
                mask=mask, annot=annot,
                vmin=-1, vmax=1,
                cmap=sns.color_palette("RdBu_r", 100));
    

def create_palette(df, by='stiffness', show=False):
    """
    Create a palette to make pretty plots.
    """
    if by == 'stiffness':
        n_levels = df.stiffness.unique().size
        palette = dict(zip(df.stiffness.unique(), sns.color_palette("Set3", n_levels)))
        row_colors = df.stiffness.map(palette)
    elif by == 'cluster':
        n_levels = df.cluster.unique().size
        if n_levels == 2:
            palette = {0 : sns.color_palette("PRGn", 20)[15], 1 : sns.color_palette("PRGn", 20)[4]}
        else:
            palette = dict(zip(df.cluster.unique(), sns.color_palette("Set2", n_levels)))
        row_colors = df.cluster.map(palette)
    elif by == 'biom':
        palette = {df.biom.unique()[0] : sns.color_palette("RdBu", 10)[1],
                   df.biom.unique()[1] : sns.color_palette("RdBu", 10)[8]}
        row_colors = df.biom.map(palette)
    elif by == 'comb':
        palette = {df.combination.unique()[0] : sns.color_palette("RdBu", 10)[1],
                   df.combination.unique()[1] : sns.color_palette("RdBu", 10)[8]}
        row_colors = df.combination.map(palette)
    elif by == 'isclumped':
        palette = {0 : sns.color_palette("Set3", 6)[4], 1 : sns.color_palette("Set3", 6)[5]}
        row_colors = df.isclumped.map(palette)
        
    if show:
#         print(list(palette.keys()))
        sns.palplot(palette.values());
    
    return row_colors, palette
# plt.savefig('../results/My palette.png', bbox_inches='tight', dpi=300);
    
    
def pca(df, cols):
    """
    Scale data and run PCA.
    Returns the first N components
    explaining 90% of variance.
    """
    # Feature map
    X = df[cols]
    
    # Standartise the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Number of components requred to describe 90% variance
    pca = PCA(n_components=0.9)

    # Apply PCA to the scaled feature matrix
    X_reduced = pca.fit_transform(X_scaled)

    pc_cols = [('pc_' + str(i)) for i in range(1, X_reduced.shape[1] + 1)]
    pc_df = pd.concat([df.loc[:, ['label', 'combination', 'stiffness', 'cluster']],
                      pd.DataFrame(data = X_reduced, columns = pc_cols)], 
                      axis=1)

    print("The first {} components explain 90% of variance.\n".format(pca.n_components_ ))

    print("Explained variance: {}\n".format(pca.explained_variance_ratio_))

    print(X.shape, pca.components_.shape, X_reduced.shape, pc_df.shape)
    
    return pca, pc_cols, pc_df


def hc(df, cols, n_clusters=2, save=True):
    X = df[cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    agg = AgglomerativeClustering(n_clusters=n_clusters)
    agg.fit(X_scaled) 

#     1 is epit, 0 => 2 is mesench
#     if n_clusters == 2:
#         df['cluster'] = np.where(agg.labels_ == 1, 1, 2)
#     else:
#         df['cluster'] = agg.labels_
    df['cluster'] = agg.labels_ + 1
    
    cluster_type = CategoricalDtype(categories=list(range(1, n_clusters+1)), ordered=True)
    df.cluster = df.cluster.astype(cluster_type)

    row_colors, palette = create_palette(df, by='cluster')

    # Cluster map
    sns.clustermap(pd.DataFrame(X_scaled, columns=cols), 
                   metric='euclidean', method='ward',
                   col_cluster=False,
                   cmap=sns.color_palette('RdBu_r', 100), 
                   vmin=-2.5, vmax=2.5,
                   robust=True, row_colors=row_colors, 
    #                figsize=(6.5, 10), 
                   xticklabels=False, yticklabels=False);
    if save:
        plt.savefig('../results/Clustering geom_cols.png', bbox_inches='tight', dpi=300);
        
        
def ttest(df, by, col, equal_var, transform=None, verbose=True):
    """
    Run a t-testWelch t-test or a Welch t-test for unequal variances.
    """
    x_start = []
    x_end = []
    signif = []
    
    group = df[by].unique().sort_values()
    for i in range(len(group) - 1):
        x1 = df.loc[df[by] == group[i], col]
        x2 = df.loc[df[by] == group[i + 1], col]
        
        if transform == 'log':
            x1 = np.log(x1)
            x2 = np.log(x2)
        elif transform == 'boxcox':
            x1,_ = stats.boxcox(x1)
            x2,_ = stats.boxcox(x2)
            
        _, p = stats.ttest_ind(x1, x2, equal_var=equal_var, nan_policy='omit')
        if p < 0.05:
            x_start.append(i)
            x_end.append(i + 1)
            if p < 0.001:
                sign = '***'
            elif p < 0.01:
                sign = '**'
            elif p < 0.05:
                sign = '*'
            signif.append(sign)
        else:
            sign = ''
            
        if verbose:
            if equal_var:
                print("A two-tailed t-test on samples {} vs {}. {} \t p-value = {:.2}."
                      .format(group[i], group[i + 1],  sign, p))
            else:
                print("Welch's unequal variances t-test on samples {} vs {}. {} \t p-value = {:.2}."
                      .format(group[i], group[i + 1],  sign, p))
            
    return x_start, x_end, signif


def mannwhitneytest(df, by, col, alternative, transform=None):
    """
    Run a Mann-whitney U test.s
    """
    group = df[by].unique()
    for i in range(len(group) - 1):
        x1 = df.loc[df[by] == group[i], col]
        x2 = df.loc[df[by] == group[i + 1], col]
        
        if transform == 'log':
            x1 = np.log(x1)
            x2 = np.log(x2)
        elif transform == 'boxcox':
            x1,_ = stats.boxcox(x1)
            x2,_ = stats.boxcox(x2)
            
        _, p = stats.mannwhitneyu(x1, x2, alternative=alternative, nan_policy='omit')
        if p < 0.001:
            sign = '***'
        elif p < 0.01:
            sign = '**'
        elif p < 0.05:
            sign = '*'
        else:
            sign = ''

        print("A {} Mann Whitney test on samples {} vs {}. {} \t p-value = {:.2}."
              .format(alternative, group[i], group[i + 1],  sign, p))
        

        
def stat_annot(df, by, col, x_start, x_end, signif, ylim, kind='barplot'):
    """
    Add annotation to show the results of statistical testing.
    """
    s = df[by].unique().sort_values()
    h = (ylim[1] - ylim[0])/50
    
    for x1, x2, label in zip(x_start, x_end, signif):

        if kind == 'barplot':
            y = max(df.loc[df[by] == s[x1], col].mean() + df.loc[df[by] == s[x1], col].std(), 
                    df.loc[df[by] == s[x2], col].mean() + df.loc[df[by] == s[x2], col].std()) + h
        elif kind == 'boxplot':
            y = max(upper_whisker(df.loc[df[by] == s[x1], col]), 
                    upper_whisker(df.loc[df[by] == s[x2], col])) + h

        plt.plot([x1+0.05, x1+0.05, x2-0.05, x2-0.05], [y, y + h, y + h, y], 
                 lw=1.5, color='k');

        plt.text((x1 + x2) * 0.5, y + h, s=label, 
                 ha='center', va='bottom', 
                 color='k', fontsize=20);
        plt.ylim(ylim)
        
        
def upper_whisker(x):
    iqr = x.quantile(0.75) - x.quantile(0.25)
    return x.quantile(0.75) + 1.5 * iqr