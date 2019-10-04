import os
import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype
from imblearn.over_sampling import SMOTE

DATA_PATH = '../cell-profiler/measurements'


def load_data(filename, data_path=DATA_PATH):
    """ 
    Read a csv file.
    """
    csv_path = os.path.join(data_path, filename)
    return pd.read_csv(csv_path)

def save_data(df, filename, data_path=DATA_PATH):
    """ 
    Write into a csv file.
    """
    csv_path = os.path.join(data_path, filename)
    df.to_csv(csv_path)
    
def parse_metadata(df):
    """
    Parse metadata tp extract information about experimental conditions
    and drop unnecessary columns.
    """
    # Drop unnecessary columns
    columns = ['Metadata_Frame', 'Metadata_Series',
               'Metadata_Stiffness.1', 'Metadata_Combination.1', 
               'Metadata_Well.1', 'Metadata_Site.1'
              ]
    df.drop(columns, axis=1, inplace=True)
    
    # Rename columns containing metadata
    df.rename(columns={'ImageNumber' : 'image', 'ObjectNumber' : 'object', 
                       'Metadata_Stiffness' : 'stiffness', 
                       'Metadata_Combination' : 'combination', 
                       'Metadata_Well' : 'well', 
                       'Metadata_Site' : 'site'}, inplace=True)
    
    # Change types and create cell and image labels 
    df = create_label(df)
    df = create_label(df, col_name='image', per_cell=False)

def as_stiff_type(x):
    """
    Convert stiffness values to a custom categorical type.
    """
    # Create a categorical data type for stiffness
    stiff_type = CategoricalDtype(categories=['0.2', '0.5', '2.0', '8.0', '16.0', '32.0', '64.0'], ordered=True)
    return x.astype(stiff_type)
    
    
def create_label(df, col_name='label', per_cell=True):
    """
    Create a unique label for each observation.
    Labels are created as follows:
    For each site: stiffness-combination-well-site
    For each cell: stiffness-combination-well-site-object
    """
    # Convert to non-numeric values
    columns = ['image', 'object', 'stiffness', 'combination', 'well', 'site']
    existing_columns = [col for col in columns if col in df.columns]
    df[existing_columns] = df[existing_columns].astype(str)
   
    df.stiffness = as_stiff_type(df.stiffness)
    
    # Create a unique label for each cell
    if per_cell:
        df[col_name] = df[['stiffness', 'combination', 'well', 'site', 'object']].apply(lambda x: '-'.join(x), axis=1)
    else:
        df[col_name] = df[['stiffness', 'combination', 'well', 'site']].apply(lambda x: '-'.join(x), axis=1)
    
    return df
    
def rename_columns(df):
    """
    Rename columns containing features measured by cell profiler.
    """
    # Convert to lower case
    df.columns = [col.lower() for col in df.columns]

    # Rename channels
    df.columns = [col.replace('_origdapi', '_dapi', 1) for col in df.columns]
    df.columns = [col.replace('_origwga', '_wga', 1) for col in df.columns]
    df.columns = [col.replace('_origker', '_ker', 1) for col in df.columns]
    df.columns = [col.replace('_origvim', '_vim', 1) for col in df.columns]
    
    # Coordinates in X
    df.columns = [col.replace('_x', 'X', 1) for col in df.columns]

    # Coordinates in Y
    df.columns = [col.replace('_y', 'Y', 1) for col in df.columns]

    # Coordinates in Z
    df.columns = [col.replace('_z', 'Z', 1) for col in df.columns]

    # Shape features
    df.columns = [col.replace('areashape_', '', 1) for col in df.columns]
    
    # Zernike features
    df.columns = [col.replace('areashapeZernike', 'zernike', 1) for col in df.columns]

    # Intensity features
    df.columns = [col.replace('intensity_', '', 1) for col in df.columns]

    # Location
    df.columns = [col.replace('location_', 'loc_', 1) for col in df.columns]
    
    # Texture
    new_names = []
    for col in df.columns:
        if 'texture_' in col:
            new_names.append(col.replace('_3', '', 1))
        else:
            new_names.append(col)
    df.columns = new_names
    df.columns = [col.replace('texture_', '', 1) for col in df.columns]

    print("The are no duplicated column names:", len(list(df.columns)) == len(set(list(df.columns))))

    
def merge_datasets(df1, df2, suffixes=[]):
    """
    Merge two datasets on a set of metedata columns.
    """
    common_cols = ['label', 'image', 'object', 'stiffness', 'combination', 'well', 'site']
    if len(suffixes)==2:
        return pd.merge(df1, df2, how='outer', on=common_cols,  suffixes=suffixes)
    elif len(suffixes)==1:
        new_names=[]
        for col in df2.columns:
            if col in common_cols:
                new_names.append(col)
            else:
                new_names.append(col + suffixes[0])
        df2.columns = new_names
        return pd.merge(df1, df2, how='outer', on=common_cols)
    else:
        return pd.merge(df1, df2, how='outer', on=common_cols)


def move_column(df, column_name, loc):
    """
    Move a columns in front of the dataframe.
    """
    columns = df.columns.tolist()
    columns.insert(loc, columns.pop(columns.index(column_name)))
    return df.reindex(columns=columns, copy=False) 

def merge_neighbors(df, df_n):
    cols = ['ImageNumber', 'Location_Center_X', 'Location_Center_Y']
    
    # Round centre locations
    df_m = df[cols].copy()
    loc_cols = ['Location_Center_X', 'Location_Center_Y']
    df_m[loc_cols] = df_m[loc_cols].round()
    df_n[loc_cols] = df_n[loc_cols].round()
    
    # Merge dataframes
    neighb = pd.merge(df_m, df_n, how='inner', on=cols)
    
    # Assign values to the original dataframe
    df.loc[:, 'Neighbors_AngleBetweenNeighbors_3' : 'Number_Object_Number'] = \
    neighb.loc[:, 'Neighbors_AngleBetweenNeighbors_3' : 'Number_Object_Number']
    
    # Delete duplicated columns
    distances = [col.split('_')[2] for col in neighb.columns if 'Neighbors_NumberOfNeighbors' in col]

    if len(distances) > 1:
        dupl_cols = ['Neighbors_AngleBetweenNeighbors_', 
                     'Neighbors_FirstClosestDistance_', 
                     'Neighbors_FirstClosestObjectNumber_', 
                     'Neighbors_SecondClosestDistance_',
                     'Neighbors_SecondClosestObjectNumber_']
        dupl_cols = [col + distances[1] for col in dupl_cols]
        df.drop(dupl_cols, axis=1, inplace=True)


def import_cell_data(data_path=DATA_PATH, suffix='', cytoplasm=False, biomarkers=False):
    """ 
    Import all the data and then 
    call functions to parse metadata, 
    rename and rearrange columns and merge datasets.
    """
    cells = load_data(filename=suffix + 'Cells.csv')
    neighbors = load_data(filename=suffix + 'Neighbours.csv')
    nuclei = load_data(filename=suffix + 'Nuclei.csv')
    info = load_data(filename=suffix + 'Image.csv')
    
    # Check that dataframes contain the correct number of cells 
    n_cells = info.Count_Cells.sum()
    print('Morphology was measured for {} cells.\n'.format(n_cells))
    
    if (cells.shape[0] == n_cells) and (nuclei.shape[0] == n_cells):
        print('The numbers of cells and nuclei correspond to each other.\n')
    else:
        print('Found {} cells and {} nuclei'.format(cells.shape[0], nuclei.shape[0]))
        
    # Merge neighbours
    merge_neighbors(cells, neighbors)
    
    # Parse and clean metadata
    parse_metadata(cells)
    parse_metadata(nuclei)

    # Rename columns
    rename_columns(cells)
    rename_columns(nuclei)

    # Merge two datasets
    measurements = merge_datasets(cells, nuclei, suffixes=['_cell', '_nucl'])   
    
    if cytoplasm == True:
        cytoplasm = load_data(filename=suffix + 'Cytoplasm.csv') 
        print('Cytoplasm measurements were taken for {} cells.\n'.format(cytoplasm.shape[0]))
                  
        # Parse and clean metadata       
        parse_metadata(cytoplasm)
        
        # Rename columns
        rename_columns(cytoplasm)
        
        # Merge with the main dataset
        measurements = merge_datasets(measurements, cytoplasm, suffixes=['_cyto']) 
        
    if biomarkers:
        #biomarkers = load_data(filename=suffix + 'Biomarkers_Cells.csv')
        print('Read biomarkers normalised by to min gain')
        biomarkers = load_data(filename='Biomarkers_Cells.csv')
        print('Biomarkers were measured for {} cells.\n'.format(biomarkers.shape[0]))
        
        # Parse and clean metadata
        parse_metadata(biomarkers)
        
        # Rename columns
        rename_columns(biomarkers)
        
        # Merge with the main dataset
        measurements = merge_datasets(measurements, biomarkers) 
    
    # Move "label" column to front
    measurements = move_column(measurements, 'label', 0)
    print("\nFull dataset has shape:", measurements.shape)
    
    return measurements
    
def split_dataset(df, channels=None):
    """
    Split a dataset in two by channel name.
    """
    subsets = []
    
    for selected in channels:
        subset = df.copy()
        
        # Channels except for the selected
        channels_to_drop = list(set(channels) - set([selected]))
        
        for channel in channels_to_drop:
            # Drop the columns with specified channel
            cols = [col for col in df.columns if channel in col]
            subset.drop(cols, axis=1, inplace=True)
            
        # Rename channel suffix in the new dataset
        subset.columns = [col.replace('_' + selected, '', 1) for col in subset.columns]
        subsets.append(subset)
        
    return subsets
    

def dist(df):
    """
    Calculate Euclidean distance on a dataframe.
    Input columns are arranged as x0, x1, y0, y1.
    """
    return np.sqrt((df.iloc[:,0] - df.iloc[:,2])**2 + (df.iloc[:,1] - df.iloc[:,3])**2)


def transform_location(df):
    """
    Calculate the following distances and drop columns with location measurements.
    * centerX, centerY are coordinates of the fartherst point from any edge 
    (calculated using scipy.ndimage.center_of_mass);
    * loc_centerX, loc_centerY are average coordinates for the binary image
    (calculated using scipy.ndimage.mean);
    * fartherstpoint is a distance between the two points
    relative to cell and nuclear boundaries, respectively;
    * loc_maxintensityX, loc_maxintensityY are coordinates of the pixel 
    with the maximum intensity within the object;
    * maxintdisplacement is a distance between this pixel and loc_center;
    * nucleusshift is a distances between centres of mass of a cell and its nucleus.
    """

    # Drop duplicate columns with location
    df.drop(['loc_centerX', 'loc_centerY'], axis=1, inplace=True, errors='ignore')

    # Drop "centermassintensity" columns
    drop_cols = [col for col in df.columns if 'centermassintensity' in col]
    df.drop(drop_cols, axis=1, inplace=True)
   
    # Calculate distances between centres of a binary image
    df['fartherstpoint_cell'] = dist(df.loc[:, ['centerX_cell', 
                                                'centerY_cell',
                                                'loc_centerX_cell', 
                                                'loc_centerY_cell']])
    df['fartherstpoint_nucl'] = dist(df.loc[:, ['centerX_nucl', 
                                                'centerY_nucl',
                                                'loc_centerX_nucl', 
                                                'loc_centerY_nucl']])
    df['nucleusshift'] = dist(df.loc[:, ['centerX_cell', 
                                         'centerY_cell', 
                                         'centerX_nucl', 
                                         'centerY_nucl']])
    
    # Calculate max intensity displacement
    suffix = ['_'.join(col.split('_')[2:]) for col in df.columns if 'loc_maxintensity' in col]
    for s in set(suffix):
        maxint_cols = [col for col in df.columns if 'loc_maxintensity' in col and s in col]
        if 'dapi' in s or 'nucl' in s:
            cols = ['loc_centerX_nucl','loc_centerY_nucl']
            cols.extend(maxint_cols)
        else:
            cols = ['loc_centerX_cell','loc_centerY_cell']
            cols.extend(maxint_cols)

        new_col = 'maxintdisplacement_' + s
        df[new_col] = dist(df.loc[:, cols])
    
    # All location measurements are in absolute coordinates and should be dropped
    drop_cols = [col for col in df.columns if 'loc' in col]
    df.drop(drop_cols, axis=1, inplace=True)
    
    drop_cols = [col for col in df.columns if 'center' in col]
    df.drop(drop_cols, axis=1, inplace=True)
    
    # Move mass displacement columns to the end
    mass_cols = [col for col in df.columns if 'mass' in col]
    for col in mass_cols:
        df = move_column(df, col, df.columns.size)
       
    return df
    
    
def clean_data(df):
    """
    Clean the dataframe by dropping variable with zero variance, 
    uninformative/duplicated columns and location measurements.
    """
    # Check if there are any missing values
    assert df.isnull().sum().sum() == 0
    # measurements[measurements.isnull().sum(axis=1) > 0]
    
    print("Initial shape is:", df.shape)
    
    # Calculate summary statistics and drop features with zero variance
    stats = df.describe()
    zerovar_cols = stats.columns[stats.loc['std', :] == 0]
    print("Features with zero variance:\n", zerovar_cols)
    df.drop(zerovar_cols, axis=1, inplace=True)
    
    # Drop columns with object numbers
    numbers_cols = [col for col in df.columns if 'object' in col and 'number' in col]
    df.drop(numbers_cols, axis=1, inplace=True)
    
    # Drop columns with parent numbers
    parent_cols = [col for col in df.columns if 'parent' in col or 'children' in col]
    df.drop(parent_cols, axis=1, inplace=True)
    
    # Transform location measurements
    df = transform_location(df)
    
    # Transform orientation angle from [-pi/2, pi/2] to [0, pi]
    angle_cols = [col for col in df.columns if 'orientation' in col]
    df[angle_cols] += 90
    
    print("\nAfter cleaning the dataset has {} rows and {} columns.\n".format(df.shape[0], df.shape[1]))
    return df


def select_features(df, filename='selected_columns.txt'):
    """
    Load the list of manually selected columns
    and return a copy of the dataset containing only
    those columns
    """
    with open(filename, 'r') as file:
        selected_cols = [line.rstrip('\n') for line in file]
    
    df_fs = pd.concat([df.loc[:, 'label' : 'well'], df[selected_cols]], axis=1)
    
    return df_fs
    
    
def undersample(df, n_samples):
    """
    Perform undersampling of majority classes
    by randomly selecting n_samples cells 
    for each stiffness level.
    """
    df_under = pd.DataFrame(columns=df.columns)

    for s in df.stiffness.unique():
        if (s == "8.0") or (s == "32.0") :
            df_under = pd.concat([df_under, df[df.stiffness == s]], axis=0)
        else:
            df_under = pd.concat([df_under, df[df.stiffness == s].sample(n_samples)], axis=0)
        
    print("Undersampling. The balanced dataset has shape", df_under.shape)
    return df_under.reset_index(drop=True)


def smote(X, y, as_df=True):   
    """
    Synthesise new observations to have equal 
    number of cells for each stiffness value.
    """
    smote = SMOTE()
    X_sm, y_sm = smote.fit_sample(X, y)
    print("\nAfter synthesing new observations the balanced dataset has {} rows and {} columns.\n"
          .format(X_sm.shape[0], X_sm.shape[1]))
    
    if as_df:
        df_smote = pd.concat([pd.DataFrame(X_sm, columns=X.columns), 
                              as_stiff_type(pd.DataFrame(y_sm, columns=['stiffness']))],
                             axis=1).sort_values(by='stiffness')
        return df_smote
    else:
        return X_sm, y_sm
    
def cv_ratio(df, col1='ctcf_ker', col2='ctcf_vim'):
    df['cvratio'] = df[col1] / df[col2]
    df['log_cvratio'] = np.log(df.cvratio)
    return df

def nc_ratio(df):
    df['ncr'] = df.area_nucl / df.area_cyto
    return df

def phenotype(df, use, cols=('meanintensity_ker', 'meanintensity_vim')):
    if use=='log_ratio':
        df['log_ratio'] = np.log(df[cols[0]]/df[cols[1]])
        q1 = df.log_ratio.quantile(0.33)
        q3 = df.log_ratio.quantile(0.66)
        df['region'] = pd.cut(df.log_ratio, 
                             bins=(df.log_ratio.min(), q1, q3, df.log_ratio.max()), 
                             labels=["low", "med", "high"], include_lowest=True)
    elif use=='log_biom':
        df['log_ker'] = np.log(df[cols[0]])
        df['log_vim'] = np.log(df[cols[1]])
        df['region'] = "low"
        df.loc[(df.log_ker < df.log_ker.median()) &
                         (df.log_vim > df.log_vim.median()), 'region'] = "high vim"
        df.loc[(df.log_ker > df.log_ker.median()) &
                         (df.log_vim < df.log_vim.median()), 'region'] = "high ker"
        df.loc[(df.log_ker > df.log_ker.median()) &
                         (df.log_vim > df.log_vim.median()), 'region'] = "high"
        