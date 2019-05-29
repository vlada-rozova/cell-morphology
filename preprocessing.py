import os
import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype
from imblearn.over_sampling import SMOTE

DATA_PATH = '../cell-profiler/measurements'


def load_data(filename, data_path=DATA_PATH):
    """ 
    Load a csv file.
    """
    csv_path = os.path.join(data_path, filename)
    return pd.read_csv(csv_path)
    
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
    
    # Change types and create a unique label 
    df = create_label(df)

def as_stiff_type(x):
    """
    Convert stiffness values to a custom categorical type.
    """
    # Create a categorical data type for stiffness
    stiff_type = CategoricalDtype(categories=['0.2', '0.5', '2.0', '8.0', '16.0', '32.0', '64.0'], ordered=True)
    return x.astype(stiff_type)
    
    
def create_label(df, per_cell=True):
    """
    Create a unique label for each observation.
    Labels are created as follows:
    stiffness-combination-well-site[-object]
    """
    # Convert to non-numeric values
    columns = ['image', 'object', 'stiffness', 'combination', 'well', 'site']
    existing_columns = [col for col in columns if col in df.columns]
    df[existing_columns] = df[existing_columns].astype(str)
   
    df.stiffness = as_stiff_type(df.stiffness)
    
    # Create a unique label for each cell
    if per_cell:
        df['label'] = df[['stiffness', 'combination', 'well', 'site', 'object']].apply(lambda x: '-'.join(x), axis=1)
    else:
        df['label'] = df[['stiffness', 'combination', 'well', 'site']].apply(lambda x: '-'.join(x), axis=1)
    
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

    # Neighbours and Zernike
    new_names = []
    for col in df.columns:
        if 'neighbors_' in col:
            new_names.append(col.split('_')[1])
        else:
            new_names.append(col)
    df.columns = new_names

    # Texture
    df.columns = [col.replace('_3', '', 1) for col in df.columns]
    df.columns = [col.replace('texture_', '', 1) for col in df.columns]

    print("The are no repeated column names:", len(list(df.columns)) == len(set(list(df.columns))))

    
def merge_datasets(df1, df2):
    """
    Merge two datasets on a set of metedata columns.
    """
    common_columns = ['label', 'image', 'object', 'stiffness', 'combination', 'well', 'site']
    return pd.merge(df1, df2, how='outer', on=common_columns,  suffixes=('_cell', '_nucl'))


def move_column(df, column_name):
    """
    Move a columns in front of the dataframe.
    """
    columns = list(df.columns)
    columns.insert(0, columns.pop(columns.index(column_name)))
    return df.reindex(columns=columns, copy=False) 


def cell_data(data_path=DATA_PATH, suffix=None, cytoplasm=False, biomarkers=False):
    """ 
    Load all the data and then 
    call functions to parse metadata, 
    rename and rearrange columns and merge datasets.
    """
    if biomarkers:
        biomarkers = load_data(filename=suffix + 'Biomarkers_Cells.csv')
        
        # Parse and clean metadata
        parse_metadata(biomarkers)
        
        # Rename columns
        rename_columns(biomarkers)
        
        # Move label column to front
        biomarkers = move_column(biomarkers, 'label')
        
        return biomarkers
    else:
        cells = load_data(filename=suffix + 'Cells.csv')
        nuclei = load_data(filename=suffix + 'Nuclei.csv')
        info = load_data(filename=suffix + 'Image.csv')

        # Check that dataframes contain the correct number of cells 
        n_cells = info.Count_Cells.sum()
        print('Total number of cells processed: {}\n'.format(n_cells))
        if (cells.shape[0] == n_cells) and (nuclei.shape[0] == n_cells):
            print('The numbers of cells and nuclei correspond to each other')
        else:
            print('Found {} cells and {} nuclei'.format(cells.shape[0], nuclei.shape[0]))
        

        # Parse and clean metadata
        parse_metadata(cells)
        parse_metadata(nuclei)

        # Rename columns
        rename_columns(cells)
        rename_columns(nuclei)
        
        print("Membrane features:", cells.shape)
        print("Chromatin features:", nuclei.shape)

        # Merge two datasets
        measurements = merge_datasets(cells, nuclei)

        # Move label column to front
        measurements = move_column(measurements, 'label')
        print("Full dataset has shape:", measurements.shape)

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


def transform_location(df, intensity=True):
    """
    Calculate the following distances and drop columns with location measurements.
    * centerX, centerY are coordinates of the fartherst point from any edge 
    (calculated using scipy.ndimage.center_of_mass);
    * loc_centerX, loc_centerY are average coordinates for the binary image
    (calculated using sccipy.ndimage.mean);
    * fartherstpoint is a distance between the two points
    relative to cell and nuclear boundaries, respectively;
    * loc_maxintensityX, loc_maxintensityY are coordinates of the pixel 
    with the maximum intensity within the object;
    * maxintdisplacement is a distance between this pixel and loc_center;
    * nucleusshift is a distances between centres of mass of a cell and its nucleus.
    """
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
    if intensity:
            df['maxintdisplacement_wga_cell'] = dist(df.loc[:, ['loc_centerX_cell',
                                                                'loc_centerY_cell', 
                                                                'loc_maxintensityX_wga_cell', 
                                                                'loc_maxintensityY_wga_cell']])
            df['maxintdisplacement_wga_nucl'] = dist(df.loc[:, ['loc_centerX_nucl',
                                                                'loc_centerY_nucl', 
                                                                'loc_maxintensityX_wga_nucl', 
                                                                'loc_maxintensityY_wga_nucl']])
            df['maxintdisplacement_dapi'] = dist(df.loc[:, ['loc_centerX_nucl',
                                                            'loc_centerY_nucl',
                                                            'loc_maxintensityX_dapi', 
                                                            'loc_maxintensityY_dapi']])
    
    # All location measurements are in absolute coordinates and should be dropped
    loc_cols = [col for col in df.columns if 'loc' in col]
    df.drop(loc_cols, axis=1, inplace=True)
    loc_cols = [col for col in df.columns if 'center' in col]
    df.drop(loc_cols, axis=1, inplace=True)
    
    
def clean_data(df, intensity=True):
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
    numbers_cols = ['number_object_number_cell', 'number_object_number_nucl', 
                    'firstclosestobjectnumber_cell', 'firstclosestobjectnumber_nucl', 
                    'secondclosestobjectnumber_cell', 'secondclosestobjectnumber_nucl']
    df.drop(numbers_cols, axis=1, inplace=True)
    #df.drop([col for col in dupl_cols if sum(df.object.astype(int) != df[col])==0], axis=1, inplace=True)
    
    # Transform location measurements
    transform_location(df, intensity)
    
    print("\nAfter cleaning the dataset has {} rows and {} columns.\n".format(df.shape[0], df.shape[1]))
    df.dtypes.value_counts()
    return df


def drop_corr_features(df, intensity=False, texture=False, zernike=False):
    """
    Drop manually chosen features with Pearson's correlation coefficient greater or equal than 0.9.
    """
    df_fs = df.loc[:, 'label' : 'well']
    print("Drop selected features with high Pearson's correlation coefficient.\n")
    
    # Cell Shape
    cell_shape = df.loc[:, 'area_cell' : 'solidity_cell']
    cell_shape.drop(['majoraxislength_cell', 'maxferetdiameter_cell', 
                     'maximumradius_cell', 'medianradius_cell', 
                     'minferetdiameter_cell'], 
                    axis=1, inplace=True)
    print("Selected {} cell shape features.".format(cell_shape.shape[1]))
    df_fs = pd.concat([df_fs, cell_shape], axis=1)
    
    # Cell Zernike
    if zernike:
        cell_zern = df.loc[:, 'zernike_0_0_cell' : 'zernike_9_9_cell']
        cell_zern.drop(['zernike_0_0_cell'], axis=1, inplace=True)
        print("Selected {} cell zernike features:".format(cell_zern.shape[1]))
        df_fs = pd.concat([df_fs, cell_zern], axis=1)
    
    # Cell intensity
    if intensity:
        cell_int = df.loc[:, 'integratedintensityedge_wga_cell' : 'upperquartileintensity_wga_cell'] 
        cell_int.drop(['medianintensity_wga_cell', 'upperquartileintensity_wga_cell', 
                       'maxintensityedge_wga_cell', 'minintensity_wga_cell'], 
                      axis=1, inplace=True)
        print("Selected {} cell intensity features.".format(cell_int.shape[1]))
        df_fs = pd.concat([df_fs, cell_int], axis=1)
    
    # Cell neighbours
    cell_neighb = df.loc[:, 'anglebetweenneighbors_cell' :'secondclosestdistance_cell']
    print("Selected {} cell neighbours features.".format(cell_neighb.shape[1]))
    df_fs = pd.concat([df_fs, cell_neighb], axis=1)
    
    # Cell texture
    if texture:
        cell_tex = df[['angularsecondmoment_wga_00_cell', 'contrast_wga_00_cell', 
                       'correlation_wga_01_cell', 'correlation_wga_02_cell',
                       'differencevariance_wga_00_cell', 'entropy_wga_00_cell', 
                       'infomeas1_wga_00_cell', 'infomeas1_wga_02_cell', 
                       'infomeas2_wga_00_cell', 'sumaverage_wga_00_cell', 
                       'variance_wga_00_cell']]
        print("Selected {} cell texture features.".format(cell_tex.shape[1]))
        df_fs = pd.concat([df_fs, cell_tex], axis=1)
    
    # Nuclear shape
    nucl_shape = df.loc[:, 'area_nucl' : 'solidity_nucl']
    nucl_shape.drop(['maxferetdiameter_nucl', 'maximumradius_nucl', 
                     'medianradius_nucl', 'minferetdiameter_nucl', 
                     'minoraxislength_nucl', 'perimeter_nucl'], 
                    axis=1, inplace=True)
    print("Selected {} nuclear shape features.".format(nucl_shape.shape[1]))
    df_fs = pd.concat([df_fs, nucl_shape], axis=1)
                          
    # Nuclear Zernike
    if zernike:
        nucl_zern = df.loc[:, 'zernike_0_0_nucl' : 'zernike_9_9_nucl']
        print("Selected {} nuclear zernike features:".format(nucl_zern.shape[1]))
        df_fs = pd.concat([df_fs, nucl_zern], axis=1)
    
    # Nuclear intensity
    if intensity:
        nucl_int = df.loc[:, 'integratedintensityedge_dapi' : 'upperquartileintensity_wga_nucl'] 
        nucl_int.drop(['lowerquartileintensity_dapi', 'medianintensity_dapi', 
                       'upperquartileintensity_dapi', 'minintensityedge_dapi', 
                       'minintensity_dapi', 
                       'lowerquartileintensity_wga_nucl', 'medianintensity_wga_nucl', 
                       'upperquartileintensity_wga_nucl', 'maxintensityedge_wga_nucl', 
                       'meanintensityedge_wga_nucl'],
                      axis=1, inplace=True)
        print("Selected {} nuclear intensity features.".format(nucl_int.shape[1]))
        df_fs = pd.concat([df_fs, nucl_int], axis=1)
    
    # Nuclear neighbours
    nucl_neighb = df.loc[:, 'anglebetweenneighbors_nucl' :'secondclosestdistance_nucl']
    nucl_neighb.drop(['percenttouching_nucl'], axis=1, inplace=True)
    print("Selected {} nuclear neighbours features.".format(nucl_neighb.shape[1]))
    df_fs = pd.concat([df_fs, nucl_neighb], axis=1)
    
    # Nuclear texture
    if texture: 
        nucl_tex = df[['angularsecondmoment_dapi_00',
                       'contrast_dapi_00', 'contrast_dapi_01', 
                       'contrast_dapi_02',
                       'correlation_dapi_00', 'correlation_dapi_01', 
                       'correlation_dapi_02', 'correlation_dapi_03', 
                       'differenceentropy_dapi_00', 'differencevariance_dapi_00',
                       'entropy_dapi_00', 'infomeas1_dapi_00', 
                       'infomeas2_dapi_00', 'sumaverage_dapi_00', 
                       'variance_dapi_00', 
                       'angularsecondmoment_wga_00_nucl', 
                       'contrast_wga_00_nucl', 
                       'correlation_wga_00_nucl', 'correlation_wga_01_nucl', 
                       'correlation_wga_02_nucl', 'correlation_wga_03_nucl', 
                       'differenceentropy_wga_00_nucl', 'differencevariance_wga_00_nucl',
                       'entropy_wga_00_nucl', 'infomeas1_wga_00_nucl', 
                       'infomeas2_wga_00_nucl', 'inversedifferencemoment_wga_00_nucl',
                       'sumaverage_wga_00_nucl', 'variance_wga_00_nucl']]
        print("Selected {} nuclear texture features.".format(nucl_tex.shape[1]))
        df_fs = pd.concat([df_fs, nucl_tex], axis=1)
    
    # Distance measurements
    dist = df.loc[:, 'fartherstpoint_cell' :'nucleusshift']
    print("Selected {} distance measurement features.".format(dist.shape[1]))
    df_fs = pd.concat([df_fs, dist], axis=1)
    
    # Intergroup correlation
    if intensity and texture:
        df_fs.drop(['sumaverage_wga_00_cell',
                    'variance_wga_00_cell', 
                    'sumaverage_dapi_00',
                    'sumaverage_wga_00_nucl', 
                    'variance_dapi_00', 
                    'variance_wga_00_nucl'], 
                   axis=1, inplace=True)
    
    print("\nAfter selecting features the dataset has {} rows and {} columns.\n".format(df_fs.shape[0], df_fs.shape[1]))
    return df_fs
    
    
def undersample(df):
    """
    Perform undersampling of majority classes
    by randomly selecting n_samples cells 
    for each stiffness level.
    """
    n_samples = 50
    df_under = pd.DataFrame(columns=df.columns)

    for s in df.stiffness.unique():
        if (s == "8.0") or (s == "32.0") :
            df_under = pd.concat([df_under, df[df.stiffness == s]], axis=0)
        else:
            df_under = pd.concat([df_under, df[df.stiffness == s].sample(n_samples)], axis=0)
        
    print("Undersampling. The balanced dataset has shape", df_under.shape)
    return df_under


def smote(X, y, as_df=True):   
    """
    Synthesise new observations to have equal 
    number of cells for each stiffness value.
    """
    smote = SMOTE()
    X_sm, y_sm = smote.fit_sample(X, y)
    print("\nAfter synthesising new observations the balanced dataset has {} rows and {} columns.\n"
          .format(X_sm.shape[0], X_sm.shape[1]))
    
    if as_df:
        df_smote = pd.concat([pd.DataFrame(X_sm, columns=X.columns), 
                              as_stiff_type(pd.DataFrame(y_sm, columns=['stiffness']))],
                             axis=1).sort_values(by='stiffness')
        return df_smote
    else:
        return X_sm, y_sm
    
        