import os
import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype
import seaborn as sns
from matplotlib import pyplot as plt  

DATA_PATH = '../cell-profiler/measurements'

def load_data(filename, data_path=DATA_PATH):
    csv_path = os.path.join(data_path, filename)
    return pd.read_csv(csv_path)
    
def parse_metadata(df):
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
    
    # Convert to non-numeric values
    columns = ['image', 'object', 'stiffness', 'combination', 'well', 'site']
    df[columns] = df[columns].astype(str)
    
    # Create a categorical data type for stiffness
    stiff_type = CategoricalDtype(categories=['0.2', '2.0', '16.0', '64.0'], ordered=True)
    df.stiffness = df.stiffness.astype(stiff_type)
    
    # Create a unique label for each cell
    df['label'] = df[['stiffness', 'combination', 'well', 'site', 'object']].apply(lambda x: '-'.join(x), axis=1)
    
    
def rename_columns(df):
    # Convert to lower case
    df.columns = [col.lower() for col in df.columns]

    # Rename channels
    df.columns = [col.replace('_origblue', '_dapi', 1) for col in df.columns]
    df.columns = [col.replace('_origgreen', '_wga', 1) for col in df.columns]

    # Coordinates in X
    df.columns = [col.replace('_x', 'X', 1) for col in df.columns]

    # Coordinates in Y
    df.columns = [col.replace('_y', 'Y', 1) for col in df.columns]

    # Coordinates in Z
    df.columns = [col.replace('_z', 'Z', 1) for col in df.columns]

    # Shape features
    df.columns = [col.replace('areashape_', '', 1) for col in df.columns]

    # Intensity features
    df.columns = [col.replace('intensity_', '', 1) for col in df.columns]

    # Location
    df.columns = [col.replace('location_', 'loc_', 1) for col in df.columns]

    # Neighbours
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
    common_columns = ['label', 'image', 'object', 'stiffness', 'combination', 'well', 'site']
    return pd.merge(df1, df2, how='left', on=common_columns,  suffixes=('_cell', '_nucl'))


def move_column(df, column_name):
    columns = list(df.columns)
    columns.insert(0, columns.pop(columns.index(column_name)))
    return df.reindex(columns=columns, copy=False) 


def cell_data(data_path=DATA_PATH, cytoplasm=False, biomarkers=False):
    """ 
    This function loads all the data,
    and then calls functions to parse metadata, 
    rename and rearrange columns and merge datasets.
    """
    cells = load_data(filename='Cells.csv')
    nuclei = load_data(filename='Nuclei.csv')
    info = load_data(filename='Image.csv')
    
    # Check that dataframes contain the correct number of cells 
    n_cells = info.Count_Cells.sum()
    print('Total number of cells processed: {}\n'.format(n_cells))
    print('Check the numbers of cells correspond: {}, {}\n'.format(cells.shape[0] == n_cells, nuclei.shape[0] == n_cells))
    
    # Parse and clean metadata
    parse_metadata(cells)
    parse_metadata(nuclei)
    
    # Rename columns
    rename_columns(cells)
    rename_columns(nuclei)
    
    # Merge two datasets
    measurements = merge_datasets(cells, nuclei)

    # Move label column to front
    measurements = move_column(measurements, 'label')
    
    return measurements

