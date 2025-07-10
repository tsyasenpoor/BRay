import os
import numpy as np
import pandas as pd
from scipy.stats import ranksums
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import scanpy as sc

# Set up environment for rpy2
os.environ['R_HOME'] = '/home/FCAM/tyasenpoor/miniconda3/lib/R'
os.environ['PATH'] = '/home/FCAM/tyasenpoor/miniconda3/lib/R/bin:' + os.environ.get('PATH', '')

# Import rpy2
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

# File paths
ajm_file_path_raw = "/labs/Aguiar/SSPA_BRAY/BRay/BRAY_AJM2/2_Data/2_SingleCellData/2_AJM_Parse_Timecourse/GEX_TC_LPSonly_Bcellonly_filt_raw_2024-02-05.rds"
ajm_file_path_norm = "/labs/Aguiar/SSPA_BRAY/BRay/BRAY_AJM2/2_Data/2_SingleCellData/2_AJM_Parse_Timecourse/GEX_TC_LPSonly_Bcellonly_filt_norm_2024-02-09.rds"
ajm_metadata_path = "/labs/Aguiar/SSPA_BRAY/BRay/BRAY_AJM2/2_Data/2_SingleCellData/2_AJM_Parse_Timecourse/meta_TC_LPSonly_Bcellonly_filt_raw_2024-02-05.csv"

# Method 1: Read RDS file using rpy2
print("Reading RDS file using rpy2...")
r = ro.r
data = r(f'readRDS("{ajm_file_path_raw}")')
print(f"RDS file read successfully!")
print(f"Object type: {type(data)}")
print(f"Object class: {r('class')(data)[0]}")

# Method 2: Convert to pandas DataFrame (if it's a matrix)
try:
    # Convert R matrix to pandas DataFrame
    pandas2ri.activate()
    df = pandas2ri.rpy2py(data)
    print(f"Converted to pandas DataFrame with shape: {df.shape}")
    print(f"DataFrame head:\n{df.head()}")
except Exception as e:
    print(f"Could not convert to DataFrame: {e}")

# Method 3: Extract matrix properties
try:
    # Get dimensions
    dims = r('dim')(data)
    print(f"Matrix dimensions: {dims[0]} x {dims[1]}")
    
    # Get row and column names
    rownames = r('rownames')(data)
    colnames = r('colnames')(data)
    print(f"Number of rows (genes): {len(rownames)}")
    print(f"Number of columns (cells): {len(colnames)}")
    
    # Get first few row and column names
    print(f"First 5 row names: {rownames[:5]}")
    print(f"First 5 column names: {colnames[:5]}")
    
except Exception as e:
    print(f"Could not extract matrix properties: {e}")

print("\nSuccess! You can now use rpy2 to read your R data files.") 