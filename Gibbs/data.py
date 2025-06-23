import pickle  
import mygene
from gseapy import read_gmt
import anndata as ad
import numpy as np
import pandas as pd
import os

from gibbs import *



cytoseeds_csv_path = "/labs/Aguiar/SSPA_BRAY/BRay/BRAY_FileTransfer/Seed genes/CYTOBEAM_Cytokines_KEGGPATHWAY_addedMif.csv"
CYTOSEEDS_df = pd.read_csv(cytoseeds_csv_path)
CYTOSEEDS = CYTOSEEDS_df['V4'].tolist() #173

mg = mygene.MyGeneInfo()


def save_cache(data, cache_file):
    """Save data to a cache file using pickle."""
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved cached data to {cache_file}")

def load_cache(cache_file):
    """Load data from a cache file if it exists."""
    if os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return None


def convert_pathways_to_ensembl(pathways, cache_file="/labs/Aguiar/SSPA_BRAY/BRay/pathways_ensembl_cache.pkl"):
    
    # Try to load from cache first
    cached_pathways = load_cache(cache_file)
    if cached_pathways is not None:

        filtered_pathways = {}
        excluded_keywords = ["ADME", "DRUG", "MISCELLANEOUS", "EMT"]

        total_pathways = len(cached_pathways)
        reactome_count = sum(1 for k in cached_pathways if k.startswith('REACTOME'))
        
        for k, v in cached_pathways.items():
            if k.startswith('REACTOME'):
                if not any(keyword in k.upper() for keyword in excluded_keywords):
                    filtered_pathways[k] = v
        
        print(f"Original pathways count: {total_pathways}")
        print(f"Reactome pathways count: {reactome_count}")
        print(f"Filtered pathways count (REACTOME only, excluding keywords): {len(filtered_pathways)}")
        
        # Define specific pathways to exclude
        specific_exclude = [
            "REACTOME_GENERIC_TRANSCRIPTION_PATHWAY",
            "REACTOME_ADAPTIVE_IMMUNE_SYSTEM",
            "REACTOME_INNATE_IMMUNE_SYSTEM",
            "REACTOME_IMMUNE_SYSTEM",
            "REACTOME_METABOLISM"
        ]
        
        # Remove specific pathways
        for pathway in specific_exclude:
            if pathway in filtered_pathways:
                filtered_pathways.pop(pathway)
            
        excluded_count = reactome_count - len(filtered_pathways)
        print(f"Reactome pathways excluded due to keywords: {excluded_count}")
        cached_pathways = filtered_pathways
        return cached_pathways
    
    print("Cache not found. Converting pathways to Ensembl IDs...")
    mg = mygene.MyGeneInfo()
    unique_genes = set()
    for genes in pathways.values():
        unique_genes.update(genes)
    gene_list = list(unique_genes)
    print(f"Number of unique genes for conversion: {len(gene_list)}")
    
    mapping = {}
    batch_size = 100  # processing in batches for memory efficiency
    for i in range(0, len(gene_list), batch_size):
        batch = gene_list[i:i+batch_size]
        query_results = mg.querymany(batch, scopes='symbol', fields='ensembl.gene', species='mouse', returnall=False)
        for hit in query_results:
            if 'ensembl' in hit:
                if isinstance(hit['ensembl'], list):
                    mapping[hit['query']] = hit['ensembl'][0]['gene']
                else:
                    mapping[hit['query']] = hit['ensembl']['gene']
            else:
                mapping[hit['query']] = hit['query']  # keep original if no conversion found
    
    new_pathways = {}
    for pathway, genes in pathways.items():
        new_pathways[pathway] = [mapping.get(g, g) for g in genes]
    
    # Save to cache for future use
    save_cache(new_pathways, cache_file)
    
    return new_pathways

def batch_query(genes, batch_size=100):
    results = []
    for i in range(0, len(genes), batch_size):
        batch = genes[i:i+batch_size]
        results.extend(mg.querymany(batch, scopes='symbol', fields='ensembl.gene', species='mouse'))
    return results

# Define a cache file for CYTOSEED conversions
cytoseed_cache_file = "/labs/Aguiar/SSPA_BRAY/BRay/cytoseed_ensembl_cache.pkl"

# Try to load CYTOSEED mappings from cache
symbol_to_ensembl_asg = load_cache(cytoseed_cache_file)

if symbol_to_ensembl_asg is None:
    query_results = batch_query(CYTOSEEDS, batch_size=100)
    
    symbol_to_ensembl_asg = {}
    for entry in query_results:
        if 'ensembl' in entry and 'gene' in entry['ensembl']:
            if isinstance(entry['ensembl'], list):
                symbol_to_ensembl_asg[entry['query']] = entry['ensembl'][0]['gene']
            else:
                symbol_to_ensembl_asg[entry['query']] = entry['ensembl']['gene']
        else:
            symbol_to_ensembl_asg[entry['query']] = None 
    
    # Save to cache for future use
    save_cache(symbol_to_ensembl_asg, cytoseed_cache_file)

CYTOSEED_ensembl = [symbol_to_ensembl_asg.get(gene) for gene in CYTOSEEDS if symbol_to_ensembl_asg.get(gene)]
print(f"CYTOSEED_ensembl length: {len(CYTOSEED_ensembl)}")

pathways_path = "/labs/Aguiar/SSPA_BRAY/BRay/BRAY_FileTransfer/m2.cp.v2024.1.Mm.symbols.gmt"
pathways = read_gmt(pathways_path)  # 1730 pathways
print(f"Number of pathways: {len(pathways)}")

pathways = convert_pathways_to_ensembl(pathways)  

# nap_file_path_raw = "/labs/Aguiar/SSPA_BRAY/BRay/BRAY_AJM2/2_Data/2_SingleCellData/1_GSE139565_NaiveAndPlasma/GEX_NAP_filt_raw_modelingonly_2024-02-05.csv"
# nap_metadata_path = "/labs/Aguiar/SSPA_BRAY/BRay/BRAY_AJM2/2_Data/2_SingleCellData/1_GSE139565_NaiveAndPlasma/meta_NAP_unfilt_fullData_2024-02-05.csv"

# Updated to use RDS file instead of CSV
ajm_file_path = "/labs/Aguiar/SSPA_BRAY/BRay/BRAY_AJM2/2_Data/2_SingleCellData/2_AJM_Parse_Timecourse/GEX_TC_LPSonly_Bcellonly_filt_raw_2024-02-05.rds"
ajm_metadata_path = "/labs/Aguiar/SSPA_BRAY/BRay/BRAY_AJM2/2_Data/2_SingleCellData/2_AJM_Parse_Timecourse/meta_TC_LPSonly_Bcellonly_filt_raw_2024-02-05.csv"


def prepare_ajm_dataset(cache_file="/labs/Aguiar/SSPA_BRAY/BRay/ajm_dataset_cache.h5ad"):
    print("Loading AJM dataset...")
    
    # Import required modules at the function's top level
    import os
    import subprocess
    import scipy.sparse as sp
    
    # Try to load from cache first
    if os.path.exists(cache_file):
        print(f"Loading AnnData object from cache file: {cache_file}")
        try:
            adata = ad.read_h5ad(cache_file)
            print(f"Successfully loaded cached AnnData object with shape: {adata.shape}")
            
            # Extract the dataset splits
            ajm_ap_samples = adata[adata.obs['dataset'] == 'ap']
            ajm_cyto_samples = adata[adata.obs['dataset'] == 'cyto']

            # Normalize and log-transform
            log_normalize_adata(ajm_ap_samples)
            log_normalize_adata(ajm_cyto_samples)
            
            print("AJM AP Samples distribution:")
            print(ajm_ap_samples.obs['ap'].value_counts())

            print("AJM CYTO Samples distribution:")
            print(ajm_cyto_samples.obs['cyto'].value_counts())
            
            return ajm_ap_samples, ajm_cyto_samples
        except Exception as e:
            print(f"Error loading cached AnnData: {e}")
            print("Proceeding with RDS conversion...")
    else:
        print(f"Cache file {cache_file} not found, converting RDS file...")
    
    # Run the R script to convert the RDS file to CSV files
    print("Converting RDS to anndata format using R...")
    r_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rds_to_anndata.R")
    
    try:
        result = subprocess.run(["Rscript", r_script_path], 
                               capture_output=True, 
                               text=True, 
                               check=True)
        print("R conversion output:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("R conversion error:")
        print(e.stdout)
        print(e.stderr)
        raise RuntimeError("Failed to convert RDS file to CSV")
    
    
    # Load the sparse matrix data
    print("Loading sparse matrix data...")
    sparse_data = pd.read_csv("matrix_sparse.csv")
    
    # Load row and column names
    row_names = pd.read_csv("matrix_rownames.csv")["row_names"].tolist()  # These are gene names
    col_names = pd.read_csv("matrix_colnames.csv")["col_names"].tolist()  # These are cell names
    
    # Load matrix dimensions (if available)
    matrix_dims = None
    if os.path.exists("matrix_dims.csv"):
        matrix_dims = pd.read_csv("matrix_dims.csv")
        nrows = matrix_dims["rows"].iloc[0]
        ncols = matrix_dims["cols"].iloc[0]
        print(f"Matrix dimensions from file: {nrows} x {ncols}")
        # Verify that dimensions match the length of row and column names
        if nrows != len(row_names) or ncols != len(col_names):
            print(f"WARNING: Dimension mismatch! Row names: {len(row_names)}, Column names: {len(col_names)}")
    else:
        print("Matrix dimensions file not found, using length of row and column names")
        nrows = len(row_names)
        ncols = len(col_names)
    
    print(f"Sparse data shape: {sparse_data.shape}")
    print(f"Number of genes (rows in original matrix): {len(row_names)}")
    print(f"Number of cells (columns in original matrix): {len(col_names)}")
    
    # Create sparse matrix from the CSV data
    # The sparse data has format: row (gene), col (cell), value
    # Indices should already be 0-based from R script
    row_indices = sparse_data["row"].values  # Gene indices
    col_indices = sparse_data["col"].values  # Cell indices
    values = sparse_data["value"].values
    
    # Debug information
    print(f"Row indices range: {row_indices.min()} to {row_indices.max()}")
    print(f"Column indices range: {col_indices.min()} to {col_indices.max()}")
    
    # In AnnData, rows are cells (observations) and columns are genes (variables)
    # So we need to transpose the matrix from our R output
    print(f"Creating sparse matrix with shape: ({nrows}, {ncols}) and transposing to match AnnData format")
    
    # Create a sparse COO matrix with original shape
    sparse_matrix = sp.coo_matrix((values, (row_indices, col_indices)), 
                                 shape=(nrows, ncols))
    
    # Transpose the matrix to have cells as rows and genes as columns
    sparse_matrix = sparse_matrix.transpose().tocsr()
    
    # Now the shape is (ncols, nrows) - (cells, genes)
    print(f"Transposed matrix shape: {sparse_matrix.shape}")
    
    
    # Create AnnData object with transposed matrix where:
    # - Rows (observations) are cells
    # - Columns (variables) are genes
    ajm_adata = ad.AnnData(X=sparse_matrix)
    
    # In AnnData:
    # - obs_names (rows) should be cell names
    # - var_names (columns) should be gene names
    ajm_adata.obs_names = col_names  # Cell names as observation names
    ajm_adata.var_names = row_names  # Gene names as variable names
    
    
    print(f"AnnData object created with shape: {ajm_adata.shape}")
    
    # Load metadata separately
    ajm_features = pd.read_csv(ajm_metadata_path, index_col=0)
    
    print(f"AJM features shape: {ajm_features.shape}")
    
    # Create label mappings
    ajm_label_mapping = {
        'TC-0hr':       {'ap':0,'cyto':0,'ig':-1},
        'TC-LPS-3hr':   {'ap':-1,'cyto':0,'ig':-1},
        'TC-LPS-6hr':   {'ap':1,'cyto':-1,'ig':-1},
        'TC-LPS-24hr':  {'ap':1,'cyto':1,'ig':-1},
        'TC-LPS-48hr':  {'ap':-1,'cyto':-1,'ig':-1},
        'TC-LPS-72hr':  {'ap':-1,'cyto':-1,'ig':-1},
    }
    
    # Initialize label columns in metadata
    for col in ['ap','cyto','ig']:
        ajm_features[col] = None
    
    # Apply mapping based on sample values
    for sample_value, labels in ajm_label_mapping.items():
        mask = ajm_features['sample'] == sample_value
        for col in labels:
            ajm_features.loc[mask,col] = labels[col]

    # Set cell_id as index if available
    if 'cell_id' in ajm_features.columns:
        ajm_features.set_index('cell_id', inplace=True)
    
    # Ensure cell IDs match between anndata and metadata
    common_cells_ajm = ajm_adata.obs_names.intersection(ajm_features.index)
    print(f"Number of common cells: {len(common_cells_ajm)}")
    
    # Subset to common cells
    ajm_adata = ajm_adata[common_cells_ajm]
    ajm_features = ajm_features.loc[common_cells_ajm]
    
    # Add metadata to AnnData object
    for col in ajm_features.columns:
        ajm_adata.obs[col] = ajm_features[col].values
    
    # Ensure gene symbols are available
    if 'gene_symbols' not in ajm_adata.var:
        ajm_adata.var['gene_symbols'] = ajm_adata.var_names
    
    # Create subset AnnData objects for different analyses
    ajm_ap_samples = ajm_adata[ajm_adata.obs['ap'].isin([0,1])]
    ajm_cyto_samples = ajm_adata[ajm_adata.obs['cyto'].isin([0,1])]

    # Normalize and log-transform
    log_normalize_adata(ajm_ap_samples)
    log_normalize_adata(ajm_cyto_samples)

    # Add dataset identifier to help with cache loading
    ajm_ap_samples.obs['dataset'] = 'ap'
    ajm_cyto_samples.obs['dataset'] = 'cyto'
    
    # Combine both datasets for caching
    combined_adata = ad.concat(
        [ajm_ap_samples, ajm_cyto_samples],
        join='outer',
        merge='same'
    )
    
    # Make sure all values in .obs are serializable (convert to string if needed)
    for col in combined_adata.obs.columns:
        if combined_adata.obs[col].dtype == 'object':
            combined_adata.obs[col] = combined_adata.obs[col].astype(str)
        
        # Convert None/NaN values to strings to avoid serialization issues
        combined_adata.obs[col] = combined_adata.obs[col].fillna('NA')
    
    # Similarly ensure .var values are serializable
    for col in combined_adata.var.columns:
        if combined_adata.var[col].dtype == 'object':
            combined_adata.var[col] = combined_adata.var[col].astype(str)
        
        # Convert None/NaN values to strings
        combined_adata.var[col] = combined_adata.var[col].fillna('NA')
    
    # Save to cache file
    print(f"Saving AnnData object to cache file: {cache_file}")
    try:
        combined_adata.write_h5ad(cache_file)
        print(f"Successfully saved AnnData to cache")
    except TypeError as e:
        print(f"Error saving AnnData to H5AD: {e}")
        print("Will proceed without caching.")
    
    print("AJM AP Samples distribution:")
    print(ajm_ap_samples.obs['ap'].value_counts())

    print("AJM CYTO Samples distribution:")
    print(ajm_cyto_samples.obs['cyto'].value_counts())

    
    # Clean up the temporary CSV files
    temp_files = ["matrix_sparse.csv", "matrix_rownames.csv", "matrix_colnames.csv", "matrix_dims.csv"]
    for file in temp_files:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"Temporary file {file} removed")
            except Exception as e:
                print(f"Warning: Could not remove temporary file {file}: {e}")
    
    return ajm_ap_samples, ajm_cyto_samples

gene_annotation_path = "/labs/Aguiar/SSPA_BRAY/BRay/BRAY_FileTransfer/ENS_mouse_geneannotation.csv"
gene_annotation = pd.read_csv(gene_annotation_path)
gene_annotation = gene_annotation.set_index('GeneID')

def filter_protein_coding_genes(adata, gene_annotation):
    protein_coding_genes = gene_annotation[gene_annotation['Genetype'] == 'protein_coding'].index
    
    common_genes = np.intersect1d(adata.var_names, protein_coding_genes)
    
    print(f"Total genes: {adata.n_vars}")
    print(f"Protein-coding genes found: {len(common_genes)}")
    
    adata_filtered = adata[:, common_genes].copy()

    return adata_filtered


def log_normalize_adata(adata, scale_factor=1e4):
    """Library size normalize and log-transform the counts in an AnnData object."""
    import scipy.sparse as sp

    X = adata.X.astype(float)

    if sp.issparse(X):
        library_sizes = np.array(X.sum(axis=1)).reshape(-1, 1)
        library_sizes[library_sizes == 0] = 1
        X = X.multiply(1 / library_sizes)
        X = X.multiply(scale_factor)
        X.data = np.log1p(X.data)
    else:
        library_sizes = X.sum(axis=1, keepdims=True)
        library_sizes[library_sizes == 0] = 1
        X = (X / library_sizes) * scale_factor
        X = np.log1p(X)

    adata.X = X
    return adata
