#!/usr/bin/env Rscript

# Set up personal library path to avoid permission issues
lib_path <- path.expand("~/R_libs_user")
dir.create(lib_path, showWarnings = FALSE, recursive = TRUE)
.libPaths(c(lib_path, .libPaths()))

# Enable better error messages
options(error = function() {
  traceback(3)
  quit(status = 1)
})

# Define file paths for both raw and normalized data
raw_file_path <- "/labs/Aguiar/SSPA_BRAY/BRay/BRAY_AJM2/2_Data/2_SingleCellData/2_AJM_Parse_Timecourse/GEX_TC_LPSonly_Bcellonly_filt_raw_2024-02-05.rds"
norm_file_path <- "/labs/Aguiar/SSPA_BRAY/BRay/BRAY_AJM2/2_Data/2_SingleCellData/2_AJM_Parse_Timecourse/GEX_TC_LPSonly_Bcellonly_filt_norm_2024-02-09.rds"

message("Raw RDS file: ", raw_file_path)
message("Normalized RDS file: ", norm_file_path)
message("R version: ", R.version.string)
message("Library paths: ", paste(.libPaths(), collapse=", "))

# Function to check package availability without installation attempts
check_and_load <- function(pkg_name) {
  if (requireNamespace(pkg_name, quietly = TRUE)) {
    message(paste("Loading package:", pkg_name))
    library(pkg_name, character.only = TRUE)
    return(TRUE)
  } else {
    message(paste("Package", pkg_name, "is not available. Skipping."))
    return(FALSE)
  }
}

# Load required Matrix package for sparse matrices
suppressMessages(library(Matrix))

# Function to process and save matrix
process_matrix <- function(matrix_obj, prefix) {
  message("Processing ", prefix, " matrix...")
  message("Object class: ", class(matrix_obj)[1])
  
  if (inherits(matrix_obj, "dgCMatrix") || inherits(matrix_obj, "dgTMatrix")) {
    # This is a sparse matrix, we can directly save it
    message(prefix, " object is a sparse matrix with dimensions: ", paste(dim(matrix_obj), collapse=" x "))
    
    # Get dimensions of the matrix
    nrows <- nrow(matrix_obj)
    ncols <- ncol(matrix_obj)
    message(prefix, " matrix dimensions: rows=", nrows, ", cols=", ncols)
    
    # Get indices and values
    sparse_summary <- summary(matrix_obj)
    
    # Adjust indices to be 0-based for Python
    row_indices <- sparse_summary[,1] - 1
    col_indices <- sparse_summary[,2] - 1
    values <- sparse_summary[,3]
    
    # Check index ranges
    message(prefix, " row index range: ", min(row_indices), " to ", max(row_indices), 
            " (should be 0 to ", nrows-1, ")")
    message(prefix, " col index range: ", min(col_indices), " to ", max(col_indices), 
            " (should be 0 to ", ncols-1, ")")
    
    # Create a data frame for the sparse matrix representation
    sparse_df <- data.frame(
      row = row_indices,
      col = col_indices,
      value = values
    )
    
    # Write the sparse matrix representation to CSV
    write.csv(sparse_df, paste0(prefix, "_matrix_sparse.csv"), row.names = FALSE)
    message("Wrote sparse matrix representation to ", prefix, "_matrix_sparse.csv")
    
    # Save row and column names
    write.csv(data.frame(row_names=rownames(matrix_obj)), paste0(prefix, "_matrix_rownames.csv"), row.names=FALSE)
    write.csv(data.frame(col_names=colnames(matrix_obj)), paste0(prefix, "_matrix_colnames.csv"), row.names=FALSE)
    message("Wrote row and column names to separate CSV files for ", prefix)
    
    # Also write matrix dimensions to a separate file for easy loading
    write.csv(data.frame(rows=nrows, cols=ncols), paste0(prefix, "_matrix_dims.csv"), row.names=FALSE)
    message("Wrote matrix dimensions to ", prefix, "_matrix_dims.csv")
    
  } else if (inherits(matrix_obj, "Seurat")) {
    # Handle Seurat object
    message(prefix, " object is a Seurat object with dimensions: ", paste(dim(matrix_obj), collapse=" x "))
    
    # Try loading SeuratDisk for h5ad conversion
    if (check_and_load("SeuratDisk")) {
      message("Converting ", prefix, " to h5ad format...")
      SeuratDisk::SaveH5Seurat(matrix_obj, paste0(prefix, "_temp_converted.h5Seurat"))
      SeuratDisk::Convert(paste0(prefix, "_temp_converted.h5Seurat"), paste0(prefix, "_temp_converted.h5ad"))
      message("Conversion to h5ad completed: ", prefix, "_temp_converted.h5ad")
    } else {
      # Fallback to CSV export if SeuratDisk is not available
      message("SeuratDisk not available. Exporting ", prefix, " to CSV instead...")
      counts <- as.matrix(matrix_obj@assays$RNA@counts)
      write.csv(counts, paste0(prefix, "_seurat_counts.csv"))
      write.csv(matrix_obj@meta.data, paste0(prefix, "_seurat_metadata.csv"))
      message("Exported CSV files: ", prefix, "_seurat_counts.csv, ", prefix, "_seurat_metadata.csv")
    }
  } else {
    # Handle other types of objects
    message(prefix, " object is of type ", class(matrix_obj)[1], ". Trying to convert to a format Python can read...")
    
    # Try to save as RDS format in case Python can read this with rpy2
    saveRDS(matrix_obj, paste0(prefix, "_converted_data.rds"))
    message("Saved object as ", prefix, "_converted_data.rds")
    
    # Try to convert to a data frame or matrix if possible
    tryCatch({
      df <- as.data.frame(matrix_obj)
      write.csv(df, paste0(prefix, "_converted_data.csv"), row.names=TRUE)
      message("Converted to data frame and saved as ", prefix, "_converted_data.csv")
    }, error = function(e) {
      message("Could not convert ", prefix, " to data frame: ", e$message)
    })
  }
}

# Try to load and process both files
tryCatch({
  # Load raw matrix
  if (file.exists(raw_file_path)) {
    message("Reading raw RDS file...")
    raw_matrix <- readRDS(raw_file_path)
    message("Raw RDS file loaded successfully.")
    process_matrix(raw_matrix, "raw")
  } else {
    stop("Raw RDS file does not exist: ", raw_file_path)
  }
  
  # Load normalized matrix
  if (file.exists(norm_file_path)) {
    message("Reading normalized RDS file...")
    norm_matrix <- readRDS(norm_file_path)
    message("Normalized RDS file loaded successfully.")
    process_matrix(norm_matrix, "norm")
  } else {
    stop("Normalized RDS file does not exist: ", norm_file_path)
  }
  
}, error = function(e) {
  message("ERROR: ", e$message)
  quit(status = 1)
})

message("Script completed.")