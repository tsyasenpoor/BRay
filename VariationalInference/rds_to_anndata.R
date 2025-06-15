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

# Get command-line arguments
args <- commandArgs(trailingOnly = TRUE)
input_file <- args[1]
if (is.na(input_file)) {
  input_file <- "/labs/Aguiar/SSPA_BRAY/BRay/BRAY_AJM2/2_Data/2_SingleCellData/2_AJM_Parse_Timecourse/GEX_TC_LPSonly_Bcellonly_filt_raw_2024-02-05.rds"
}

message("Input RDS file: ", input_file)
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

# Try to directly save the RDS
tryCatch({
  # Check if we can read the RDS file
  if (file.exists(input_file)) {
    message("Reading RDS file...")
    matrix_obj <- readRDS(input_file)
    message("RDS file loaded successfully.")
    message("Object class: ", class(matrix_obj)[1])
    
    # Handle different types of objects
    if (inherits(matrix_obj, "dgCMatrix") || inherits(matrix_obj, "dgTMatrix")) {
      # This is a sparse matrix, we can directly save it
      message("Object is a sparse matrix with dimensions: ", paste(dim(matrix_obj), collapse=" x "))
      
      # Get dimensions of the matrix
      nrows <- nrow(matrix_obj)
      ncols <- ncol(matrix_obj)
      message("Matrix dimensions: rows=", nrows, ", cols=", ncols)
      
      # Get indices and values
      sparse_summary <- summary(matrix_obj)
      
      # Adjust indices to be 0-based for Python
      # This assumes summary() returns 1-based indices (R style)
      row_indices <- sparse_summary[,1] - 1
      col_indices <- sparse_summary[,2] - 1
      values <- sparse_summary[,3]
      
      # Check index ranges
      message("Row index range: ", min(row_indices), " to ", max(row_indices), 
              " (should be 0 to ", nrows-1, ")")
      message("Col index range: ", min(col_indices), " to ", max(col_indices), 
              " (should be 0 to ", ncols-1, ")")
      
      # Create a data frame for the sparse matrix representation
      sparse_df <- data.frame(
        row = row_indices,
        col = col_indices,
        value = values
      )
      
      # Write the sparse matrix representation to CSV
      write.csv(sparse_df, "matrix_sparse.csv", row.names = FALSE)
      message("Wrote sparse matrix representation to matrix_sparse.csv")
      
      # Save row and column names
      write.csv(data.frame(row_names=rownames(matrix_obj)), "matrix_rownames.csv", row.names=FALSE)
      write.csv(data.frame(col_names=colnames(matrix_obj)), "matrix_colnames.csv", row.names=FALSE)
      message("Wrote row and column names to separate CSV files")
      
      # Also write matrix dimensions to a separate file for easy loading
      write.csv(data.frame(rows=nrows, cols=ncols), "matrix_dims.csv", row.names=FALSE)
      message("Wrote matrix dimensions to matrix_dims.csv")
      
    } else if (inherits(matrix_obj, "Seurat")) {
      # Handle Seurat object
      message("Object is a Seurat object with dimensions: ", paste(dim(matrix_obj), collapse=" x "))
      
      # Try loading SeuratDisk for h5ad conversion
      if (check_and_load("SeuratDisk")) {
        message("Converting to h5ad format...")
        SeuratDisk::SaveH5Seurat(matrix_obj, "temp_converted.h5Seurat")
        SeuratDisk::Convert("temp_converted.h5Seurat", "temp_converted.h5ad")
        message("Conversion to h5ad completed: temp_converted.h5ad")
      } else {
        # Fallback to CSV export if SeuratDisk is not available
        message("SeuratDisk not available. Exporting to CSV instead...")
        counts <- as.matrix(matrix_obj@assays$RNA@counts)
        write.csv(counts, "seurat_counts.csv")
        write.csv(matrix_obj@meta.data, "seurat_metadata.csv")
        message("Exported CSV files: seurat_counts.csv, seurat_metadata.csv")
      }
    } else {
      # Handle other types of objects
      message("Object is of type ", class(matrix_obj)[1], ". Trying to convert to a format Python can read...")
      
      # Try to save as RDS format in case Python can read this with rpy2
      saveRDS(matrix_obj, "converted_data.rds")
      message("Saved object as converted_data.rds")
      
      # Try to convert to a data frame or matrix if possible
      tryCatch({
        df <- as.data.frame(matrix_obj)
        write.csv(df, "converted_data.csv", row.names=TRUE)
        message("Converted to data frame and saved as converted_data.csv")
      }, error = function(e) {
        message("Could not convert to data frame: ", e$message)
      })
    }
  } else {
    stop("RDS file does not exist: ", input_file)
  }
}, error = function(e) {
  message("ERROR: ", e$message)
  quit(status = 1)
})

message("Script completed.")