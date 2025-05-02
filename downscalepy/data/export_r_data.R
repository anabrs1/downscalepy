# Script to export R data from downscalr package to CSV files

# Function to export components of an R data file to separate CSV files
export_rdata_components <- function(file_path, output_dir) {
  # Load the data
  load(file_path)
  
  # Get the object name
  obj_name <- ls()[!ls() %in% c("file_path", "output_dir")]
  
  # Get the object
  obj <- get(obj_name)
  
  # Base name for output files
  base_name <- tools::file_path_sans_ext(basename(file_path))
  
  # If it's a list, export each component
  if (is.list(obj) && !is.data.frame(obj)) {
    cat("Exporting list components for", base_name, ":\n")
    for (name in names(obj)) {
      component <- obj[[name]]
      output_file <- file.path(output_dir, paste0(base_name, "_", name, ".csv"))
      
      # Export based on component type
      if (is.data.frame(component)) {
        write.csv(component, file=output_file, row.names=FALSE)
        cat("  - Exported", name, "to", output_file, "\n")
      } else if (is.matrix(component)) {
        write.csv(as.data.frame(component), file=output_file, row.names=FALSE)
        cat("  - Exported", name, "to", output_file, "\n")
      } else if (is.vector(component) && !is.list(component)) {
        # For simple vectors, create a data frame with one column
        df <- data.frame(value = component)
        write.csv(df, file=output_file, row.names=FALSE)
        cat("  - Exported", name, "to", output_file, "\n")
      } else {
        cat("  - Skipped", name, "(not a data frame, matrix, or simple vector)\n")
      }
    }
  } else if (is.data.frame(obj)) {
    # If it's a data frame, export it directly
    output_file <- file.path(output_dir, paste0(base_name, ".csv"))
    write.csv(obj, file=output_file, row.names=FALSE)
    cat("Exported data frame", base_name, "to", output_file, "\n")
  } else {
    cat("Object", base_name, "is not a list or data frame, cannot export\n")
  }
}

# Directory paths
original_dir <- "~/repos/downscalepy/downscalepy/data/original"
converted_dir <- "~/repos/downscalepy/downscalepy/data/converted"

# Create converted directory if it doesn't exist
dir.create(converted_dir, showWarnings = FALSE, recursive = TRUE)

# Export each R data file
rdata_files <- list.files(original_dir, pattern = "\\.RData$", full.names = TRUE)
for (file in rdata_files) {
  cat("\nProcessing", basename(file), ":\n")
  export_rdata_components(file, converted_dir)
}

cat("\nDone!\n")
