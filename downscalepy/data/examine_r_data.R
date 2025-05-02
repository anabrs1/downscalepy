# Script to examine and export R data from downscalr package

# Function to examine an R data file
examine_rdata <- function(file_path) {
  # Load the data
  load(file_path)
  
  # Get the object name
  obj_name <- ls()[!ls() %in% c("file_path")]
  
  # Get the object
  obj <- get(obj_name)
  
  # Print object name
  cat("Object name:", obj_name, "\n")
  
  # Print object class
  cat("Object class:", class(obj), "\n")
  
  # If it's a list, examine each component
  if (is.list(obj)) {
    cat("List components:\n")
    for (name in names(obj)) {
      component <- obj[[name]]
      cat("  - ", name, ": ", class(component), "\n")
      
      # If component is a data frame, print its dimensions and column names
      if (is.data.frame(component)) {
        cat("    Dimensions: ", nrow(component), " x ", ncol(component), "\n")
        cat("    Columns: ", paste(colnames(component), collapse=", "), "\n")
      } else if (is.matrix(component)) {
        cat("    Dimensions: ", nrow(component), " x ", ncol(component), "\n")
      } else if (is.vector(component)) {
        cat("    Length: ", length(component), "\n")
      }
    }
  } else if (is.data.frame(obj)) {
    # If it's a data frame, print its dimensions and column names
    cat("Dimensions: ", nrow(obj), " x ", ncol(obj), "\n")
    cat("Columns: ", paste(colnames(obj), collapse=", "), "\n")
  }
  
  # Return the object name
  return(obj_name)
}

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
  if (is.list(obj)) {
    cat("Exporting list components:\n")
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
    cat("Exported data frame to", output_file, "\n")
  } else {
    cat("Object is not a list or data frame, cannot export\n")
  }
}

# Directory paths
original_dir <- "~/repos/downscalepy/downscalepy/data/original"
converted_dir <- "~/repos/downscalepy/downscalepy/data/converted"

# Create converted directory if it doesn't exist
dir.create(converted_dir, showWarnings = FALSE, recursive = TRUE)

# Examine each R data file
cat("\n=== EXAMINING R DATA FILES ===\n")
rdata_files <- list.files(original_dir, pattern = "\\.RData$", full.names = TRUE)
for (file in rdata_files) {
  cat("\nExamining", basename(file), ":\n")
  examine_rdata(file)
}

# Export components of each R data file
cat("\n=== EXPORTING R DATA FILES ===\n")
for (file in rdata_files) {
  cat("\nExporting", basename(file), ":\n")
  export_rdata_components(file, converted_dir)
}

cat("\nDone!\n")
