#!/usr/bin/env Rscript
# Convert argentina_raster.RData to GeoTIFF format
# This script loads the argentina_raster.RData file and saves it as a GeoTIFF

# Print information about what we're doing
cat("Converting argentina_raster.RData to GeoTIFF format...\n")

# Load required libraries
if (!require("terra")) {
  cat("Installing terra package...\n")
  install.packages("terra", repos = "https://cloud.r-project.org")
  library(terra)
}

# Define paths
script_dir <- getwd()
cat("Current directory:", script_dir, "\n")

# Try to find the RData file
rdata_paths <- c(
  "argentina_raster.RData",
  "data/argentina_raster.RData",
  "../data/argentina_raster.RData",
  "../../data/argentina_raster.RData",
  "/storage/lopesas/downscalepy/downscalepy/data/argentina_raster.RData",
  "/home/ubuntu/temp/downscalr/data/argentina_raster.RData"
)

rdata_file <- NULL
for (path in rdata_paths) {
  if (file.exists(path)) {
    rdata_file <- path
    cat("Found RData file at:", rdata_file, "\n")
    break
  }
}

if (is.null(rdata_file)) {
  stop("Could not find argentina_raster.RData file. Please provide the correct path.")
}

# Define output path
output_dir <- "converted"
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
  cat("Created output directory:", output_dir, "\n")
}

output_file <- file.path(output_dir, "argentina_raster.tif")
cat("Output file will be:", output_file, "\n")

# Load the RData file
cat("Loading RData file...\n")
load(rdata_file)

# List objects in the environment
objects <- ls()
cat("Objects in environment:", paste(objects, collapse=", "), "\n")

# Find the raster object
raster_obj <- NULL
for (obj in objects) {
  if (inherits(get(obj), "SpatRaster") || 
      inherits(get(obj), "RasterLayer") || 
      inherits(get(obj), "RasterStack") || 
      inherits(get(obj), "RasterBrick")) {
    raster_obj <- obj
    cat("Found raster object:", raster_obj, "\n")
    break
  }
}

if (is.null(raster_obj)) {
  stop("Could not find a raster object in the RData file.")
}

# Get the raster object
raster <- get(raster_obj)

# Print information about the raster
cat("Raster information:\n")
cat("Dimensions:", paste(dim(raster), collapse=" x "), "\n")
cat("Resolution:", paste(res(raster), collapse=" x "), "\n")
cat("Extent:", paste(as.vector(ext(raster)), collapse=", "), "\n")
cat("CRS:", crs(raster), "\n")

# Save as GeoTIFF
cat("Saving as GeoTIFF...\n")
writeRaster(raster, output_file, overwrite=TRUE)

# Verify the file was created
if (file.exists(output_file)) {
  cat("SUCCESS: GeoTIFF file created at:", output_file, "\n")
  cat("File size:", file.size(output_file), "bytes\n")
} else {
  cat("ERROR: Failed to create GeoTIFF file.\n")
}

# Also save to the storage path if it exists
storage_path <- "/storage/lopesas/downscalepy/downscalepy/data/converted"
if (dir.exists(storage_path)) {
  storage_file <- file.path(storage_path, "argentina_raster.tif")
  cat("Also saving to storage path:", storage_file, "\n")
  writeRaster(raster, storage_file, overwrite=TRUE)
  
  if (file.exists(storage_file)) {
    cat("SUCCESS: GeoTIFF file created at:", storage_file, "\n")
    cat("File size:", file.size(storage_file), "bytes\n")
  } else {
    cat("ERROR: Failed to create GeoTIFF file at storage path.\n")
  }
}

cat("Conversion complete!\n")
