"""
Script to convert argentina_raster.RData to GeoTIFF format.
This is a standalone script that doesn't require the full package.
"""

import os
import subprocess
import sys
from datetime import datetime

def log(message):
    """Log a message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

def create_r_script(output_path):
    """Create a temporary R script to convert the RData file."""
    r_script = """
    if (!require("terra")) {{
        install.packages("terra", repos="https://cloud.r-project.org")
        library(terra)
    }}
    if (!require("raster")) {{
        install.packages("raster", repos="https://cloud.r-project.org")
        library(raster)
    }}

    data_dir <- "/storage/lopesas/downscalepy/downscalepy/data"
    rdata_file <- file.path(data_dir, "argentina_raster.RData")
    output_file <- "{output_path}"

    dir.create(dirname(output_file), showWarnings = FALSE, recursive = TRUE)

    print(paste("Loading RData file:", rdata_file))
    load(rdata_file)

    print("Objects loaded:")
    print(ls())

    raster_objects <- ls()[sapply(ls(), function(x) {{
        obj <- get(x)
        return(inherits(obj, "RasterLayer") || 
               inherits(obj, "SpatRaster") || 
               inherits(obj, "RasterStack") || 
               inherits(obj, "RasterBrick"))
    }})]

    print(paste("Found raster objects:", paste(raster_objects, collapse=", ")))

    if (length(raster_objects) > 0) {{
        raster_obj <- get(raster_objects[1])
        print(paste("Using raster object:", raster_objects[1]))
        
        if (inherits(raster_obj, "RasterLayer") || 
            inherits(raster_obj, "RasterStack") || 
            inherits(raster_obj, "RasterBrick")) {{
            print("Converting raster to terra SpatRaster")
            raster_obj <- terra::rast(raster_obj)
        }}
        
        print(paste("Writing to GeoTIFF:", output_file))
        terra::writeRaster(raster_obj, output_file, overwrite=TRUE)
        print("Conversion complete!")
    }} else {{
        for (obj_name in ls()) {{
            obj <- get(obj_name)
            print(paste("Examining object:", obj_name, "of class:", class(obj)[1]))
            
            tryCatch({{
                if (is.matrix(obj) || is.array(obj)) {{
                    print(paste("Converting matrix/array to raster:", obj_name))
                    r <- terra::rast(obj)
                    print(paste("Writing to GeoTIFF:", output_file))
                    terra::writeRaster(r, output_file, overwrite=TRUE)
                    print("Conversion complete!")
                    break
                }}
            }}, error = function(e) {{
                print(paste("Error converting", obj_name, ":", e$message))
            }})
        }}
    }}

    if (file.exists(output_file)) {{
        print(paste("Success! GeoTIFF created at:", output_file))
        if (require("terra")) {{
            r <- terra::rast(output_file)
            print("Raster information:")
            print(paste("Dimensions:", nrow(r), "x", ncol(r), "x", nlyr(r)))
            print(paste("Resolution:", xres(r), "x", yres(r)))
            print(paste("Extent:", xmin(r), ",", xmax(r), ",", ymin(r), ",", ymax(r)))
            print("CRS information:")
            crs_text <- crs(r)
            print(paste("CRS:", ifelse(is.na(crs_text), "Not available", as.character(crs_text))))
        }}
    }} else {{
        print(paste("Error: Failed to create GeoTIFF at:", output_file))
    }}
    """.format(output_path=output_path)
    
    script_path = "/tmp/convert_argentina_raster.R"
    with open(script_path, "w") as f:
        f.write(r_script)
    
    return script_path

def main():
    """Main function to convert the RData file to GeoTIFF."""
    log("Starting conversion of argentina_raster.RData to GeoTIFF")
    
    data_dir = "/storage/lopesas/downscalepy/downscalepy/data"
    converted_dir = os.path.join(data_dir, "converted")
    output_path = os.path.join(converted_dir, "argentina_raster.tif")
    
    os.makedirs(converted_dir, exist_ok=True)
    
    r_script_path = create_r_script(output_path)
    log(f"Created R script at {r_script_path}")
    
    log("Running R script to convert RData to GeoTIFF")
    try:
        result = subprocess.run(
            ["Rscript", r_script_path],
            check=True,
            text=True,
            capture_output=True
        )
        log("R script output:")
        for line in result.stdout.splitlines():
            log(f"  {line}")
    except subprocess.CalledProcessError as e:
        log(f"Error running R script: {e}")
        log("R script error output:")
        for line in e.stdout.splitlines():
            log(f"  {line}")
        for line in e.stderr.splitlines():
            log(f"  {line}")
        return 1
    
    if os.path.exists(output_path):
        log(f"Success! GeoTIFF created at: {output_path}")
    else:
        log(f"Error: Failed to create GeoTIFF at: {output_path}")
        return 1
    
    log("Conversion complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
