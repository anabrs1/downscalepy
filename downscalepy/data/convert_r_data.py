"""
Convert R data files to Python format.

This script converts the original R data files from the downscalr package
to Python-friendly formats (CSV or pickle) that can be loaded by the
downscalepy package.
"""

import os
import subprocess
import pandas as pd
import numpy as np
import pickle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ORIGINAL_DIR = os.path.join(SCRIPT_DIR, 'original')
CONVERTED_DIR = os.path.join(SCRIPT_DIR, 'converted')

os.makedirs(CONVERTED_DIR, exist_ok=True)

R_SCRIPT = """
load("{input_file}")

obj_name <- ls()[!ls() %in% c("input_file", "output_file")]

write.csv(get(obj_name), file="{output_file}", row.names=FALSE)

cat("Columns:", paste(colnames(get(obj_name)), collapse=", "), "\n")
"""

def convert_rdata_to_csv(rdata_file, output_dir=CONVERTED_DIR):
    """
    Convert an R data file to CSV using an R script.
    
    Parameters
    ----------
    rdata_file : str
        Path to the R data file.
    output_dir : str
        Directory to save the CSV file.
        
    Returns
    -------
    str
        Path to the converted CSV file.
    """
    base_name = os.path.basename(rdata_file)
    name_without_ext = os.path.splitext(base_name)[0]
    csv_file = os.path.join(output_dir, f"{name_without_ext}.csv")
    
    r_script_path = os.path.join(output_dir, "temp_convert.R")
    with open(r_script_path, "w") as f:
        f.write(R_SCRIPT.format(
            input_file=rdata_file,
            output_file=csv_file
        ))
    
    try:
        result = subprocess.run(
            ["Rscript", r_script_path],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"Converted {rdata_file} to {csv_file}")
        print(result.stdout)
        
        os.remove(r_script_path)
        
        return csv_file
    except subprocess.CalledProcessError as e:
        print(f"Error converting {rdata_file}: {e}")
        print(e.stdout)
        print(e.stderr)
        return None

def load_and_process_argentina_data():
    """
    Load and process all Argentina data files.
    
    Returns
    -------
    dict
        Dictionary containing all the converted data.
    """
    data = {}
    
    for file_name in os.listdir(ORIGINAL_DIR):
        if file_name.endswith('.RData'):
            rdata_path = os.path.join(ORIGINAL_DIR, file_name)
            csv_path = convert_rdata_to_csv(rdata_path)
            
            if csv_path:
                df = pd.read_csv(csv_path)
                
                data_key = os.path.splitext(file_name)[0]
                data[data_key] = df
                
                pickle_path = os.path.join(CONVERTED_DIR, f"{data_key}.pkl")
                with open(pickle_path, 'wb') as f:
                    pickle.dump(df, f)
                print(f"Saved {pickle_path}")
    
    
    with open(os.path.join(CONVERTED_DIR, 'argentina_data.pkl'), 'wb') as f:
        pickle.dump(data, f)
    
    return data

if __name__ == "__main__":
    data = load_and_process_argentina_data()
    
    for key, df in data.items():
        print(f"\n{key}:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Sample:\n{df.head(2)}")
