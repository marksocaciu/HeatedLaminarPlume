import pandas as pd

def convert_cgi_to_csv(input_file: str, output_file: str):
    """
    Convert a space/tab-delimited NIST/REFPROP-style .cgi file into a clean .csv.

    Args:
        input_file (str): Path to input .cgi file
        output_file (str): Path to save the cleaned .csv
    """
    # Read file (pandas infers spaces/tabs as delimiter with delim_whitespace=True)
    df = pd.read_csv(input_file, delim_whitespace=True)

    # Clean column names (remove parentheses, spaces, replace with underscores)
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"[()]", "", regex=True)
        .str.replace(r"\.", "", regex=True)
        .str.replace(r"\*", "star", regex=True)
        .str.replace(r"\s+", "_", regex=True)
    )

    # Save as proper CSV
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    # Example usage
    input_path = "water.cgi"   # your input file
    output_path = "water_properties.csv"  # desired output
    convert_cgi_to_csv(input_path, output_path)
