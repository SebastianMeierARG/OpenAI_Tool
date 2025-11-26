import pandas as pd
import numpy as np

def find_header_row(df, keywords):
    """
    Scans the DataFrame to find the row index where the first column contains
    one of the keywords.
    """
    for i, row in df.iterrows():
        # Check first few columns for the keyword
        # We convert to string and strip to handle potential whitespace
        row_values = [str(x).strip() for x in row.values if pd.notna(x)]
        for val in row_values:
            if val in keywords:
                return i
    return 0

def load_data(filepath='MRM Tables.xlsx'):
    """
    Loads data from MRM Tables.xlsx, handles variable headers,
    and performs required calculations.
    """
    try:
        # Load sheets without header initially to scan
        xls = pd.ExcelFile(filepath)

        # Process PD - Overview
        df_overview_raw = pd.read_excel(xls, sheet_name='PD - Overview', header=None)
        header_row_ov = find_header_row(df_overview_raw, ['Reference_Date', 'Reference Date'])

        # Reload with correct header
        df_overview = pd.read_excel(xls, sheet_name='PD - Overview', header=header_row_ov)

        # Calculate Gini: 2 * AUC_curr - 1
        if 'AUC_curr' in df_overview.columns:
            df_overview['Gini'] = 2 * df_overview['AUC_curr'] - 1
        else:
            print("Warning: 'AUC_curr' column not found in PD - Overview")

        # Process PD - Structure
        df_structure_raw = pd.read_excel(xls, sheet_name='PD - Structure', header=None)
        header_row_st = find_header_row(df_structure_raw, ['Grade', 'Reference_Date', 'Reference Date'])

        # Reload with correct header
        df_structure = pd.read_excel(xls, sheet_name='PD - Structure', header=header_row_st)

        # Calculate ODR: n_default / n (ignoring existing ODR)
        if 'n_default' in df_structure.columns and 'n' in df_structure.columns:
            # Ensure numeric
            df_structure['n_default'] = pd.to_numeric(df_structure['n_default'], errors='coerce').fillna(0)
            df_structure['n'] = pd.to_numeric(df_structure['n'], errors='coerce').fillna(1) # avoid div by zero

            df_structure['ODR_Calculated'] = df_structure['n_default'] / df_structure['n']
        else:
            print("Warning: 'n_default' or 'n' columns not found in PD - Structure")

        return df_overview, df_structure

    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame()

if __name__ == "__main__":
    ov, st = load_data()
    print("Overview Head:")
    print(ov.head())
    print("\nStructure Head:")
    print(st.head())
