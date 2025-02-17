import pandas as pd
import numpy as np

# Load datasets
v1996_path = '/Users/ilsenovis/Documents/GitHub/ECON470HW2/data/output/final_hcris_v1996.csv'
v2010_path = '/Users/ilsenovis/Documents/GitHub/ECON470HW2/data/output/final_hcris_v2010.csv'

final_hcris_v1996 = pd.read_csv(v1996_path)
final_hcris_v2010 = pd.read_csv(v2010_path)

# Check column differences before merging
v1996_cols = set(final_hcris_v1996.columns)
v2010_cols = set(final_hcris_v2010.columns)

# Rename incorrect columns
rename_columns = {
    'PRVDR_NUM': 'provider_number',
    'NPI': 'npi',
    'FY_BGN_DT': 'fy_start',
    'FY_END_DT': 'fy_end',
    'PROC_DT': 'date_processed',
    'FI_CREAT_DT': 'date_created'
}

final_hcris_v1996.rename(columns=rename_columns, inplace=True)
final_hcris_v2010.rename(columns=rename_columns, inplace=True)

# Add source year
final_hcris_v1996['source_year'] = '1996'
final_hcris_v2010['source_year'] = '2010'

# Create missing variables for columns introduced in v2010
final_hcris_v1996['hvbp_payment'] = np.nan
final_hcris_v1996['hrrp_payment'] = np.nan

final_hcris_v1996['fyear'] = pd.to_datetime(final_hcris_v1996['fy_end'], errors='coerce').dt.year
final_hcris_v2010['fyear'] = pd.to_datetime(final_hcris_v2010['fy_end'], errors='coerce').dt.year

print("Years in 1996:", sorted(final_hcris_v1996['fyear'].dropna().unique()))
print("Years in 2010:", sorted(final_hcris_v2010['fyear'].dropna().unique()))

print(final_hcris_v2010[['fy_end', 'fyear']].dropna().head())

# Combine datasets
final_hcris = pd.concat([final_hcris_v1996, final_hcris_v2010], ignore_index=True)

# Convert date columns to datetime format
date_columns = ['fy_end', 'fy_start', 'date_processed', 'date_created']
for col in date_columns:
    if col in final_hcris.columns:
        final_hcris[col] = pd.to_datetime(final_hcris[col], errors='coerce')

print("Years in merged dataset:", sorted(final_hcris['fyear'].dropna().unique()))
print(final_hcris[final_hcris['fyear'] == 2012])

# Convert to absolute values
final_hcris['tot_discounts'] = final_hcris['tot_discounts'].abs()
final_hcris['hrrp_payment'] = final_hcris['hrrp_payment'].abs()

# Extract fiscal year and sort
if 'fy_end' in final_hcris.columns:
    final_hcris['fyear'] = final_hcris['fy_end'].dt.year
final_hcris = final_hcris.sort_values(by=['provider_number', 'fyear']).drop(columns=['year'], errors='ignore')

# Save combined dataset
output_path = '/Users/ilsenovis/Documents/GitHub/ECON470HW2/data/output/HCRIS_Data.csv'
final_hcris.to_csv(output_path, index=False)

# Confirmation message
print(f"Final merged dataset saved successfully to {output_path}!")

# Check data ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Check for NaN values in all columns
nan_counts = final_hcris.isna().sum()
print("\nNaN Counts for Each Column:")
print(nan_counts[nan_counts > 0])

# List of key variables to check
key_vars = [
    'tot_discharges', 'mcare_discharges', 'mcaid_discharges',
    'tot_charges', 'tot_discounts', 'tot_operating_exp',
    'ip_charges', 'icu_charges', 'ancillary_charges',
    'tot_mcare_payment', 'secondary_mcare_payment',
    'hvbp_payment', 'hrrp_payment'
]

# Check for NaN counts in key variables
print("\nNaN Counts for Key Variables:")
for var in key_vars:
    if var in final_hcris.columns:
        print(f"{var}: {final_hcris[var].isna().sum()} NaN values")

# Percentage of missing data per column
missing_percentage = (final_hcris.isna().sum() / len(final_hcris)) * 100
print("\nMissing Percentage by Column:")
print(missing_percentage[missing_percentage > 0].sort_values(ascending=False))

# Missing data by year
missing_by_year = final_hcris.groupby('fyear')[key_vars].apply(lambda x: x.isna().sum())
print("\nMissing Data by Year:")
print(missing_by_year)

# Missing data by provider
missing_by_provider = final_hcris.groupby('provider_number')[key_vars].apply(lambda x: x.isna().sum())
print("\nMissing Data by Provider:")
print(missing_by_provider.head(10))

# Check if provider_number exists and its missing count
print("\nMissing Provider Numbers:")
print(final_hcris['provider_number'].isna().sum())

# Check how many rows are labeled as '1996' and '2010'
print("\nSource Year Counts:")
print(final_hcris['source_year'].value_counts())

# Confirmation
print("\nData checks completed!")