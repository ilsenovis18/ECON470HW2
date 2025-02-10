# Import libraries
import pandas as pd
import os
import warnings

# Suppress warnings
warnings.simplefilter(action='ignore', category=pd.errors.DtypeWarning)

# Create a list of variables
hcris2010_vars = pd.DataFrame([
    ('beds', 'S300001', 'ALL', '00200', 'numeric'),
    ('tot_charges', 'G300000', 'ALL', '00100', 'numeric'),
    ('tot_discounts', 'G300000', 'ALL', '00100', 'numeric'),
    ('tot_operating_exp', 'G300000', 'ALL', '00100', 'numeric'),
    ('ip_charges', 'G200000', 'ALL', '00100', 'numeric'),
    ('icu_charges', 'G200000', 'ALL', '00100', 'numeric'),
    ('ancillary_charges', 'G200000', 'ALL', '00100', 'numeric'),
    ('tot_discharges', 'S300001', 'ALL', '01500', 'numeric'),
    ('mcare_discharges', 'S300001', 'ALL', '01300', 'numeric'),
    ('mcaid_discharges', 'S300001', 'ALL', '01400', 'numeric'),
    ('tot_mcare_payment', 'E00A18A', 'ALL', '00100', 'numeric'),
    ('secondary_mcare_payment', 'E00A18A', 'ALL', '00100', 'numeric'),
    ('street', 'S200001', 'ALL', '00100', 'alpha'),
    ('city', 'S200001', 'ALL', '00100', 'alpha'),
    ('state', 'S200001', 'ALL', '00200', 'alpha'),
    ('zip', 'S200001', 'ALL', '00300', 'alpha'),
    ('county', 'S200001', 'ALL', '00400', 'alpha'),
    ('hvbp_payment', 'E00A18A', 'ALL', '00100', 'numeric'),
    ('hrrp_payment', 'E00A18A', 'ALL', '00100', 'numeric')
], columns=['variable', 'WKSHT_CD', 'LINE_NUM', 'CLMN_NUM', 'source'])

# Directory with data files
extract_dir = '/Users/ilsenovis/Documents/GitHub/ECON470HW2/data/input/HCRIS-v2010'

# Initialize final DataFrame
final_hcris_v2010 = None

# Loop through each year (2010 to 2019)
for year in range(2010, 2020):
    print(f"Processing year: {year}")
    
    report_csv_path = os.path.join(extract_dir, f'hosp10_{year}_RPT.CSV')
    alpha_csv_path = os.path.join(extract_dir, f'hosp10_{year}_ALPHA.CSV')
    numeric_csv_path = os.path.join(extract_dir, f'hosp10_{year}_NMRC.CSV')

    if not os.path.exists(report_csv_path) or not os.path.exists(alpha_csv_path) or not os.path.exists(numeric_csv_path):
        print(f"Missing data files for year {year}. Skipping...")
        continue

    hcris_alpha = pd.read_csv(alpha_csv_path, names=['RPT_REC_NUM', 'WKSHT_CD', 'LINE_NUM', 'CLMN_NUM', 'ITM_VAL_NUM'])
    hcris_numeric = pd.read_csv(numeric_csv_path, names=['RPT_REC_NUM', 'WKSHT_CD', 'LINE_NUM', 'CLMN_NUM', 'ITM_VAL_NUM'])

    # Ensure proper data types
    hcris_alpha['LINE_NUM'] = hcris_alpha['LINE_NUM'].astype(str)
    hcris_alpha['CLMN_NUM'] = hcris_alpha['CLMN_NUM'].astype(str)
    hcris_numeric['LINE_NUM'] = hcris_numeric['LINE_NUM'].astype(str)
    hcris_numeric['CLMN_NUM'] = hcris_numeric['CLMN_NUM'].astype(str)

    report_data = pd.read_csv(report_csv_path, names=['RPT_REC_NUM', 'PRVDR_CTRL_TYPE_CD', 'PRVDR_NUM', 'NPI',
                                                      'RPT_STUS_CD', 'FY_BGN_DT', 'FY_END_DT', 'PROC_DT',
                                                      'INITL_RPT_SW', 'LAST_RPT_SW', 'TRNSMTL_NUM', 'FI_NUM',
                                                      'ADR_VNDR_CD', 'FI_CREAT_DT', 'UTIL_CD', 'NPR_DT',
                                                      'SPEC_IND', 'FI_RCPT_DT'])

    report_data['year'] = year

    for _, row in hcris2010_vars.iterrows():
        hcris_data = hcris_numeric if row['source'] == 'numeric' else hcris_alpha

        if row['LINE_NUM'] == 'ALL':
            val = (hcris_data[(hcris_data['WKSHT_CD'] == row['WKSHT_CD']) &
                               (hcris_data['CLMN_NUM'] == row['CLMN_NUM'])]
                   .groupby('RPT_REC_NUM')['ITM_VAL_NUM']
                   .sum()
                   .reset_index())
        else:
            val = hcris_data[(hcris_data['WKSHT_CD'] == row['WKSHT_CD']) &
                             (hcris_data['LINE_NUM'] == row['LINE_NUM']) &
                             (hcris_data['CLMN_NUM'] == row['CLMN_NUM'])][['RPT_REC_NUM', 'ITM_VAL_NUM']]

        val.columns = ['RPT_REC_NUM', row['variable']]
        report_data = report_data.merge(val, on='RPT_REC_NUM', how='left')

    if final_hcris_v2010 is None:
        final_hcris_v2010 = report_data
    else:
        final_hcris_v2010 = pd.concat([final_hcris_v2010, report_data], ignore_index=True)

# Save final output
output_path = '/Users/ilsenovis/Documents/GitHub/ECON470HW2/data/output/final_hcris_v2010.csv'
final_hcris_v2010.to_csv(output_path, index=False)

print(f"Final dataset saved to {output_path}")

# Check NaN counts
key_vars = ['tot_discharges', 'mcare_discharges', 'mcaid_discharges',
            'tot_charges', 'tot_discounts', 'tot_operating_exp',
            'ip_charges', 'icu_charges', 'ancillary_charges',
            'tot_mcare_payment', 'secondary_mcare_payment']

final_data = pd.read_csv(output_path)

print("\nNaN Counts for Key Variables:")
for var in key_vars:
    print(f"{var}: {final_data[var].isna().sum()} NaN values")
