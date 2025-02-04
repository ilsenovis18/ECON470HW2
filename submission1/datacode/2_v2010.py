# Import libraries
import pandas as pd
import zipfile
import os
import glob
import matplotlib.pyplot as plt
from IPython.display import Markdown, display
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.DtypeWarning)

# Create a list of variables
hcris2010_vars = [
    ('beds', 'S300001', '01400', '00200', 'numeric'),
    ('tot_charges', 'G300000', '00100', '00100', 'numeric'),
    ('tot_discounts', 'G300000', '00200', '00100', 'numeric'),
    ('tot_operating_exp', 'G300000', '00400', '00100', 'numeric'),
    ('ip_charges', 'G200000', '00100', '00100', 'numeric'),
    ('icu_charges', 'G200000', '01600', '00100', 'numeric'),
    ('ancillary_charges', 'G200000', '01800', '00100', 'numeric'),
    ('tot_discharges', 'S300001', '00100', '01500', 'numeric'),
    ('mcare_discharges', 'S300001', '00100', '01300', 'numeric'),
    ('mcaid_discharges', 'S300001', '00100', '01400', 'numeric'),
    ('tot_mcare_payment', 'E00A18A', '05900', '00100', 'numeric'),
    ('secondary_mcare_payment', 'E00A18A', '06000', '00100', 'numeric'),
    ('street', 'S200001', '00100', '00100', 'alpha'),
    ('city', 'S200001', '00200', '00100', 'alpha'),
    ('state', 'S200001', '00200', '00200', 'alpha'),
    ('zip', 'S200001', '00200', '00300', 'alpha'),
    ('county', 'S200001', '00200', '00400', 'alpha'),
    ('hvbp_payment', 'E00A18A', '07093', '00100', 'numeric'),
    ('hrrp_payment', 'E00A18A', '07094', '00100', 'numeric')
]

# Convert the list into a DataFrame
hcris2010_df = pd.DataFrame(hcris2010_vars, columns=["variable", "WKSHT_CD", "LINE_NUM", "CLMN_NUM", "source"])

# Print the DataFrame to check the result
print(hcris2010_df)

# Define the directory where the files are stored
extract_dir = '/Users/ilsenovis/Documents/GitHub/ECON470HW2/data/input/HCRIS-v2010'

# Initialize an empty list to store the processed data for all years
final_hcris_v2010 = []

# Loop through each year (2010 to 2019)
for year in range(2010, 2021):
    print(f"Processing year: {year}")
    
    report_csv_path = os.path.join(extract_dir, f'hosp10_{year}_RPT.CSV')
    alpha_csv_path = os.path.join(extract_dir, f'hosp10_{year}_ALPHA.CSV')
    numeric_csv_path = os.path.join(extract_dir, f'hosp10_{year}_NMRC.CSV')
    
    if not os.path.exists(report_csv_path):
        print(f"File {report_csv_path} not found. Skipping year {year}...")
        continue
    
    if not os.path.exists(alpha_csv_path) or not os.path.exists(numeric_csv_path):
        print(f"One or more required files for year {year} are missing. Skipping...")
        continue
    
    hcris2010_report = pd.read_csv(report_csv_path,
                                   names=['RPT_REC_NUM', 'PRVDR_CTRL_TYPE_CD', 'PRVDR_NUM', 'NPI',
                                          'RPT_STUS_CD', 'FY_BGN_DT', 'FY_END_DT', 'PROC_DT',
                                          'INITL_RPT_SW', 'LAST_RPT_SW', 'TRNSMTL_NUM', 'FI_NUM',
                                          'ADR_VNDR_CD', 'FI_CREAT_DT', 'UTIL_CD', 'NPR_DT',
                                          'SPEC_IND', 'FI_RCPT_DT'])
    hcris2010_report['year'] = year
    
    for var in hcris2010_vars:
        hcris2010_alpha = pd.read_csv(alpha_csv_path,
                                      names=['RPT_REC_NUM', 'WKSHT_CD', 'LINE_NUM', 'CLMN_NUM', 'ITM_VAL_NUM'])
        hcris2010_numeric = pd.read_csv(numeric_csv_path,
                                        names=['RPT_REC_NUM', 'WKSHT_CD', 'LINE_NUM', 'CLMN_NUM', 'ITM_VAL_NUM'])
        
        hcris2010_data = hcris2010_alpha[(hcris2010_alpha['WKSHT_CD'] == var[1]) &
                                         (hcris2010_alpha['LINE_NUM'] == var[2]) &
                                         (hcris2010_alpha['CLMN_NUM'] == var[3])]
        
        hcris2010_report = pd.merge(hcris2010_report, hcris2010_data[['RPT_REC_NUM', 'ITM_VAL_NUM']],
                                    on='RPT_REC_NUM', how='left')
        hcris2010_report[var[0]] = hcris2010_report['ITM_VAL_NUM']
        hcris2010_report.drop(columns=['ITM_VAL_NUM'], inplace=True)
        
    final_hcris_v2010.append(hcris2010_report)

# Combine all years' data into one DataFrame
final_hcris_v2010_df = pd.concat(final_hcris_v2010, ignore_index=True)

# Save the final dataset to the specified path
output_path = '/Users/ilsenovis/Documents/GitHub/ECON470HW2/data/output/final_hcris_v2010.csv'
final_hcris_v2010_df.to_csv(output_path, index=False)

print(f"Final dataset saved to {output_path}")