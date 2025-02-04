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
hcris96_vars = [
    ('beds', 'S300001', '01200', '0100', 'numeric'),
    ('tot_charges', 'G300000', '00100', '0100', 'numeric'),
    ('tot_discounts', 'G300000', '00200', '0100', 'numeric'),
    ('tot_operating_exp', 'G300000', '00400', '0100', 'numeric'),
    ('ip_charges', 'G200000', '00100', '0100', 'numeric'),
    ('icu_charges', 'G200000', '01500', '0100', 'numeric'),
    ('ancillary_charges', 'G200000', '01700', '0100', 'numeric'),
    ('tot_discharges', 'S300001', '00100', '1500', 'numeric'),
    ('mcare_discharges', 'S300001', '00100', '1300', 'numeric'),
    ('mcaid_discharges', 'S300001', '00100', '1400', 'numeric'),
    ('tot_mcare_payment', 'E00A18A', '01600', '0100', 'numeric'),
    ('secondary_mcare_payment', 'E00A18A', '01700', '0100', 'numeric'),
    ('street', 'S200000', '00100', '0100', 'alpha'),
    ('city', 'S200000', '00101', '0100', 'alpha'),
    ('state', 'S200000', '00101', '0200', 'alpha'),
    ('zip', 'S200000', '00101', '0300', 'alpha'),
    ('county', 'S200000', '00101', '0400', 'alpha')
]

# Convert the list into a DataFrame
hcris96_df = pd.DataFrame(hcris96_vars, columns=["variable", "WKSHT_CD", "LINE_NUM", "CLMN_NUM", "source"])

#print data frame
print(hcris96_df)


extract_dir = '/Users/ilsenovis/Documents/GitHub/ECON470HW2/data/input/HCRIS-v1996'

# Initialize an empty list to store data for all years
final_hcris_v1996 = []

# Loop through the years (1995 to 2011) and load the corresponding files
for year in range(1995, 2012):
    print(f"Processing year: {year}")
    report_csv_path = os.path.join(extract_dir, f'hosp_{year}_RPT.CSV')
    alpha_csv_path = os.path.join(extract_dir, f'hosp_{year}_ALPHA.CSV')
    numeric_csv_path = os.path.join(extract_dir, f'hosp_{year}_NMRC.CSV')
    
    if not os.path.exists(report_csv_path):
        print(f"File {report_csv_path} not found. Skipping year {year}...")
        continue
    if not os.path.exists(alpha_csv_path) or not os.path.exists(numeric_csv_path):
        print(f"One or more required files for year {year} are missing. Skipping year {year}...")
        continue

    # Read the report data
    hcris96_report = pd.read_csv(report_csv_path,
                                 names=['RPT_REC_NUM', 'PRVDR_CTRL_TYPE_CD', 'PRVDR_NUM', 'NPI',
                                        'RPT_STUS_CD', 'FY_BGN_DT', 'FY_END_DT', 'PROC_DT',
                                        'INITL_RPT_SW', 'LAST_RPT_SW', 'TRNSMTL_NUM', 'FI_NUM',
                                        'ADR_VNDR_CD', 'FI_CREAT_DT', 'UTIL_CD', 'NPR_DT',
                                        'SPEC_IND', 'FI_RCPT_DT'])
    hcris96_report['year'] = year

    for var in hcris96_vars:
        hcris96_alpha = pd.read_csv(alpha_csv_path,
                                    names=['RPT_REC_NUM', 'WKSHT_CD', 'LINE_NUM', 'CLMN_NUM', 'ITM_VAL_NUM'])
        hcris96_numeric = pd.read_csv(numeric_csv_path,
                                      names=['RPT_REC_NUM', 'WKSHT_CD', 'LINE_NUM', 'CLMN_NUM', 'ITM_VAL_NUM'])
        hcris96_data = hcris96_alpha[(hcris96_alpha['WKSHT_CD'] == var[1]) &
                                     (hcris96_alpha['LINE_NUM'] == var[2]) &
                                     (hcris96_alpha['CLMN_NUM'] == var[3])]
        hcris96_report = pd.merge(hcris96_report, hcris96_data[['RPT_REC_NUM', 'ITM_VAL_NUM']],
                                  on='RPT_REC_NUM', how='left')
        hcris96_report[var[0]] = hcris96_report['ITM_VAL_NUM']
        hcris96_report.drop(columns=['ITM_VAL_NUM'], inplace=True)
        
    # Append the report for the current year
    final_hcris_v1996.append(hcris96_report)

# Combine all years' data into one DataFrame
final_hcris_v1996_df = pd.concat(final_hcris_v1996, ignore_index=True)

# Save the final dataset to the specified path
output_path = '/Users/ilsenovis/Documents/GitHub/ECON470HW2/data/output/final_hcris_v1996.csv'
final_hcris_v1996_df.to_csv(output_path, index=False)

print(f"Final dataset saved to {output_path}")

""" for year in range(1995, 2012):
    year_folder_path = os.path.join(extract_dir, f'HospitalFY{year}')
    if not os.path.exists(year_folder_path):
        print(f"Warning: Folder for year {year} does not exist. Skipping...")
        continue
    report_csv_path = os.path.join(year_folder_path, f'hosp_{year}_RPT.CSV')
    if not os.path.exists(report_csv_path):
        print(f"File {report_csv_path} not found. Skipping year {year}...")
        continue
    hcris96_report = pd.read_csv(report_csv_path,
                                 names=['RPT_REC_NUM', 'PRVDR_CTRL_TYPE_CD', 'PRVDR_NUM', 'NPI',
                                        'RPT_STUS_CD', 'FY_BGN_DT', 'FY_END_DT', 'PROC_DT',
                                        'INITL_RPT_SW', 'LAST_RPT_SW', 'TRNSMTL_NUM', 'FI_NUM',
                                        'ADR_VNDR_CD', 'FI_CREAT_DT', 'UTIL_CD', 'NPR_DT',
                                        'SPEC_IND', 'FI_RCPT_DT'])
    hcris96_report['year'] = year
      
    # Loop through the variables (hcris96_vars) and join the corresponding data
    for var in hcris96_vars:
        alpha_csv_path = os.path.join(year_folder_path, f'hosp_{year}_ALPHA.CSV')
        numeric_csv_path = os.path.join(year_folder_path, f'hosp_{year}_NMRC.CSV')
        if not os.path.exists(alpha_csv_path) or not os.path.exists(numeric_csv_path):
            print(f"One or mroe required files for year {year} are missing. Skippingl...")
            continue
        hcris96_alpha = pd.read_csv(alpha_csv_path,
                                    names=['RPT_REC_NUM', 'WKSHT_CD', 'LINE_NUM', 'CLMN_NUM', 'ITM_VAL_NUM'])
        hcris96_numeric = pd.read_csv(numeric_csv_path,
                                      names=['RPT_REC_NUM', 'WKSHT_CD', 'LINE_NUM', 'CLMN_NUM', 'ITM_VAL_NUM'])
        hcris96_data = hcris96_alpha[(hcris96_alpha['WKSHT_CD'] == var[1]) &
                                     (hcris96_alpha['LINE_NUM'] == var[2]) &
                                     (hcris96_alpha['CLMN_NUM'] == var[3])]
        hcris96_report = pd.merge(hcris96_report, hcris96_data[['RPT_REC_NUM', 'ITM_VAL_NUM']],
                                  on='RPT_REC_NUM', how='left')
        hcris96_report[var[0]] = hcris96_report['ITM_VAL_NUM']
        hcris96_report.drop(columns=['ITM_VAL_NUM'], inplace=True)
    final_hcris_v1996.append(hcris96_report)

# Combine all years' data into one DataFrame
final_hcris_v1996_df = pd.concat(final_hcris_v1996, ignore_index=True)

# Save the final dataset to the specified path
output_path = '/Users/ilsenovis/Documents/GitHub/ECON470HW2/data/output/final_hcris_v1996.csv'
final_hcris_v1996_df.to_csv(output_path, index=False)

print(f"Final dataset saved to {output_path}") """