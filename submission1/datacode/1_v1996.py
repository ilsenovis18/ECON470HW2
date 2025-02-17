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
hcris96_vars = pd.DataFrame([
    ('beds', 'S300001', 'ALL', '0100', 'numeric'),
    ('tot_charges', 'G300000', '100', '0100', 'numeric'),
    ('tot_discounts', 'G300000', '200', '0100', 'numeric'),
    ('tot_operating_exp', 'G300000', '400', '0100', 'numeric'),
    ('ip_charges', 'G200000', '100', '0100', 'numeric'),
    ('icu_charges', 'G200000', '1500', '0100', 'numeric'),
    ('ancillary_charges', 'G200000', '1700', '0100', 'numeric'),
    ('tot_discharges', 'S300001', 'ALL', '1500', 'numeric'),
    ('mcare_discharges', 'S300001', 'ALL', '1300', 'numeric'),
    ('mcaid_discharges', 'S300001', 'ALL', '1400', 'numeric'),
    ('tot_mcare_payment', 'E00A18A', '1600', '0100', 'numeric'),
    ('secondary_mcare_payment', 'E00A18A', '1700', '0100', 'numeric'),
    ('street', 'S200000', '00100', '0100', 'alpha'),
    ('city', 'S200000', '00101', '0100', 'alpha'),
    ('state', 'S200000', '00101', '0200', 'alpha'),
    ('zip', 'S200000', '00101', '0300', 'alpha'),
    ('county', 'S200000', '00101', '0400', 'alpha')
], columns=['variable', 'WKSHT_CD', 'LINE_NUM', 'CLMN_NUM', 'source'])

# Pull relevant data: v1996 of HCRIS forms run through 2011 due to lags in processing and hospital FYs
final_hcris_v1996 = None

# Loop through the years (2008 to 2009)
for year in range(2008, 2010):
    print(f"Processing year: {year}")
    report_csv_path = f"/Users/ilsenovis/Documents/GitHub/ECON470HW2/data/input/HCRIS-v1996/hosp_{year}_RPT.CSV"
    alpha_csv_path = f"/Users/ilsenovis/Documents/GitHub/ECON470HW2/data/input/HCRIS-v1996/hosp_{year}_ALPHA.CSV"
    numeric_csv_path = f"/Users/ilsenovis/Documents/GitHub/ECON470HW2/data/input/HCRIS-v1996/hosp_{year}_NMRC.CSV"

    HCRIS_alpha = pd.read_csv(alpha_csv_path, names=['RPT_REC_NUM', 'WKSHT_CD', 'LINE_NUM', 'CLMN_NUM', 'ITM_VAL_NUM'])
    HCRIS_numeric = pd.read_csv(numeric_csv_path, names=['RPT_REC_NUM', 'WKSHT_CD', 'LINE_NUM', 'CLMN_NUM', 'ITM_VAL_NUM'])

    # Ensure LINE_NUM and CLMN_NUM are strings
    HCRIS_alpha['LINE_NUM'] = HCRIS_alpha['LINE_NUM'].astype(str)
    HCRIS_alpha['CLMN_NUM'] = HCRIS_alpha['CLMN_NUM'].astype(str)
    HCRIS_numeric['LINE_NUM'] = HCRIS_numeric['LINE_NUM'].astype(str)
    HCRIS_numeric['CLMN_NUM'] = HCRIS_numeric['CLMN_NUM'].astype(str)

    HCRIS_report = pd.read_csv(report_csv_path, names=['RPT_REC_NUM', 'PRVDR_CTRL_TYPE_CD', 'PRVDR_NUM', 'NPI',
                                                       'RPT_STUS_CD', 'FY_BGN_DT', 'FY_END_DT', 'PROC_DT',
                                                       'INITL_RPT_SW', 'LAST_RPT_SW', 'TRNSMTL_NUM', 'FI_NUM',
                                                       'ADR_VNDR_CD', 'FI_CREAT_DT', 'UTIL_CD', 'NPR_DT',
                                                       'SPEC_IND', 'FI_RCPT_DT'])
    final_reports = HCRIS_report[['RPT_REC_NUM', 'PRVDR_NUM', 'NPI', 'FY_BGN_DT', 'FY_END_DT', 'PROC_DT',
                                  'FI_CREAT_DT', 'RPT_STUS_CD']].copy()
    final_reports.columns = ['report', 'provider_number', 'npi', 'fy_start', 'fy_end', 'date_processed',
                             'date_created', 'status']
    final_reports.loc[:, 'year'] = year

    for _, row in hcris96_vars.iterrows():
        hcris_data = HCRIS_numeric if row['source'] == 'numeric' else HCRIS_alpha

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

        val.columns = ['report', row['variable']]
        final_reports = final_reports.merge(val, on='report', how='left')

    if final_hcris_v1996 is None:
        final_hcris_v1996 = final_reports
    else:
        final_hcris_v1996 = pd.concat([final_hcris_v1996, final_reports], ignore_index=True)

# Save the final dataset to the specified path
output_path = '/Users/ilsenovis/Documents/GitHub/ECON470HW2/data/output/final_hcris_v1996.csv'
final_hcris_v1996.to_csv(output_path, index=False)

print(f"Final dataset saved to {output_path}")

# Check for NaN counts in discharge and payment/charge variables
discharge_vars = ['tot_discharges', 'mcare_discharges', 'mcaid_discharges']
charge_payment_vars = ['tot_charges', 'tot_discounts', 'tot_operating_exp', 'ip_charges', 'icu_charges', 'ancillary_charges', 'tot_mcare_payment', 'secondary_mcare_payment']

final_data = pd.read_csv(output_path)

# Check data ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# NaN counts for discharges
print("\nNaN Counts for Discharge Variables:")
for var in discharge_vars:
    print(f"{var}: {final_data[var].isna().sum()} NaN values")

# NaN counts for charges/payments
print("\nNaN Counts for Charge and Payment Variables:")
for var in charge_payment_vars:
    print(f"{var}: {final_data[var].isna().sum()} NaN values")

