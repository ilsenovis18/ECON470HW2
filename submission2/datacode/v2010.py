import pandas as pd
import warnings
warnings.simplefilter('ignore')

# Define variables and locations (converted to DataFrame)
hcris_vars = pd.DataFrame([
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
], columns=["variable", "WKSHT_CD", "LINE_NUM", "CLMN_NUM", "source"])

# Initialize final DataFrame
final_hcris_v2010 = None

for year in range(2012, 2016):
    print(f"Processing year: {year}")
    alpha_path = f"/Users/ilsenovis/Documents/GitHub/ECON470HW2/data/input/HCRIS_v2010/HospitalFY{year}/hosp10_{year}_ALPHA.CSV"
    numeric_path = f"/Users/ilsenovis/Documents/GitHub/ECON470HW2/data/input/HCRIS_v2010/HospitalFY{year}/hosp10_{year}_NMRC.CSV"
    report_path = f"/Users/ilsenovis/Documents/GitHub/ECON470HW2/data/input/HCRIS_v2010/HospitalFY{year}/hosp10_{year}_RPT.CSV"

    col_names = ['RPT_REC_NUM', 'WKSHT_CD', 'LINE_NUM', 'CLMN_NUM', 'ITM_VAL_NUM']
    hcris_alpha = pd.read_csv(alpha_path, names=col_names, dtype=str)
    hcris_numeric = pd.read_csv(numeric_path, names=col_names, dtype=str)
    hcris_report = pd.read_csv(report_path, names=['RPT_REC_NUM', 'PRVDR_CTRL_TYPE_CD', 'PRVDR_NUM', 'NPI', 'RPT_STUS_CD', 'FY_BGN_DT', 'FY_END_DT', 'PROC_DT',
                                                   'INITL_RPT_SW', 'LAST_RPT_SW', 'TRNSMTL_NUM', 'FI_NUM',
                                                   'ADR_VNDR_CD', 'FI_CREAT_DT', 'UTIL_CD', 'NPR_DT',
                                                   'SPEC_IND', 'FI_RCPT_DT'], dtype=str)
    
    # Select relevant columns from the report
    final_reports = hcris_report[['RPT_REC_NUM', 'PRVDR_NUM', 'NPI', 'FY_BGN_DT', 'FY_END_DT', 'PROC_DT',
                                  'FI_CREAT_DT', 'RPT_STUS_CD']]
    final_reports.columns = ['report', 'provider_number', 'npi', 'fy_start', 'fy_end', 'date_processed',
                             'date_created', 'status']
    final_reports['year'] = year

    # Extract variables
    for _, row in hcris_vars.iterrows():
        hcris_data = hcris_numeric if row['source'] == 'numeric' else hcris_alpha
        val = hcris_data[(hcris_data['WKSHT_CD'] == row['WKSHT_CD']) &
            (hcris_data['LINE_NUM'] == row['LINE_NUM']) &
            (hcris_data['CLMN_NUM'] == row['CLMN_NUM'])][['RPT_REC_NUM', 'ITM_VAL_NUM']]
        val.columns = ['report', row['variable']]
        final_reports = final_reports.merge(val, on='report', how='left')
        if row['source'] == 'numeric':
            final_reports[row['variable']] = final_reports[row['variable']].astype(float)

        if final_hcris_v2010 is None:
            final_hcris_v2010 = final_reports
        else:
            final_hcris_v2010 = pd.concat([final_hcris_v2010, final_reports], ignore_index=True) 

# Save final dataset
final_hcris_v2010.to_csv("data/output/HCRIS_v2010.csv", index=False)

print("Data processing complete. File saved as 'HCRIS_v2010.csv'.")