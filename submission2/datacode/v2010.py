import pandas as pd
import warnings
warnings.simplefilter('ignore')

# Load 2010-2015 numeric data
for year in range(2010, 2016):
    print(f"\n📌 Checking data for year {year}...")

    numeric_2010 = pd.read_csv(
        f"/Users/ilsenovis/Documents/GitHub/ECON470HW2/data/input/HCRIS_v2010/hosp10_{year}_NMRC.CSV",
        names=['RPT_REC_NUM', 'WKSHT_CD', 'LINE_NUM', 'CLMN_NUM', 'ITM_VAL_NUM'],
        dtype=str
    )

    alpha_2010 = pd.read_csv(
        f"/Users/ilsenovis/Documents/GitHub/ECON470HW2/data/input/HCRIS_v2010/hosp10_{year}_ALPHA.CSV",
        names=['RPT_REC_NUM', 'WKSHT_CD', 'LINE_NUM', 'CLMN_NUM', 'ITM_VAL_NUM'],
        dtype=str
    )

    # Print unique worksheet codes to verify
    print("📌 Unique Worksheet Codes in Numeric Data:")
    print(numeric_2010['WKSHT_CD'].unique())

    print("\n📌 Unique Worksheet Codes in Alpha Data:")
    print(alpha_2010['WKSHT_CD'].unique())

    print("\n📌 Most Common Line Numbers in Numeric Data:")
    print(numeric_2010['LINE_NUM'].value_counts().head(10))

    print("\n📌 Most Common Line Numbers in Alpha Data:")
    print(alpha_2010['LINE_NUM'].value_counts().head(10))



# Define variables and locations
hcris_vars = [
    ('beds', 'A000000', '00100', '00200', 'numeric'),
    ('tot_charges', 'G300000', '02400', '00000', 'numeric'),
    ('tot_discounts', 'G300000', '02401', '00000', 'numeric'),
    ('tot_operating_exp', 'A000000', '00500', '00100', 'numeric'),
    ('ip_charges', 'A000000', '01000', '00000', 'numeric'),
    ('icu_charges', 'A000000', '03100', '00000', 'numeric'),
    ('ancillary_charges', 'A000000', '01800', '00000', 'numeric'),
    ('tot_discharges', 'S200001', '00200', '00300', 'alpha'),
    ('mcare_discharges', 'S200001', '00200', '00400', 'alpha'),
    ('mcaid_discharges', 'S200001', '00200', '00400', 'alpha'),
    ('tot_mcare_payment', 'G300000', '02400', '00000', 'numeric'),
    ('secondary_mcare_payment', 'G300000', '02401', '00000', 'numeric'),
    ('street', 'S200001', '00100', '00100', 'alpha'),
    ('city', 'S200001', '00200', '00100', 'alpha'),
    ('state', 'S200001', '00200', '00200', 'alpha'),
    ('zip', 'S200001', '00200', '00300', 'alpha'),
    ('county', 'S200001', '00200', '00400', 'alpha'),
    ('hvbp_payment', 'G300000', '02400', '00000', 'numeric'),
    ('hrrp_payment', 'G300000', '02401', '00000', 'numeric')
]


hcris_vars_df = pd.DataFrame(hcris_vars, columns=["variable", "WKSHT_CD", "LINE_NUM", "CLMN_NUM", "source"])

# Pull relevant data
final_hcris_v2010 = pd.DataFrame()

for year in range(2010, 2016):
    print(f"Processing year: {year}")
    hcris_alpha = pd.read_csv(f"/Users/ilsenovis/Documents/GitHub/ECON470HW2/data/input/HCRIS_v2010/hosp10_{year}_ALPHA.CSV", 
                              names=['RPT_REC_NUM','WKSHT_CD','LINE_NUM','CLMN_NUM','ITM_VAL_NUM'])
    hcris_numeric = pd.read_csv(f"/Users/ilsenovis/Documents/GitHub/ECON470HW2/data/input/HCRIS_v2010/hosp10_{year}_NMRC.CSV", 
                                names=['RPT_REC_NUM','WKSHT_CD','LINE_NUM','CLMN_NUM','ITM_VAL_NUM'])
    hcris_report = pd.read_csv(f"/Users/ilsenovis/Documents/GitHub/ECON470HW2/data/input/HCRIS_v2010/hosp10_{year}_RPT.CSV", 
                               names=['RPT_REC_NUM','PRVDR_CTRL_TYPE_CD','PRVDR_NUM','NPI','RPT_STUS_CD','FY_BGN_DT',
                                      'FY_END_DT','PROC_DT','INITL_RPT_SW','LAST_RPT_SW','TRNSMTL_NUM','FI_NUM',
                                      'ADR_VNDR_CD','FI_CREAT_DT','UTIL_CD','NPR_DT','SPEC_IND','FI_RCPT_DT'])
    
    final_reports = hcris_report[['RPT_REC_NUM', 'PRVDR_NUM', 'NPI', 'FY_BGN_DT', 'FY_END_DT', 'PROC_DT', 'FI_CREAT_DT', 'RPT_STUS_CD']]
    final_reports.columns = ['report', 'provider_number', 'npi', 'fy_start', 'fy_end', 'date_processed', 'date_created', 'status']
    final_reports['year'] = year
    
    for _, row in hcris_vars_df.iterrows():
        hcris_data = hcris_numeric if row['source'] == 'numeric' else hcris_alpha
        val = hcris_data[
            (hcris_data['WKSHT_CD'] == row['WKSHT_CD']) &
            (hcris_data['LINE_NUM'] == row['LINE_NUM']) &
            (hcris_data['CLMN_NUM'] == row['CLMN_NUM'])
        ][['RPT_REC_NUM', 'ITM_VAL_NUM']]

        val.columns = ['report', row['variable']]

        if row['source'] == 'numeric':
            val[row['variable']] = pd.to_numeric(val[row['variable']], errors='coerce')

        final_reports = final_reports.merge(val, on='report', how='left')

        print(f"Extracting {row['variable']} - Found {val.shape[0]} rows")
        print(val.head(5))
    final_hcris_v2010 = pd.concat([final_hcris_v2010, final_reports], ignore_index=True)

final_hcris_v2010.to_csv("data/output/HCRIS_v2010.csv", index=False)