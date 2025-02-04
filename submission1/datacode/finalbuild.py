import pandas as pd
import os
from datetime import datetime

# Load both datasets (1996 and 2010)
v1996_path = '/Users/ilsenovis/Documents/GitHub/ECON470HW2/data/output/final_hcris_v1996.csv'
v2010_path = '/Users/ilsenovis/Documents/GitHub/ECON470HW2/data/output/final_hcris_v2010.csv'

final_hcris_v1996 = pd.read_csv(v1996_path)
final_hcris_v2010 = pd.read_csv(v2010_path)

# Create missing variables for columns introduced in v2010 (like hvbp_payment, hrrp_payment)
final_hcris_v1996['hvbp_payment'] = None
final_hcris_v1996['hrrp_payment'] = None

# Drop the 'year' column if it already exists in either dataset to avoid duplication
final_hcris_v1996 = final_hcris_v1996.drop(columns=['year'], errors='ignore')
final_hcris_v2010 = final_hcris_v2010.drop(columns=['year'], errors='ignore')

# Combine v1996 and v2010 datasets, and sort by provider_number/year
final_hcris = pd.concat([final_hcris_v1996, final_hcris_v2010], ignore_index=True)

# Rename 'PRVDR_NUM' to 'provider_number' to match the expected column name
final_hcris.rename(columns={'PRVDR_NUM': 'provider_number'}, inplace=True)

# Inspect the columns before trying to use the 'fy_end' column
print(final_hcris.columns)

# Convert date columns to datetime format
final_hcris['fy_end'] = pd.to_datetime(final_hcris['FY_END_DT'], format='%m/%d/%Y', errors='coerce')
final_hcris['fy_start'] = pd.to_datetime(final_hcris['FY_BGN_DT'], format='%m/%d/%Y', errors='coerce')
final_hcris['date_processed'] = pd.to_datetime(final_hcris['FI_RCPT_DT'], format='%m/%d/%Y', errors='coerce')
final_hcris['date_created'] = pd.to_datetime(final_hcris['FI_CREAT_DT'], format='%m/%d/%Y', errors='coerce')

# Ensure that numerical columns are positive, handling NaN values
final_hcris['tot_discounts'] = final_hcris['tot_discounts'].abs()
final_hcris['hrrp_payment'] = final_hcris['hrrp_payment'].fillna(0).abs()

# Create fiscal year from fy_end
final_hcris['fyear'] = final_hcris['fy_end'].dt.year

# Sort and arrange the data by 'provider_number' and 'fyear'
final_hcris = final_hcris.sort_values(by=['provider_number', 'fyear'])

# Create a count of hospitals/provider_number by year
report_count_by_year = final_hcris.groupby('fyear').size().reset_index(name='count')

# Create count of reports by hospital fiscal year
final_hcris['total_reports'] = final_hcris.groupby(['provider_number', 'fyear'])['provider_number'].transform('count')

# Create a running total of reports for each provider
final_hcris['report_number'] = final_hcris.groupby(['provider_number', 'fyear']).cumcount() + 1

# Identify hospitals with only one report per fiscal year
unique_hcris1 = final_hcris[final_hcris['total_reports'] == 1]
unique_hcris1 = unique_hcris1.drop(columns=['total_reports', 'report_number', 'npi', 'status'], errors='ignore')
unique_hcris1['source'] = 'unique reports'

# Print the columns to check
print(unique_hcris1.columns)

# Identify hospitals with multiple reports per fiscal year
duplicate_hcris = final_hcris[final_hcris['total_reports'] > 1]
duplicate_hcris.loc[:, 'time_diff'] = duplicate_hcris['fy_end'] - duplicate_hcris['fy_start']

# Calculate elapsed time between fy_start and fy_end for hospitals with multiple reports
duplicate_hcris.loc[:, 'total_days'] = duplicate_hcris.groupby(['provider_number', 'fyear'])['time_diff'].transform('sum')

# Convert the 'total_days' to an integer number of days before comparing to 370
duplicate_hcris['total_days'] = duplicate_hcris['total_days'].dt.days  # Convert to days as an integer

# Hospitals where total elapsed time is less than 370 days, take the total of the two reports
unique_hcris2 = duplicate_hcris[duplicate_hcris['total_days'] < 370]
unique_hcris2 = unique_hcris2.groupby(['provider_number', 'fyear']).agg({
    'beds': 'max',
    'tot_charges': 'sum',
    'tot_discounts': 'sum',
    'tot_operating_exp': 'sum',
    'ip_charges': 'sum',
    'icu_charges': 'sum',
    'ancillary_charges': 'sum',
    'tot_discharges': 'sum',
    'mcare_discharges': 'sum',
    'mcaid_discharges': 'sum',
    'tot_mcare_payment': 'sum',
    'secondary_mcare_payment': 'sum',
    'hvbp_payment': 'sum',
    'hrrp_payment': 'sum',
    'fy_start': 'min',
    'fy_end': 'max',
    'date_processed': 'max',
    'date_created': 'min',
    'street': 'first',
    'city': 'first',
    'state': 'first',
    'zip': 'first',
    'county': 'first'
}).reset_index()
unique_hcris2['source'] = 'total for year'

# Hospitals with more than one report and elapsed time exceeding 370 days
duplicate_hcris2 = duplicate_hcris[duplicate_hcris['total_days'] >= 370]

# Hospitals that seem to have changed their fiscal year
duplicate_hcris2['total_days'] = duplicate_hcris2['total_days'].astype(int)
duplicate_hcris2['time_diff'] = duplicate_hcris2['time_diff'].astype(int)

# Apply weighted averages for values
unique_hcris3 = duplicate_hcris2.groupby(['provider_number', 'fyear']).agg({
    'tot_charges': 'sum',
    'tot_discounts': 'sum',
    'tot_operating_exp': 'sum',
    'ip_charges': 'sum',
    'icu_charges': 'sum',
    'ancillary_charges': 'sum',
    'tot_discharges': 'sum',
    'mcare_discharges': 'sum',
    'mcaid_discharges': 'sum',
    'tot_mcare_payment': 'sum',
    'secondary_mcare_payment': 'sum',
    'hvbp_payment': 'sum',
    'hrrp_payment': 'sum',
    'beds': 'max',
    'fy_start': 'min',
    'fy_end': 'max',
    'date_processed': 'max',
    'date_created': 'min',
    'street': 'first',
    'city': 'first',
    'state': 'first',
    'zip': 'first',
    'county': 'first'
}).reset_index()
unique_hcris3['source'] = 'weighted_average'

# Combine all unique and duplicate data
final_hcris_data = pd.concat([unique_hcris1, unique_hcris2, unique_hcris3])

# Save the final data
final_hcris_data = final_hcris_data.rename(columns={'fyear': 'year'})
final_hcris_data = final_hcris_data.sort_values(by=['provider_number', 'year'])

# Save final dataset
final_hcris_data.to_csv('/Users/ilsenovis/Documents/GitHub/ECON470HW2/data/output/final_hcris_combined.csv', index=False)

print("Final dataset saved successfully.")

"""import pandas as pd
import os
from datetime import datetime

# Load both datasets (1996 and 2010)
v1996_path = '/Users/ilsenovis/Documents/GitHub/ECON470HW2/data/output/final_hcris_v1996.csv'
v2010_path = '/Users/ilsenovis/Documents/GitHub/ECON470HW2/data/output/final_hcris_v2010.csv'

final_hcris_v1996 = pd.read_csv(v1996_path)
final_hcris_v2010 = pd.read_csv(v2010_path)

# Create missing variables for columns introduced in v2010 (like hvbp_payment, hrrp_payment)
final_hcris_v1996['hvbp_payment'] = None
final_hcris_v1996['hrrp_payment'] = None

# Combine v1996 and v2010 datasets, and sort by provider_number/year
final_hcris = pd.concat([final_hcris_v1996, final_hcris_v2010], ignore_index=True)

# Rename 'PRVDR_NUM' to 'provider_number' to match the expected column name
final_hcris.rename(columns={'PRVDR_NUM': 'provider_number'}, inplace=True)

# Inspect the columns before trying to use the 'fy_end' column
print(final_hcris.columns)

# Convert date columns to datetime format
final_hcris['fy_end'] = pd.to_datetime(final_hcris['FY_END_DT'], format='%m/%d/%Y', errors='coerce')
final_hcris['fy_start'] = pd.to_datetime(final_hcris['FY_BGN_DT'], format='%m/%d/%Y', errors='coerce')
final_hcris['date_processed'] = pd.to_datetime(final_hcris['FI_RCPT_DT'], format='%m/%d/%Y', errors='coerce')
final_hcris['date_created'] = pd.to_datetime(final_hcris['FI_CREAT_DT'], format='%m/%d/%Y', errors='coerce')

# Ensure that numerical columns are positive, handling NaN values
final_hcris['tot_discounts'] = final_hcris['tot_discounts'].abs()
final_hcris['hrrp_payment'] = final_hcris['hrrp_payment'].fillna(0).abs()

# Create fiscal year from fy_end
final_hcris['fyear'] = final_hcris['fy_end'].dt.year

# Sort and arrange the data by 'provider_number' and 'fyear'
final_hcris = final_hcris.sort_values(by=['provider_number', 'fyear'])

# Create a count of hospitals/provider_number by year
report_count_by_year = final_hcris.groupby('fyear').size().reset_index(name='count')

# Create count of reports by hospital fiscal year
final_hcris['total_reports'] = final_hcris.groupby(['provider_number', 'fyear'])['provider_number'].transform('count')

# Create a running total of reports for each provider
final_hcris['report_number'] = final_hcris.groupby(['provider_number', 'fyear']).cumcount() + 1

# Identify hospitals with only one report per fiscal year
unique_hcris1 = final_hcris[final_hcris['total_reports'] == 1]
unique_hcris1 = unique_hcris1.drop(columns=['total_reports', 'report_number', 'npi', 'status'], errors='ignore')
unique_hcris1['source'] = 'unique reports'

# Print the columns to check
print(unique_hcris1.columns)

# Identify hospitals with multiple reports per fiscal year
duplicate_hcris = final_hcris[final_hcris['total_reports'] > 1]
duplicate_hcris.loc[:, 'time_diff'] = duplicate_hcris['fy_end'] - duplicate_hcris['fy_start']

# Calculate elapsed time between fy_start and fy_end for hospitals with multiple reports
duplicate_hcris.loc[:, 'total_days'] = duplicate_hcris.groupby(['provider_number', 'fyear'])['time_diff'].transform('sum')

# Convert the 'total_days' to an integer number of days before comparing to 370
duplicate_hcris['total_days'] = duplicate_hcris['total_days'].dt.days  # Convert to days as an integer

# Hospitals where total elapsed time is less than 370 days, take the total of the two reports
unique_hcris2 = duplicate_hcris[duplicate_hcris['total_days'] < 370]
unique_hcris2 = unique_hcris2.groupby(['provider_number', 'fyear']).agg({
    'beds': 'max',
    'tot_charges': 'sum',
    'tot_discounts': 'sum',
    'tot_operating_exp': 'sum',
    'ip_charges': 'sum',
    'icu_charges': 'sum',
    'ancillary_charges': 'sum',
    'tot_discharges': 'sum',
    'mcare_discharges': 'sum',
    'mcaid_discharges': 'sum',
    'tot_mcare_payment': 'sum',
    'secondary_mcare_payment': 'sum',
    'hvbp_payment': 'sum',
    'hrrp_payment': 'sum',
    'fy_start': 'min',
    'fy_end': 'max',
    'date_processed': 'max',
    'date_created': 'min',
    'street': 'first',
    'city': 'first',
    'state': 'first',
    'zip': 'first',
    'county': 'first'
}).reset_index()
unique_hcris2['source'] = 'total for year'

# Hospitals with more than one report and elapsed time exceeding 370 days
duplicate_hcris2 = duplicate_hcris[duplicate_hcris['total_days'] >= 370]

# Hospitals that seem to have changed their fiscal year
duplicate_hcris2['total_days'] = duplicate_hcris2['total_days'].astype(int)
duplicate_hcris2['time_diff'] = duplicate_hcris2['time_diff'].astype(int)

# Apply weighted averages for values
unique_hcris3 = duplicate_hcris2.groupby(['provider_number', 'fyear']).agg({
    'tot_charges': 'sum',
    'tot_discounts': 'sum',
    'tot_operating_exp': 'sum',
    'ip_charges': 'sum',
    'icu_charges': 'sum',
    'ancillary_charges': 'sum',
    'tot_discharges': 'sum',
    'mcare_discharges': 'sum',
    'mcaid_discharges': 'sum',
    'tot_mcare_payment': 'sum',
    'secondary_mcare_payment': 'sum',
    'hvbp_payment': 'sum',
    'hrrp_payment': 'sum',
    'beds': 'max',
    'fy_start': 'min',
    'fy_end': 'max',
    'date_processed': 'max',
    'date_created': 'min',
    'street': 'first',
    'city': 'first',
    'state': 'first',
    'zip': 'first',
    'county': 'first'
}).reset_index()
unique_hcris3['source'] = 'weighted_average'

# Combine all unique and duplicate data
final_hcris_data = pd.concat([unique_hcris1, unique_hcris2, unique_hcris3])

# Save the final data
final_hcris_data = final_hcris_data.rename(columns={'fyear': 'year'})
final_hcris_data = final_hcris_data.sort_values(by=['provider_number', 'year'])

# Save final dataset
final_hcris_data.to_csv('/Users/ilsenovis/Documents/GitHub/ECON470HW2/data/output/final_hcris_combined.csv', index=False)

print("Final dataset saved successfully.")"""

"""import pandas as pd
import os
from datetime import datetime

# Load both datasets (1996 and 2010)
v1996_path = '/Users/ilsenovis/Documents/GitHub/ECON470HW2/data/output/final_hcris_v1996.csv'
v2010_path = '/Users/ilsenovis/Documents/GitHub/ECON470HW2/data/output/final_hcris_v2010.csv'

final_hcris_v1996 = pd.read_csv(v1996_path)
final_hcris_v2010 = pd.read_csv(v2010_path)

# Create missing variables for columns introduced in v2010 (like hvbp_payment, hrrp_payment)
final_hcris_v1996['hvbp_payment'] = None
final_hcris_v1996['hrrp_payment'] = None

# Combine v1996 and v2010 datasets, and sort by provider_number/year
final_hcris = pd.concat([final_hcris_v1996, final_hcris_v2010], ignore_index=True)

# Rename 'PRVDR_NUM' to 'provider_number' to match the expected column name
final_hcris.rename(columns={'PRVDR_NUM': 'provider_number'}, inplace=True)

# Inspect the columns before trying to use the 'fy_end' column
print(final_hcris.columns)

# Convert date columns to datetime format
final_hcris['fy_end'] = pd.to_datetime(final_hcris['FY_END_DT'], format='%m/%d/%Y', errors='coerce')
final_hcris['fy_start'] = pd.to_datetime(final_hcris['FY_BGN_DT'], format='%m/%d/%Y', errors='coerce')
final_hcris['date_processed'] = pd.to_datetime(final_hcris['FI_RCPT_DT'], format='%m/%d/%Y', errors='coerce')
final_hcris['date_created'] = pd.to_datetime(final_hcris['FI_CREAT_DT'], format='%m/%d/%Y', errors='coerce')

# Ensure that numerical columns are positive, handling NaN values
final_hcris['tot_discounts'] = final_hcris['tot_discounts'].abs()
final_hcris['hrrp_payment'] = final_hcris['hrrp_payment'].fillna(0).abs()

# Create fiscal year from fy_end
final_hcris['fyear'] = final_hcris['fy_end'].dt.year

# Sort and arrange the data by 'provider_number' and 'fyear'
final_hcris = final_hcris.sort_values(by=['provider_number', 'fyear'])

# Create a count of hospitals/provider_number by year
report_count_by_year = final_hcris.groupby('fyear').size().reset_index(name='count')

# Create count of reports by hospital fiscal year
final_hcris['total_reports'] = final_hcris.groupby(['provider_number', 'fyear'])['provider_number'].transform('count')

# Create a running total of reports for each provider
final_hcris['report_number'] = final_hcris.groupby(['provider_number', 'fyear']).cumcount() + 1

# Identify hospitals with only one report per fiscal year
unique_hcris1 = final_hcris[final_hcris['total_reports'] == 1]
unique_hcris1 = unique_hcris1.drop(columns=['total_reports', 'report_number', 'npi', 'status'], errors='ignore')
unique_hcris1['source'] = 'unique reports'

# Identify hospitals with multiple reports per fiscal year
duplicate_hcris = final_hcris[final_hcris['total_reports'] > 1]
duplicate_hcris['time_diff'] = duplicate_hcris['fy_end'] - duplicate_hcris['fy_start']

# Calculate elapsed time between fy_start and fy_end for hospitals with multiple reports
duplicate_hcris['total_days'] = duplicate_hcris.groupby(['provider_number', 'fyear'])['time_diff'].transform('sum')

# Hospitals where total elapsed time is less than 370 days, take the total of the two reports
unique_hcris2 = duplicate_hcris[duplicate_hcris['total_days'] < 370]
unique_hcris2 = unique_hcris2.groupby(['provider_number', 'fyear']).agg({
    'beds': 'max',
    'tot_charges': 'sum',
    'tot_discounts': 'sum',
    'tot_operating_exp': 'sum',
    'ip_charges': 'sum',
    'icu_charges': 'sum',
    'ancillary_charges': 'sum',
    'tot_discharges': 'sum',
    'mcare_discharges': 'sum',
    'mcaid_discharges': 'sum',
    'tot_mcare_payment': 'sum',
    'secondary_mcare_payment': 'sum',
    'hvbp_payment': 'sum',
    'hrrp_payment': 'sum',
    'fy_start': 'min',
    'fy_end': 'max',
    'date_processed': 'max',
    'date_created': 'min',
    'street': 'first',
    'city': 'first',
    'state': 'first',
    'zip': 'first',
    'county': 'first'
}).reset_index()
unique_hcris2['source'] = 'total for year'

# Hospitals with more than one report and elapsed time exceeding 370 days
duplicate_hcris2 = duplicate_hcris[duplicate_hcris['total_days'] >= 370]

# Hospitals that seem to have changed their fiscal year
duplicate_hcris2['total_days'] = duplicate_hcris2['total_days'].astype(int)
duplicate_hcris2['time_diff'] = duplicate_hcris2['time_diff'].astype(int)

# Apply weighted averages for values
unique_hcris3 = duplicate_hcris2.groupby(['provider_number', 'fyear']).agg({
    'tot_charges': 'sum',
    'tot_discounts': 'sum',
    'tot_operating_exp': 'sum',
    'ip_charges': 'sum',
    'icu_charges': 'sum',
    'ancillary_charges': 'sum',
    'tot_discharges': 'sum',
    'mcare_discharges': 'sum',
    'mcaid_discharges': 'sum',
    'tot_mcare_payment': 'sum',
    'secondary_mcare_payment': 'sum',
    'hvbp_payment': 'sum',
    'hrrp_payment': 'sum',
    'beds': 'max',
    'fy_start': 'min',
    'fy_end': 'max',
    'date_processed': 'max',
    'date_created': 'min',
    'street': 'first',
    'city': 'first',
    'state': 'first',
    'zip': 'first',
    'county': 'first'
}).reset_index()
unique_hcris3['source'] = 'weighted_average'

# Combine all unique and duplicate data
final_hcris_data = pd.concat([unique_hcris1, unique_hcris2, unique_hcris3])

# Save the final data
final_hcris_data = final_hcris_data.rename(columns={'fyear': 'year'})
final_hcris_data = final_hcris_data.sort_values(by=['provider_number', 'year'])

# Save final dataset
final_hcris_data.to_csv('/Users/ilsenovis/Documents/GitHub/ECON470HW2/data/output/final_hcris_combined.csv', index=False)

print("Final dataset saved successfully.")"""

"""import pandas as pd
import os
from datetime import datetime

# Load both datasets (1996 and 2010)
v1996_path = '/Users/ilsenovis/Documents/GitHub/ECON470HW2/data/output/final_hcris_v1996.csv'
v2010_path = '/Users/ilsenovis/Documents/GitHub/ECON470HW2/data/output/final_hcris_v2010.csv'

final_hcris_v1996 = pd.read_csv(v1996_path)
final_hcris_v2010 = pd.read_csv(v2010_path)

# Create missing variables for columns introduced in v2010 (like hvbp_payment, hrrp_payment)
final_hcris_v1996['hvbp_payment'] = None
final_hcris_v1996['hrrp_payment'] = None

# Combine v1996 and v2010 datasets, and sort by provider_number/year
final_hcris = pd.concat([final_hcris_v1996, final_hcris_v2010], ignore_index=True)

# Rename 'PRVDR_NUM' to 'provider_number' to match the expected column name
final_hcris.rename(columns={'PRVDR_NUM': 'provider_number'}, inplace=True)

# Inspect the columns before trying to use the 'fy_end' column
print(final_hcris.columns)

# Convert date columns to datetime format
final_hcris['fy_end'] = pd.to_datetime(final_hcris['FY_END_DT'], format='%m/%d/%Y', errors='coerce')
final_hcris['fy_start'] = pd.to_datetime(final_hcris['FY_BGN_DT'], format='%m/%d/%Y', errors='coerce')
final_hcris['date_processed'] = pd.to_datetime(final_hcris['FI_RCPT_DT'], format='%m/%d/%Y', errors='coerce')
final_hcris['date_created'] = pd.to_datetime(final_hcris['FI_CREAT_DT'], format='%m/%d/%Y', errors='coerce')

# Ensure that numerical columns are positive, handling NaN values
final_hcris['tot_discounts'] = final_hcris['tot_discounts'].abs()
final_hcris['hrrp_payment'] = final_hcris['hrrp_payment'].fillna(0).abs()

# Create fiscal year from fy_end
final_hcris['fyear'] = final_hcris['fy_end'].dt.year

# Sort and arrange the data by 'provider_number' and 'fyear'
final_hcris = final_hcris.sort_values(by=['provider_number', 'fyear'])

# Create a count of hospitals/provider_number by year
report_count_by_year = final_hcris.groupby('fyear').size().reset_index(name='count')

# Create count of reports by hospital fiscal year
final_hcris['total_reports'] = final_hcris.groupby(['provider_number', 'fyear'])['provider_number'].transform('count')

# Create a running total of reports for each provider
final_hcris['report_number'] = final_hcris.groupby(['provider_number', 'fyear']).cumcount() + 1

# Identify hospitals with only one report per fiscal year
unique_hcris1 = final_hcris[final_hcris['total_reports'] == 1]
unique_hcris1 = unique_hcris1.drop(columns=['total_reports', 'report_number', 'npi', 'status'])
unique_hcris1['source'] = 'unique reports'

# Identify hospitals with only one report per fiscal year
unique_hcris1 = final_hcris[final_hcris['total_reports'] == 1]
unique_hcris1['source'] = 'unique reports'

# Print the columns to check
print(unique_hcris1.columns)

# Drop columns only if they exist
columns_to_drop = ['total_reports', 'report_number', 'npi', 'status']
unique_hcris1 = unique_hcris1.drop(columns=[col for col in columns_to_drop if col in unique_hcris1.columns], errors='ignore')

# Identify hospitals with multiple reports per fiscal year
duplicate_hcris = final_hcris[final_hcris['total_reports'] > 1]
duplicate_hcris['time_diff'] = duplicate_hcris['fy_end'] - duplicate_hcris['fy_start']

# Calculate elapsed time between fy_start and fy_end for hospitals with multiple reports
duplicate_hcris['total_days'] = duplicate_hcris.groupby(['provider_number', 'fyear'])['time_diff'].transform('sum')

# Hospitals where total elapsed time is less than 370 days, take the total of the two reports
unique_hcris2 = duplicate_hcris[duplicate_hcris['total_days'] < 370]
unique_hcris2 = unique_hcris2.groupby(['provider_number', 'fyear']).agg({
    'beds': 'max',
    'tot_charges': 'sum',
    'tot_discounts': 'sum',
    'tot_operating_exp': 'sum',
    'ip_charges': 'sum',
    'icu_charges': 'sum',
    'ancillary_charges': 'sum',
    'tot_discharges': 'sum',
    'mcare_discharges': 'sum',
    'mcaid_discharges': 'sum',
    'tot_mcare_payment': 'sum',
    'secondary_mcare_payment': 'sum',
    'hvbp_payment': 'sum',
    'hrrp_payment': 'sum',
    'fy_start': 'min',
    'fy_end': 'max',
    'date_processed': 'max',
    'date_created': 'min',
    'street': 'first',
    'city': 'first',
    'state': 'first',
    'zip': 'first',
    'county': 'first'
}).reset_index()
unique_hcris2['source'] = 'total for year'

# Hospitals with more than one report and elapsed time exceeding 370 days
duplicate_hcris2 = duplicate_hcris[duplicate_hcris['total_days'] >= 370]

# Hospitals that seem to have changed their fiscal year
duplicate_hcris2['total_days'] = duplicate_hcris2['total_days'].astype(int)
duplicate_hcris2['time_diff'] = duplicate_hcris2['time_diff'].astype(int)

# Apply weighted averages for values
unique_hcris3 = duplicate_hcris2.groupby(['provider_number', 'fyear']).agg({
    'tot_charges': 'sum',
    'tot_discounts': 'sum',
    'tot_operating_exp': 'sum',
    'ip_charges': 'sum',
    'icu_charges': 'sum',
    'ancillary_charges': 'sum',
    'tot_discharges': 'sum',
    'mcare_discharges': 'sum',
    'mcaid_discharges': 'sum',
    'tot_mcare_payment': 'sum',
    'secondary_mcare_payment': 'sum',
    'hvbp_payment': 'sum',
    'hrrp_payment': 'sum',
    'beds': 'max',
    'fy_start': 'min',
    'fy_end': 'max',
    'date_processed': 'max',
    'date_created': 'min',
    'street': 'first',
    'city': 'first',
    'state': 'first',
    'zip': 'first',
    'county': 'first'
}).reset_index()
unique_hcris3['source'] = 'weighted_average'

# Combine all unique and duplicate data
final_hcris_data = pd.concat([unique_hcris1, unique_hcris2, unique_hcris3])

# Save the final data
final_hcris_data = final_hcris_data.rename(columns={'fyear': 'year'})
final_hcris_data = final_hcris_data.sort_values(by=['provider_number', 'year'])

# Save final dataset
final_hcris_data.to_csv('/Users/ilsenovis/Documents/GitHub/ECON470HW2/data/output/final_hcris_combined.csv', index=False)

print("Final dataset saved successfully.")"""