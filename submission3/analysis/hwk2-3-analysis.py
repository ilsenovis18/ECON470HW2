## Title:           ECON 470 Homework 2 Answers
## Author:          Ilse Novis
## Date Created:    2/17/2025
## Date Edited:     2/19/2025
## Description:     This file renders/runs Python code for the assignment

# Import libraries
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from scipy.spatial.distance import mahalanobis
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import statsmodels.api as sm
import matplotlib.ticker as ticker
import matplotlib
matplotlib.use('TkAgg')
import seaborn as sns
from statsmodels.formula.api import ols
from sklearn.neighbors import NearestNeighbors
from IPython.display import Markdown, display
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.DtypeWarning)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Read in data
hcris_data=pd.read_csv('/Users/ilsenovis/Documents/GitHub/ECON470HW2/data/output/HCRIS_Clean_Data.csv')
hcris_data_preclean=pd.read_csv('/Users/ilsenovis/Documents/GitHub/ECON470HW2/data/output/HCRIS_Data_preclean.csv')

# Convert 'fy_start' to datetime to extract the year
hcris_data_preclean['fy_start'] = pd.to_datetime(hcris_data_preclean['fy_start'], errors='coerce')
hcris_data_preclean['year'] = hcris_data_preclean['fy_start'].dt.year

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
hcris_v1996 = pd.read_csv('/Users/ilsenovis/Documents/GitHub/ECON470HW2/data/output/HCRIS_v1996.csv')
# Convert dates and extract year
hcris_v1996['fy_start'] = pd.to_datetime(hcris_v1996['fy_start'], errors='coerce')
hcris_v1996['year'] = hcris_v1996['fy_start'].dt.year

# Check available years
print(hcris_v1996['year'].value_counts().sort_index())

final_hcris_v1996 = pd.read_csv('/Users/ilsenovis/Documents/GitHub/ECON470HW2/data/output/HCRIS_v1996.csv')
final_hcris_v2010 = pd.read_csv('/Users/ilsenovis/Documents/GitHub/ECON470HW2/data/output/HCRIS_v2010.csv')

final_hcris = pd.concat([final_hcris_v1996, final_hcris_v2010], ignore_index=True)

final_hcris['fy_start'] = pd.to_datetime(final_hcris['fy_start'], errors='coerce')
final_hcris['year'] = final_hcris['fy_start'].dt.year

# Check if 2008 and 2009 exist
print(final_hcris['year'].value_counts().sort_index())

print(final_hcris[['provider_number', 'year']].drop_duplicates().groupby('year').count())

final_hcris = pd.concat([final_hcris_v1996, final_hcris_v2010], ignore_index=True)

final_hcris['fy_start'] = pd.to_datetime(final_hcris['fy_start'], errors='coerce')
final_hcris['year'] = final_hcris['fy_start'].dt.year

# Check if 2008 and 2009 exist
print(final_hcris['year'].value_counts().sort_index())


# Count hospitals that filed more than one report in the same year
hosp_counts = hcris_data_preclean.groupby(['provider_number', 'year']).size().reset_index(name='report_count')

print(hcris_data_preclean['year'].value_counts().sort_index())
print(hcris_data_preclean[hcris_data_preclean['year'].isin([2008, 2009])].head())
print(hosp_counts[hosp_counts['year'].isin([2008, 2009])])
print(hosp_counts.groupby('year')['report_count'].sum().reset_index())
dup_reports = hosp_counts[hosp_counts['report_count'] > 0]  # Change to include single reports too
print(dup_reports)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

dup_reports = hosp_counts[hosp_counts['report_count'] > 1]

# Count number of hospitals per year
hospitals_over_time = dup_reports.groupby('year')['provider_number'].nunique()
print(hospitals_over_time)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Merge datasets
merge_data96 = pd.read_csv('/Users/ilsenovis/Documents/GitHub/ECON470HW2/data/output/HCRIS_v1996.csv')
merge_data10 = pd.read_csv('/Users/ilsenovis/Documents/GitHub/ECON470HW2/data/output/HCRIS_v2010.csv')
merge_data = pd.concat([merge_data96, merge_data10], ignore_index=True)

# Convert date columns to datetime
merge_data['fy_start'] = pd.to_datetime(merge_data['fy_start'], errors='coerce')
merge_data['year'] = merge_data['fy_start'].dt.year

# --------------------------------------------------------------------------------------------------
# Q1: How many hospitals filed more than one report in the same year? Show as a line graph.
# --------------------------------------------------------------------------------------------------
# Count distinct providers
num_providers = merge_data['provider_number'].nunique()
print(f"Number of distinct providers: {num_providers}")

# Count hospitals with more than one report in the same year
hosp_count = merge_data.groupby(['provider_number', 'year']).size().reset_index(name='report_count')
multiple_reports = hosp_count[hosp_count['report_count'] > 1]

# Count number of hospitals per year
hosp_over_time = multiple_reports.groupby('year')['provider_number'].nunique()
print(hosp_over_time)





merge_data.groupby(['provider_number', 'year']).size().value_counts()
merge_data.info()

print(merge_data.groupby(['provider_number', 'year']).size().describe())

# Count duplicate reports per provider per year
dup_count = merge_data.groupby(['provider_number', 'year']).size().reset_index(name='total_reports')

# Count duplicate reports per year
dup_count['dup_report'] = dup_count['total_reports'] > 1
dup_count.info()

# Summarize duplicates per year
dup_summary = dup_count.groupby('year')['dup_report'].sum().reset_index()
print(dup_summary)

# Line plot for hospitals with multiple reports
plt.figure(figsize=(8, 5))
plt.plot(dup_summary['year'].values, dup_summary['dup_report'].values, marker='o', linestyle='-', color='blue')

# Set y-axis to log scale to handle large jumps
plt.xticks(range(2007, 2016))

plt.title('Figure 1: Number of Hospitals with Multiple Reports', fontsize=14)
plt.xlabel('Fiscal Year')
plt.ylabel('Number of Hospitals')
plt.ylim(0, 2000)
plt.grid(axis='y', color='gray', linestyle='--', alpha=0.5)

# Save line plot
output_dir = 'submission3/analysis'
plt.savefig(os.path.join(output_dir, 'multi_report_hospitals.png'))
print("Plot saved successfully!")

# --------------------------------------------------------------------------------------------------
# Q2: After removing/combining multiple reports, how many unique hospital IDs exist?
# --------------------------------------------------------------------------------------------------
dup_summary = dup_count.groupby('fy_year')['total_reports'].sum().reset_index()
hosp_count = data.groupby('fy_year').size().reset_index(name='hosp_count')
print(hosp_count)

# Count of unique hospitals per year
plt.figure(figsize=(8, 5))
plt.plot(hosp_count['fy_year'].values, hosp_count['hosp_count'].values,
         marker='o', linestyle='-', color='blue')
plt.title('Figure 2: Unqiue Hospital IDs', fontsize=14)
plt.xlabel('Fiscal Year')
plt.ylabel('Number of Hospital IDs')
plt.ylim(0, hosp_count['hosp_count'].max() * 1.1)
plt.grid(axis='y', color='gray', linestyle='--', alpha=0.5)

# Save line plot
output_dir = 'submission3/analysis'
plt.savefig(os.path.join(output_dir, 'unique_hospitals.png'))
print("Plot saved successfully!")

# --------------------------------------------------------------------------------------------------
# Q3: Distribution of total charges per year (Violin Plot)
# --------------------------------------------------------------------------------------------------
charge_data = data.copy()

# Drop rows where 'tot_charges' is NaN 
charge_data = charge_data.dropna(subset=['tot_charges'])

# Comput 1st and 99th percentiles
charge_data['tot_charges_low'] = charge_data.groupby('year')['tot_charges'].transform(lambda x: np.nanpercentile(x, 1))
charge_data['tot_charges_high'] = charge_data.groupby('year')['tot_charges'].transform(lambda x: np.nanpercentile(x, 99))

# Filter out extreme vlaues and missing data
charge_data = charge_data[
    (charge_data['tot_charges'] > charge_data['tot_charges_low']) &
    (charge_data['tot_charges'] < charge_data['tot_charges_high']) &
     charge_data['tot_charges'].notna() &
     (charge_data['year'] > 2008)
]

# Compute log of total charges
charge_data['log_charge'] = np.log(charge_data['tot_charges'])

# Prepare data for violin plot
years = sorted(charge_data['year'].unique())
charge_violin_data = [charge_data[charge_data['year'] == y]['log_charge'].dropna().values for y in years]

# Plot distribution of total charges
fig, ax = plt.subplots(figsize=(8, 5))
parts = ax.violinplot(charge_violin_data, positions=years, showmedians=False)

# Customize Violin Plot
for pc in parts['bodies']:
    pc.set_facecolor('gold')
    pc.set_alpha(0.3)

# Add median line
median = charge_data.groupby('year')['log_charge'].median()
plt.plot(years, median.values, marker='o', linestyle='-', color='blue', linewidth=2)

# Format plot
plt.title('Figure 3: Distribution of Total Charges', fontsize=14)
plt.xlabel('Fiscal Year')
plt.ylabel('Log $')
plt.grid(axis='y', color='gray', linestyle='--', alpha=0.5)

# Save line plot
output_dir = 'submission3/analysis'
plt.savefig(os.path.join(output_dir, 'tot_charges.png'))
print("Plot saved successfully!")

# --------------------------------------------------------------------------------------------------
# Q4: Distribution of estimated prices per year (Violin Plot)
# --------------------------------------------------------------------------------------------------
if not isinstance(data, pd.DataFrame):
    data = pd.read_csv(hcris_data)

# Price Calculation
data['discount_factor'] = 1 - data['tot_discounts'] / data['tot_charges']
data['price_num'] = (
    (data['ip_charges'] + data['icu_charges'] + data['ancillary_charges']) * 
    data['discount_factor']
) - data['tot_mcare_payment']
data['price_denom'] = data['tot_discharges'] - data['mcare_discharges']
data['price'] = data['price_num'] / data['price_denom']

# Replace invalid denominators (zero or NaN) with NaN
data['price_denom'] = np.where(data['price_denom'] > 0, data['price_denom'], np.nan)

# Compute price only for valid values
data['price'] = data['price_num'] / data['price_denom']

# Filtering the data
price_data = data[
    (data['price_denom'] > 100) &
    (~data['price_denom'].isna()) &
    (data['price_num'] > 0) &
    (~data['price_num'].isna()) &
    (data['price'] < 100000) &
    (data['beds'] > 30) &
    (~data['beds'].isna())
]

# Prepare for Violin plot
years = sorted(price_data['year'].unique())
price_violin_data = [price_data[price_data['year'] == y]['price'].dropna().values for y in years]

# Plot distribution of total charges
fig, ax = plt.subplots(figsize=(8, 5))
parts = ax.violinplot(price_violin_data, positions=years, showmedians=False)

# Customize violin plot
for pc in parts['bodies']:
    pc.set_facecolor('gold')
    pc.set_alpha(0.3)

# Add median line
median = price_data.groupby('year')['price'].median()
plt.plot(years, median.values, marker='o', linestyle='-', color='blue', linewidth=2)

# Format plot
plt.title('Figure 4: Distribution of Estimated Prices', fontsize=14)
plt.xlabel('Fiscal Year')
plt.ylabel('Total Price')
plt.grid(axis='y', color='gray', linestyle='--', alpha=0.5)

# Save line plot
output_dir = 'submission3/analysis'
plt.savefig(os.path.join(output_dir, 'price_distribution.png'))
print("Plot saved successfully!")

# --------------------------------------------------------------------------------------------------
# Q5: Calculate the average price among penalized versus non-penalized hospitals
# --------------------------------------------------------------------------------------------------
# Ensure 'penalty' is created as a float column using. loc
price_data = data.copy()
price_data.loc[:, 'penalty'] = ((price_data['hvbp_payment'].fillna(0) - price_data['hrrp_payment'].fillna(0)).abs().astype(float) < 9)

print(price_data.columns)

# Filter data for the year 2012
pen_data_2012 = price_data[price_data['year'] == 2012].copy()

# Compute Quartiles
beds_q1 = pen_data_2012['beds'].quantile(0.25)
beds_q2 = pen_data_2012['beds'].quantile(0.50)
beds_q3 = pen_data_2012['beds'].quantile(0.75)
beds_q4 = pen_data_2012['beds'].max()

# Assign bed quartiles
pen_data_2012['bed_quart'] = np.select(
    [
        pen_data_2012['beds'] < beds_q1,
        (pen_data_2012['beds'] >= beds_q1) & (pen_data_2012['beds'] < beds_q2),
        (pen_data_2012['beds'] >= beds_q2) & (pen_data_2012['beds'] < beds_q3),
        pen_data_2012['beds'] >= beds_q3
    ],
    [1, 2, 3, 4],
    default=0
)

# Filter out invalid quartile values
pen_data_2012 = pen_data_2012[pen_data_2012['bed_quart'] > 0]

# Compute average price by penalty status
avg_pen = pen_data_2012.groupby('penalty')['price'].mean().round(2)
print(avg_pen)

# --------------------------------------------------------------------------------------------------
# Q6: Split hospitals into quartiles based on bed size. Provide a table of the average price among treated/control groups for each quartile.
# --------------------------------------------------------------------------------------------------
# Compute the average price by treatment status (penalized vs. non-penalized) and quartile
quartile_avg_price = pen_data_2012.groupby(['bed_quart', 'penalty'])['price'].mean().unstack()

# Rename columns for clarity
quartile_avg_price.columns = ['Control (No Penalty)', 'Treated (Penalty)']
quartile_avg_price.index.name = 'Bed Quartile'
quartile_avg_price = quartile_avg_price.round(2)

# Display the final table
print("\nTable: Average Price by Treatment Status for Each Bed Size Quartile\n")
print(quartile_avg_price)

# --------------------------------------------------------------------------------------------------
# Q7: Use Different Estimators
# --------------------------------------------------------------------------------------------------
# Q7.A: Nearest Neighbor Matching (Inverse Variance and Mahalanobis Distance)
# Prepare data
match_data = pen_data_2012.copy()
match_data = match_data.dropna()

# Separate treated and control groups
treated = match_data[match_data['penalty'] == True]
control = match_data[match_data['penalty'] == False]

# Use bed quartiles as the matching variables
X_treated = treated[['bed_quart']].values
X_control = control[['bed_quart']].values

# Use inverse variance weighting for distance
bed_var = match_data.groupby('bed_quart')['price'].var().fillna(1)
inv_var_weights = 1 / bed_var.loc[treated['bed_quart']].values

# Perform NN Matching (1-to-1)
nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
nn.fit(X_control)
_, indices = nn.kneighbors(X_treated)

# Get matched control prices
matched_control_prices = control.iloc[indices.flatten()]['price'].values
treated_prices = treated['price'].values

# Compute ATE
ATE_nn = np.mean(treated_prices - matched_control_prices)
print(f"Nearest Neighbor Matching ATE: {ATE_nn:.2f}")

# Q7.B: Nearest Neighbor Matching using Mahalanobis Distance
# Prepare data
match_mah_data = pen_data_2012.copy()
match_mah_data = match_mah_data.dropna()

# Separate treated and control groups
treated_mah = match_mah_data[match_mah_data['penalty'] == True]
control_mah = match_mah_data[match_mah_data['penalty'] == False]

# Use bed quartiles as matching variables
X_mah_treated = treated[['bed_quart']].values
X_mah_control = control[['bed_quart']].values

# Compute inverse covariance matrix for Mahalonobis distance
cov_matrix = np.cov(match_mah_data[['bed_quart']].values.T, rowvar=False)
cov_matrix = np.atleast_2d(cov_matrix)
inv_cov_matrix = np.linalg.inv(cov_matrix)

# Compute Mahalanobis distance between each treated and control unit
dist_matrix = np.array([
    [mahalanobis(t, c, inv_cov_matrix) for c in X_mah_control]
    for t in X_mah_treated
])

# Perform NN Matching (1-to-1)
nn_indices = dist_matrix.argmin(axis=1)

# Get matched control prices
matched_mah_control_prices = control_mah.iloc[nn_indices]['price'].values
treated_mah_prices = treated_mah['price'].values

# Compute ATE
ATE_nn_mah = np.mean(treated_mah_prices - matched_mah_control_prices)
print(f"Nearest Neighbor Matching (Mahalanobis) ATE: {ATE_nn_mah:.2f}")

# Q7.C: Propesntiy Score Matching and Weighting
# Prepare data
ps_model_data = pen_data_2012.copy()
ps_model_data = ps_model_data.dropna()

# Fit log regression model to estimate PS 
logit_model = LogisticRegression()
logit_model.fit(ps_model_data[['beds', 'mcaid_discharges', 'ip_charges', 'mcare_discharges', 'tot_mcare_payment']], ps_model_data['penalty'])

# Compute PS scores (predicted possibilities)
ps_model_data['ps'] = logit_model.predict_proba(ps_model_data[['beds', 'mcaid_discharges', 'ip_charges', 'mcare_discharges', 'tot_mcare_payment']])[:, 1]

# Compute inverse probability weights (IPW)
ps_model_data['ipw'] = np.where(
    ps_model_data['penalty'] == 1,
    1 / ps_model_data['ps'], #For treated
    1 / (1- ps_model_data['ps']) #For control
)

# Compute weighte means for treated and control groups
mean_treated =  np.average(ps_model_data.loc[ps_model_data['penalty'] == 1, 'price'],
                           weights=ps_model_data.loc[ps_model_data['penalty'] ==1, 'ipw'])

mean_control = np.average(ps_model_data.loc[ps_model_data['penalty'] == 0, 'price'],
                          weights=ps_model_data.loc[ps_model_data['penalty'] == 0, 'ipw'])

# Compute ATE
ATE_ipw = mean_treated - mean_control

# Print results
print(f"Mean Price (Treated): {mean_treated:.2f}")
print(f"Mean Price (Control): {mean_control:.2f}")
print(f"ATE (IPW): {ATE_ipw:.2f}")

#Q7.D: One-step Regression for ATE Estimation
# Ensure relevant columns are numeric
reg_data = pen_data_2012

# Extract ATE estimate (coefficient of 'penalty')
ate_regression = reg_model.params['penalty']
print(f"Estimated ATE using One-Step Regression: {ate_regression:.2f}")





# Q7: Final Summary Table
# Dictionary to store ATE estimates
ate_results = {
    "Estimator": ["Nearest Neighbor Matching",
                  "Mahalanobis Distance Matching",
                  "Inverse Propensity Weighting"],
    "ATE Estimator": [ATE_nn, ATE_nn_mah, ATE_ipw]
}

# Convert to DF
ate_table = pd.DataFrame(ate_results).round(2)

print(ate_table)

# --------------------------------------------------------------------------------------------------
# Q8: With these different treatment effect estimators, are the results similar, identical, very different?
# --------------------------------------------------------------------------------------------------
print("Although I am getting no output right now, I would expect results to be different as each estimator makes different assumptions and uses different techniques to estimate the treatment effect. Nearest neighbor matching with Mahalonbis and inverse variance may produce closer estimates, while regression-based appraoches might differ if the linea model does not fully capture the data structure.")
# --------------------------------------------------------------------------------------------------
# Q9: Do you think you’ve estimated a causal effect of the penalty? Why or why not? (just a couple of sentences)
# --------------------------------------------------------------------------------------------------
print("I do not have any outputs right now, but I would assume that even if my estimates suggest a relationship between hospital penalities and pricing, it does not imply a causal effect. Since penalties were not randomly assigned and unobserved confounders might influence both penalty status and pricing, my estimates may suffer from selection bias.")
# --------------------------------------------------------------------------------------------------
# Q10: Briefly describe your experience working with these data (just a few sentences). Tell me one thing you learned and one thing that really aggravated or surprised you.
# --------------------------------------------------------------------------------------------------
print("One thing that really aggravated me was that the datasets didn't download correctly so it took a while to actually clean/fix the data before I could merge it into the final dataset. One thing that suprised me was the large difference in the charges from the hospitals versus the actual prices")

