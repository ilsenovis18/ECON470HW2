## Title:           ECON 470 Homework 2 Answers
## Author:          Ilse Novis
## Date Created:    2/7/2025
## Date Edited:     2/17/2025
## Description:     This file renders/runs Python code for the assignment

# Import libraries
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
matplotlib.use('TkAgg')
import seaborn as sns
from statsmodels.formula.api import ols
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from IPython.display import Markdown, display
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.DtypeWarning)

# Load and inspect cleaned HCRIS dataset
hcris_data = pd.read_csv('/Users/ilsenovis/Documents/GitHub/ECON470HW2/data/output/HCRIS_Data.csv')

print(hcris_data.head())
print("Columns in dataset:", hcris_data.columns)

# Ensure 'fyear' is properly set as an integer
hcris_data['fyear'] = pd.to_numeric(hcris_data['fyear'], errors='coerce').astype('Int64')
print("Unique years in dataset:", sorted(hcris_data['fyear'].dropna().unique()))

# --------------------------------------------------------------------------------------------------
# Q1: How many hospitals filed more than one report in the same year? Show as a line graph.
# --------------------------------------------------------------------------------------------------

multi_report_hospitals = hcris_data.groupby(['fyear', 'provider_number']).size().reset_index(name='report_count')
multi_report_counts = multi_report_hospitals.groupby('fyear')['report_count'].apply(lambda x: (x > 1).sum())

# Create 'graph_output' directory if it doesn't exist
os.makedirs('graph_output', exist_ok=True)

# Line plot for hospitals with multiple reports
plt.figure(figsize=(10, 6))
plt.plot(multi_report_counts.index, multi_report_counts.values, marker='o')
plt.title('Number of Hospitals with Multiple Reports Over Time')
plt.xlabel('Fiscal Year')
plt.ylabel('Number of Hospitals')
plt.grid()
plt.savefig('graph_output/multi_report_hospitals.png')
print("Plot saved successfully!")

# --------------------------------------------------------------------------------------------------
# Q2: After removing/combining multiple reports, how many unique hospital IDs exist?
# --------------------------------------------------------------------------------------------------

unique_hospitals = hcris_data['provider_number'].nunique()
print(f"Total Unique Hospitals (Medicare Provider Number): {unique_hospitals}")

# --------------------------------------------------------------------------------------------------
# Q3: Distribution of total charges per year (Violin Plot)
# --------------------------------------------------------------------------------------------------

hcris_data_clean = hcris_data[hcris_data['tot_charges'] > 0].copy()
hcris_data_clean['log_tot_charges'] = np.log1p(hcris_data_clean['tot_charges'])

plt.figure(figsize=(12, 6))
sns.violinplot(x='fyear', y='log_tot_charges', data=hcris_data_clean, palette='viridis')
plt.title('Distribution of Log-Transformed Total Charges by Year')
plt.xlabel('Fiscal Year')
plt.ylabel('Log of Total Charges')
plt.xticks(rotation=45)
plt.savefig('graph_output/violin_plot_total_charges.png')
print("Violin plot saved successfully!")

# --------------------------------------------------------------------------------------------------
# Q4: Distribution of estimated prices per year (Violin Plot)
# --------------------------------------------------------------------------------------------------

# Price Calculation
hcris_data['discount_factor'] = 1 - (hcris_data['tot_discounts'] / hcris_data['tot_charges'])
hcris_data['price_num'] = (hcris_data['ip_charges'] + hcris_data['icu_charges'] + hcris_data['ancillary_charges']) * hcris_data['discount_factor'] - hcris_data['tot_mcare_payment']
hcris_data['price_denom'] = hcris_data['tot_discharges'] - hcris_data['mcare_discharges']
hcris_data['price'] = hcris_data['price_num'] / hcris_data['price_denom']

# Remove outliers and negative prices
hcris_data = hcris_data[(hcris_data['price_denom'] > 100) & (hcris_data['price_num'] > 0) & (hcris_data['price'] < 1000000)]

plt.figure(figsize=(12, 6))
sns.violinplot(x='fyear', y='price', data=hcris_data, palette='viridis')
plt.title('Distribution of Estimated Prices by Year')
plt.xlabel('Fiscal Year')
plt.ylabel('Estimated Price')
plt.xticks(rotation=45)
plt.savefig('graph_output/violin_plot_estimated_prices.png')
print("Violin plot saved successfully!")

# --------------------------------------------------------------------------------------------------
# Q5: Estimate ATEs - Only use 2012 data
# --------------------------------------------------------------------------------------------------
# Load dataset
hcris_data = pd.read_csv('/Users/ilsenovis/Documents/GitHub/ECON470HW2/data/output/HCRIS_Data.csv')

# Ensure fyear is integer
hcris_data['fyear'] = pd.to_numeric(hcris_data['fyear'], errors='coerce')

# Filter data for 2012
hcris_2012 = hcris_data[(hcris_data['fyear'] == 2012) & (hcris_data['beds'] > 30)].copy()

# Recalculate price after filtering
hcris_2012['discount_factor'] = 1 - (hcris_2012['tot_discounts'] / hcris_2012['tot_charges'])
hcris_2012['price_num'] = (hcris_2012['ip_charges'] + hcris_2012['icu_charges'] + hcris_2012['ancillary_charges']) * hcris_2012['discount_factor'] - hcris_2012['tot_mcare_payment']
hcris_2012['price_denom'] = hcris_2012['tot_discharges'] - hcris_2012['mcare_discharges']
hcris_2012['price'] = hcris_2012['price_num'] / hcris_2012['price_denom']

# Ensure denominator is valid (avoid division by zero or negative values)
hcris_2012 = hcris_2012[hcris_2012['price_denom'] > 0]

# Compute price only for valid rows
hcris_2012['price'] = hcris_2012['price_num'] / hcris_2012['price_denom']

# Drop remaining invalid rows
hcris_2012 = hcris_2012.replace([np.inf, -np.inf], np.nan).dropna(subset=['price'])

print("Final number of rows in 2012 dataset after cleaning:", len(hcris_2012))

# Check if 'price' column now exists
#print("Columns in hcris_2012 after recalculating price:", hcris_2012.columns)

# Ensure penalty calculation works
hcris_2012['hvbp_payment'] = hcris_2012['hvbp_payment'].fillna(0)
hcris_2012['hrrp_payment'] = hcris_2012['hrrp_payment'].fillna(0).abs()

# Define Penalty Variable (penalized = True if sum of payments is negative)
hcris_2012['penalty'] = (hcris_2012['hvbp_payment'] - hcris_2012['hrrp_payment']) < 0

print("\nChecking Penalty Distribution in 2012 Data:")
print(hcris_2012['penalty'].value_counts())

# If penalty is always False, check if HVBP and HRRP are zero
print("\nSummary of HVBP and HRRP Payments Before Abs Transformation:")
print(hcris_2012[['hvbp_payment', 'hrrp_payment']].describe())

# Take absolute value of HRRP
hcris_2012['hrrp_payment'] = hcris_2012['hrrp_payment'].abs()

# Redefine penalty after correcting HRRP
hcris_2012['penalty'] = (hcris_2012['hvbp_payment'] - hcris_2012['hrrp_payment']) < 0

# Check penalty distribution again
print("\nPenalty Distribution After Fixing HRRP:")
print(hcris_2012['penalty'].value_counts())

# Summary statistics for penalty
print("\nChecking Penalty Distribution in 2012 Data:")
print(hcris_2012['penalty'].value_counts())

# Compute Mean Price by Penalty
mean_price_penalized = round(hcris_2012[hcris_2012['penalty']]['price'].mean(), 2)
mean_price_non_penalized = round(hcris_2012[~hcris_2012['penalty']]['price'].mean(), 2)

print(f"Mean Price - Penalized Hospitals: {mean_price_penalized}")
print(f"Mean Price - Non-Penalized Hospitals: {mean_price_non_penalized}")

# --------------------------------------------------------------------------------------------------
# Q6: Split hospitals into quartiles based on bed size
# --------------------------------------------------------------------------------------------------

# Create bed size quartiles
hcris_2012['bed_quartile'] = pd.qcut(hcris_2012['beds'], q=4, labels=[1, 2, 3, 4])

# Check distribution
print("\nBed Quartile Distribution:")
print(hcris_2012['bed_quartile'].value_counts())

# Compute mean prices by penalty and quartile
price_by_quartile = hcris_2012.groupby(['bed_quartile', 'penalty'])['price'].mean().unstack()
print("\nAverage Prices by Quartile & Treatment Group:")
print(price_by_quartile)

# --------------------------------------------------------------------------------------------------
# Q7: Use Different Estimators
# --------------------------------------------------------------------------------------------------
# Nearest Neighbor Matching (Inverse Variance and Mahalanobis Distance)
# Selecting matching covariates
covariates = ['beds', 'mcaid_discharges', 'ip_charges', 'mcare_discharges', 'tot_mcare_payment']

# Check for NaN values in covariates before matching
print("Missing values in matching covariates before imputation:\n", hcris_2012[covariates].isna().sum())

# Fill missing values with column median (better than mean for outliers)
hcris_2012[covariates] = hcris_2012[covariates].apply(lambda x: x.fillna(x.median()))

# Verify no missing values remain
print("Missing values in matching covariates after imputation:\n", hcris_2012[covariates].isna().sum())

# Selecting matching covariates
covariates = ['beds', 'mcaid_discharges', 'ip_charges', 'mcare_discharges', 'tot_mcare_payment']
X = hcris_2012[covariates]
X = hcris_2012[covariates]
T = hcris_2012['penalty'].astype(int)
Y = hcris_2012['price']

# Nearest Neighbor Matching using Mahalanobis Distance
nn = NearestNeighbors(n_neighbors=1, metric='mahalanobis').fit(X)
_, indices = nn.kneighbors(X)

matched_prices_mahalanobis = Y.iloc[indices.flatten()].mean()
print(f"\nNearest Neighbor Matching (Mahalanobis Distance) ATE: {matched_prices_mahalanobis - Y.mean():.2f}")

# Nearest Neighbor Matching using Inverse Variance Distance
weights = 1 / (X.var().values)  # Compute inverse variance weights
nn_var = NearestNeighbors(n_neighbors=1, metric='euclidean', metric_params={'w': weights}).fit(X)
_, indices_var = nn_var.kneighbors(X)

matched_prices_variance = Y.iloc[indices_var.flatten()].mean()
print(f"Nearest Neighbor Matching (Inverse Variance) ATE: {matched_prices_variance - Y.mean():.2f}")

# Propesntiy Score Matching and Weighting
# Propensity Score Model
ps_model = LogisticRegression(solver='lbfgs', max_iter=1000)
ps_model.fit(X, T)
ps = ps_model.predict_proba(X)[:, 1]  # Get probabilities of penalty

# Nearest Neighbor Matching on Propensity Score
nn_ps = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(ps.reshape(-1, 1))
_, indices_ps = nn_ps.kneighbors(ps.reshape(-1, 1))

matched_prices_ps = Y.iloc[indices_ps.flatten()].mean()
print(f"Nearest Neighbor Matching (Propensity Score) ATE: {matched_prices_ps - Y.mean():.2f}")

# Inverse Propensity Score Weighting
weights_treated = 1 / ps
weights_control = 1 / (1 - ps)

ate_ipw = ((T * Y * weights_treated).sum() - ((1 - T) * Y * weights_control).sum()) / len(Y)
print(f"Inverse Propensity Score Weighting ATE: {ate_ipw:.2f}")

#Linear Regression for ATE Estimation
# Regression-Based ATE Estimation
regression_model = ols("price ~ penalty + C(bed_quartile)", data=hcris_2012).fit()
print("\nRegression-Based ATE Estimation:")
print(regression_model.summary())

# Final Summary Table
# Extract ATE from regression coefficient
ate_regression = regression_model.params['penalty']
print(f"\nRegression-Based ATE: {ate_regression:.2f}")

ate_results = pd.DataFrame({
    'Method': [
        'Exact Matching',
        'Nearest Neighbor (Inverse Variance)',
        'Nearest Neighbor (Mahalanobis)',
        'Nearest Neighbor (Propensity Score)',
        'Inverse Propensity Weighting',
        'Regression-Based ATE'
    ],
    'ATE': [
        None,  # Exact matching not yet implemented
        matched_prices_variance - Y.mean(),
        matched_prices_mahalanobis - Y.mean(),
        matched_prices_ps - Y.mean(),
        ate_ipw,
        ate_regression
    ]
})

print("\nFinal ATE Results:")
print(ate_results)

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

