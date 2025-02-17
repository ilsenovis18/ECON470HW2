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

# Load and look at cleaned HCRIS dataset
hcris_data = pd.read_csv('/Users/ilsenovis/Documents/GitHub/ECON470HW2/data/output/HCRIS_Data.csv')
print(hcris_data.head())
print(hcris_data.columns)

# Data Summary ----------------------------------------------------------------------------------------------------------------
# Count of hospitals with more than 1 report per year
multi_report_hospitals = hcris_data.groupby(['fyear', 'provider_number']).size().reset_index(name='report_count')
multi_report_hospitals['fyear'] = multi_report_hospitals['fyear'].astype(int)
multi_report_counts = multi_report_hospitals.groupby('fyear')['report_count'].apply(lambda x: (x > 1).sum())

# Create the 'output' directory if it doesn't exist
os.makedirs('graph_output', exist_ok=True)

# Line plot for hospitals with multiple reports
plt.figure(figsize=(10, 6))
plt.plot(multi_report_counts.index, multi_report_counts.values, marker='o')
plt.title('Number of Hospitals with Multiple Reports Over Time')
plt.xlabel('Fiscal Year')
plt.ylabel('Number of Hospitals')
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.grid()
plt.tight_layout()
plt.savefig('graph_output/multi_report_hospitals.png')
print("Plot saved successfully!")

# Unique Hospital IDs
unique_hospitals = hcris_data['provider_number'].nunique()
print(f"Total Unique Hospitals (Medicare Provider Number): {unique_hospitals}")

# Distribution of Total Charges ----------------------------------------------------------------------------------------------------------------
# Remove NaN and negative values before log transformation
hcris_data_clean = hcris_data[hcris_data['tot_charges'] > 0].copy()

# Log-transform for better visualization
hcris_data_clean['log_tot_charges'] = np.log1p(hcris_data_clean['tot_charges'])

# Violin Plot for Total Charges
plt.figure(figsize=(12, 6))
sns.violinplot(x='fyear', y='log_tot_charges', data=hcris_data_clean, palette='viridis')
plt.title('Distribution of Log-Transformed Total Charges by Year')
plt.xlabel('Fiscal Year')
plt.ylabel('Log of Total Charges')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('graph_output/violin_plot_total_charges.png')
print("Violin plot saved successfuly to 'graph_output/violin_plot_total_charges.png'!")

# Distribution of Estimated Prices ----------------------------------------------------------------------------------------------------------------
# Price Calculation
hcris_data['discount_factor'] = 1 - (hcris_data['tot_discounts'] / hcris_data['tot_charges'])
hcris_data['price_num'] = (hcris_data['ip_charges'] + hcris_data['icu_charges'] + hcris_data['ancillary_charges']) * hcris_data['discount_factor'] - hcris_data['tot_mcare_payment']
hcris_data['price_denom'] = hcris_data['tot_discharges'] - hcris_data['mcare_discharges']

# Filter to avoid division by zero
hcris_data = hcris_data[hcris_data['price_denom'] != 0]

# Price Calculation
hcris_data['price'] = hcris_data['price_num'] / hcris_data['price_denom']

# Remove outliers and negative prices
hcris_data = hcris_data[(hcris_data['price'] > 0) & (hcris_data['price'] < hcris_data['price'].quantile(0.99))]

# Violin Plot for Estimated Prices
plt.figure(figsize=(12, 6))
sns.violinplot(x='fyear', y='price', data=hcris_data, palette='viridis')
plt.title('Distribution of Estimated Prices by Year')
plt.xlabel('Fiscal Year')
plt.ylabel('Estimated Price')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('graph_output/violin_plot_estimated_prices.png')
print("Violin plot saved successfuly to 'graph_output/violin_plot_estimated_prices.png'!")

# Estimate ATEs -------------------------------------------------------------------------------------
print("\nStarting ATE Estimation...")
hcris_data['fyear'] = hcris_data['fyear'].astype(int)

# Filter data for 2012
hcris_2012 = hcris_data[(hcris_data['fyear'] == 2012) &
                        (hcris_data['price_denom'] > 100) &
                        (hcris_data['price_num'] > 0) &
                        (hcris_data['price'].notna()) &
                        (hcris_data['price'] < 1000000) &
                        (hcris_data['beds'] > 30)].copy()

hcris_2012['hvbp_payment'] = hcris_2012['hvbp_payment'].fillna(0)
hcris_2012['hrrp_payment'] = hcris_2012['hrrp_payment'].fillna(0)

# Define Penalty Variable
hcris_2012['penalty'] = (hcris_2012['hvbp_payment'] - hcris_2012['hrrp_payment']) < 0

# Compute Mean Price
mean_price_penalized = round(hcris_2012.loc[hcris_2012['penalty'], 'price'].mean(), 2)
mean_price_non_penalized = round(hcris_2012.loc[~hcris_2012['penalty'], 'price'].mean(), 2)

print(f"Mean Price - Penalized Hospitals: {mean_price_penalized}")
print(f"Mean Price - Non-Penalized Hospitals: {mean_price_non_penalized}")

# Recalculate ATEs
avg_price_penalized = hcris_2012[hcris_2012['penalty'] == True]['price'].mean()
avg_price_non_penalized = hcris_2012[hcris_2012['penalty'] == False]['price'].mean()

print(f"Average Price - Penalized Hospitals: {avg_price_penalized:.2f}")
print(f"Average Price - Non-Penalized Hospitals: {avg_price_non_penalized:.2f}")
