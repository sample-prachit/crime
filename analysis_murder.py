# importing all the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# loading the csv data to a Pandas DataFrame
murder_data = pd.read_csv('Data/Violent/VICTIM_OF_MURDER_0.csv')

# Processing the data adding all the values in the Victims_Total column and grouping by year
year_wise = murder_data.groupby('YEAR')['Total'].sum()
state_wise = murder_data.groupby('STATE/UT')['Total'].sum().sort_values()
state_wise_other = state_wise.sort_values(ascending=False)
others = state_wise_other[state_wise_other/state_wise_other.sum() < 0.025].sum()
state_wise_other = state_wise_other[state_wise_other/state_wise_other.sum() >= 0.025]
state_wise_other['Others'] = others

# plotting the data points in 2D
plt.figure(figsize=(10,10))
year_wise.plot(kind='line',marker='o',color='skyblue')
plt.xlabel('Year')
plt.ylabel('Number of Victims')
plt.grid(True)
plt.savefig('Murder/year_wise.png')
plt.show()

plt.figure(figsize=(10,10))
state_wise.plot(kind='barh',color='skyblue')
plt.xlabel('Number of Victims')
plt.ylabel('State/UT')
plt.savefig('Murder/state_wise.png')
plt.show()

plt.figure(figsize=(10,10))
state_wise_other.plot(kind='pie',autopct='%1.1f%%',startangle=140,pctdistance=0.85)
plt.ylabel('')
plt.savefig('Murder/state_wise_other(pie).png')
plt.show()

# Stacked Bar Chart
gender_totals = murder_data.drop(columns=['YEAR'], errors='ignore')
gender_totals = gender_totals.groupby("GENDER").sum(numeric_only=True)
plt.figure(figsize=(10, 10))
gender_totals.T.plot(kind='bar', stacked=True)
plt.title('Age Group Distribution by Gender')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.legend(title='Gender')
plt.savefig('Murder/gender_totals.png')
plt.show()

if 'YEAR' in murder_data.columns:
    # Aggregate data to avoid duplicates
    murder_data_grouped = murder_data.groupby(['YEAR', 'GENDER']).sum(numeric_only=True).reset_index()

    # Set index and unstack for plotting
    murder_data_grouped.set_index(['YEAR', 'GENDER']).unstack().plot(kind='line', figsize=(14, 10))

    # Set titles and labels
    plt.title('Trend Analysis of Age Groups by Gender Over Years')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.legend(title='Age Group', loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.savefig('Murder/murder_data_grouped.png')
    plt.show()
else:
    print("Column 'YEAR' does not exist in the DataFrame.")