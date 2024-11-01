import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data(file_path):
    """Load data from a CSV file into a Pandas DataFrame."""
    return pd.read_csv(file_path)

def process_data(data, group_by_col, sum_col, threshold=0.025):
    """Process the data by grouping and summing the specified column."""
    grouped_data = data.groupby(group_by_col)[sum_col].sum()
    if group_by_col == 'STATE/UT':
        others = grouped_data[grouped_data / grouped_data.sum() < threshold].sum()
        grouped_data = grouped_data[grouped_data / grouped_data.sum() >= threshold]
        grouped_data['Others'] = others
    return grouped_data

def plot_line(data, x_label, y_label, file_name):
    """Plot line chart."""
    plt.figure(figsize=(10, 10))
    data.plot(kind='line', marker='o', color='skyblue')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.savefig(f'Murder/{file_name}')
    plt.show()

def plot_barh(data, x_label, y_label, file_name):
    """Plot horizontal bar chart."""
    plt.figure(figsize=(10, 10))
    data.plot(kind='barh', color='skyblue')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(f'Murder/{file_name}')
    plt.show()

def plot_pie(data, file_name):
    """Plot pie chart."""
    plt.figure(figsize=(10, 10))
    data.plot(kind='pie', autopct='%1.1f%%', startangle=140, pctdistance=0.85)
    plt.ylabel('')
    plt.savefig(f'Murder/{file_name}')
    plt.show()

def plot_stacked_bar(data, title, x_label, y_label, file_name):
    """Plot stacked bar chart."""
    plt.figure(figsize=(10, 10))
    data.T.plot(kind='bar', stacked=True)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(title='Gender')
    plt.savefig(f'Murder/{file_name}')
    plt.show()

def plot_trend(data, file_name):
    """Plot trend analysis."""
    data_grouped = data.groupby(['YEAR', 'GENDER']).sum(numeric_only=True).reset_index()
    data_grouped.set_index(['YEAR', 'GENDER']).unstack().plot(kind='line', figsize=(14, 10))
    plt.title('Trend Analysis of Age Groups by Gender Over Years')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.legend(title='Age Group', loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f'Murder/{file_name}')
    plt.show()

def main(file_path):
    # Load data
    murder_data = load_data(file_path)

    # Process and plot year-wise data
    year_wise = process_data(murder_data, 'YEAR', 'Total')
    plot_line(year_wise, 'Year', 'Number of Victims', 'year_wise.png')

    # Process and plot state-wise data
    state_wise = process_data(murder_data, 'STATE/UT', 'Total')
    plot_barh(state_wise, 'Number of Victims', 'State/UT', 'state_wise.png')

    # Plot pie chart with 'Others' category
    state_wise_other = process_data(murder_data, 'STATE/UT', 'Total')
    plot_pie(state_wise_other, 'state_wise_other(pie).png')

    # Plot stacked bar chart for gender totals
    gender_totals = murder_data.drop(columns=['YEAR'], errors='ignore').groupby("GENDER").sum(numeric_only=True)
    plot_stacked_bar(gender_totals, 'Age Group Distribution by Gender', 'Age Group', 'Count', 'gender_totals.png')

    # Plot trend analysis if 'YEAR' column exists
    if 'YEAR' in murder_data.columns:
        plot_trend(murder_data, 'murder_data_grouped.png')
    else:
        print("Column 'YEAR' does not exist in the DataFrame.")

# Example usage
main('Data/Violent/VICTIM_OF_MURDER_0.csv')