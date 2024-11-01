import glob
import pandas as pd

# Define the columns you want to keep
column_mapping = {
    "States/UTs": "STATE/UT",
    "District": "DISTRICT",
    "Year": "Year",
    "Rape": "Rape",
    "Kidnapping & Abduction_Total": "Kidnapping and Abduction",
    "Dowry Deaths": "Dowry Deaths",
    "Assault on Women with intent to outrage her Modesty_Total": "Assault on women with intent to outrage her modesty",
    "Insult to the Modesty of Women_Total": "Insult to modesty of Women",
    "Cruelty by Husband or his Relatives": "Cruelty by Husband or his Relatives",
    "Importation of Girls from Foreign Country": "Importation of Girls"
}

# Path to your CSV files (update this to the location of your files)
path = "Data/women/42_District_wise_crimes_committed_against_women_2014.csv"

combined_data = pd.DataFrame()

# Loop through each CSV file
for filename in glob.glob(path):
    # Read the CSV file
    data = pd.read_csv(filename)

    # Rename the columns according to the mapping
    data.rename(columns=column_mapping, inplace=True)

    # Select only the relevant columns
    filtered_data = data[list(column_mapping.values())]

    # Append the filtered data to the combined DataFrame
    combined_data = pd.concat([combined_data, filtered_data], ignore_index=True)

# Save the combined data to a new CSV file
combined_data.to_csv("combined_data.csv", index=False)
