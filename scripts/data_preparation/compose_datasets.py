import pandas as pd
import pyreadstat

# Load the CSV file
df_csv = pd.read_csv('Element_freetext_SO.csv', sep='|', usecols=["Patnr", "DEDUCE_omschrijving,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,"])

# Load the SPSS file
df_spss, meta = pyreadstat.read_sav('Element_sample_freetext.sav')

# Merge the dataframes
df = pd.merge(df_csv, df_spss, on='Patnr')