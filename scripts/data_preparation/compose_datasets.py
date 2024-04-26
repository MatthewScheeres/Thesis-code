import pandas as pd
import pyreadstat

# Load the CSV file
df_ehr = pd.read_csv('Element_freetext_SO.csv', sep='|', usecols=["Patnr", "DEDUCE_omschrijving,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,"])

# Load the SPSS file
symptoms = ["Temp", "Hoesten", "Dyspnoe"]
keep_columns = ["Patnr"] + symptoms

df_spss, meta = pyreadstat.read_sav('Element_sample_freetext.sav')
df_symptoms = df_spss[keep_columns]

# Parse Koorts Data
df_symptoms['Koorts'] = df_symptoms['Temp'].apply(lambda x: 0 if x in [0, 1] else 1 if x == 2 else 2)

df_symptoms[: , symptoms] = df_symptoms[symptoms].fillna(2)
df_symptoms = df_symptoms.rename(columns={'Dyspnoe': 'Kortademigheid', 'Temp': 'Koorts'})

# Merge the dataframes
df = pd.merge(df_ehr, df_spss, on='Patnr')


df.to_csv('output.csv', index=False)