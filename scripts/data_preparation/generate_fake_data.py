"""
This data preparation script generates a fake dataset of patient data and saves it to a CSV file.
"""

import pandas as pd
import random
import faker
from sklearn.model_selection import train_test_split

# Set the number of samples
num_samples = 1000

# Initialize a Faker instance for generating fake data
fake = faker.Faker('nl_NL')

options = ['0', '1', '2']

# Generate random data
data = {
    'Patnr': [random.randint(1000000, 1040000) for _ in range(num_samples)],
    'Praktnr': [random.randint(1000, 1083) for _ in range(num_samples)],
    'Koorts': [random.choice(options) for _ in range(num_samples)],
    'Hoesten': [random.choice(options) for _ in range(num_samples)],
    'Kortademigheid': [random.choice(options) for _ in range(num_samples)],
}

# Create a DataFrame
df = pd.DataFrame(data)

# Add relevant keywords to the 'DEDUCE_omschrijving' field based on the values of the 'Koorts', 'Hoesten', and 'Kortademigheid' fields
df['DEDUCE_omschrijving'] = df.apply(lambda row: fake.text().replace('\n', ' ') +
                                      (' Koorts' if row['Koorts'] == '1' else ' geen Koorts' if row['Koorts'] == '0' else '') +
                                      (' Hoesten' if row['Hoesten'] == '1' else ' geen Hoesten' if row['Hoesten'] == '0' else '') +
                                      (' Kortademigheid' if row['Kortademigheid'] == '1' else ' geen Kortademigheid' if row['Kortademigheid'] == '0' else ''), axis=1)

def divide_set(df: pd.DataFrame, k: int = 5) -> list[pd.DataFrame]:
    """Divides the provided set into k equally sized subsets, then returns them in a list.

    Args:
        df (pd.DataFrame): Dataframe to use.
        k (int, optional): Amount of subsets. Defaults to 5.

    Returns:
        list[pd.DataFrame]: A list containing the chunked-up subsets of the original DataFrame.
    """
    subset_size = len(df) // k
    subsets = []
    for i in range(k):
        subset = df[i*subset_size:(i+1)*subset_size]
        subsets.append(subset)
    return subsets
    
test_sets = divide_set(df, k=5)

# Save the DataFrame to a CSV file
df.to_csv('data/raw/patient_data/fake_set_complete.csv', index=False)
for i, test_set in enumerate(test_sets):
    test_set.to_csv(f'data/raw/patient_data/fake_test_{i+1}.csv', index=False)