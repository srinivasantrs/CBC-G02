import random
import pandas as pd
import numpy as np
from faker import Faker

fake = Faker('en_IN')  # Set locale to Indian English

def generate_pan_number():
    return fake.random_uppercase_letter() + fake.random_uppercase_letter() + fake.random_uppercase_letter() + fake.random_uppercase_letter() + fake.random_uppercase_letter() + str(
        random.randint(10000, 999999))

def generate_name():
    return fake.name()

def generate_place_of_birth_country():
    return 'India' if random.random() < 0.9 else fake.country()

def generate_place_of_residence():
    return fake.state()

def generate_occupation():
    occupations = ['Engineer', 'Doctor', 'Teacher', 'Lawyer', 'Accountant', 'Artist', 'Manager', 'Entrepreneur',
                   'Scientist', 'Writer']
    return random.choice(occupations)

def generate_source_of_income():
    sources_of_income = ['Salary', 'Business', 'Investments', 'Rent', 'Freelance', 'Royalties', 'Dividends',
                         'Consulting', 'Pension', 'Others']
    return random.choice(sources_of_income)

def calculate_tax(income):
    if income <= 250000:
        return 0
    elif 250000 < income <= 500000:
        return round(0.05 * (income - 250000), 2)
    elif 500000 < income <= 750000:
        return round(0.1 * (income - 500000) + 0.05 * 250000, 2)
    elif 750000 < income <= 1000000:
        return round(0.15 * (income - 750000) + 0.1 * 250000 + 0.05 * 250000, 2)
    elif 1000000 < income <= 1250000:
        return round(0.2 * (income - 1000000) + 0.15 * 250000 + 0.1 * 250000 + 0.05 * 250000, 2)
    elif 1250000 < income <= 1500000:
        return round(0.25 * (income - 1250000) + 0.2 * 250000 + 0.15 * 250000 + 0.1 * 250000 + 0.05 * 250000, 2)
    else:
        return round(0.3 * (income - 1500000) + 0.25 * 250000 + 0.2 * 250000 + 0.15 * 250000 + 0.1 * 250000 + 0.05 * 250000, 2)

def generate_financial_data(year):
    net_income = round(random.uniform(50000, 1000000), 2)
    amount_after_exemptions = round(net_income * random.uniform(0.8, 1.0), 2)
    return net_income, amount_after_exemptions, calculate_tax(amount_after_exemptions)

# Generate synthetic data for 15,000 samples
dataset = []
for _ in range(15000):
    pan_number = generate_pan_number()
    name = generate_name()
    place_of_birth_country = generate_place_of_birth_country()
    place_of_residence = generate_place_of_residence()
    occupation = generate_occupation()
    source_of_income = generate_source_of_income()

    financial_data = {}
    for year in range(2012, 2023):
        net_income, amount_after_exemptions, tax_to_pay = generate_financial_data(year)
        tax_actually_paid = tax_to_pay if random.random() < 0.9 else round(random.uniform(0, tax_to_pay), 2)
        financial_data[f'{year}_net_income'] = net_income
        financial_data[f'{year}_amount_after_exemptions'] = amount_after_exemptions
        financial_data[f'{year}_tax_to_pay'] = tax_to_pay
        financial_data[f'{year}_tax_actually_paid'] = tax_actually_paid

    record = {
        'pan_number': pan_number,
        'name': name,
        'place_of_birth_country': place_of_birth_country,
        'place_of_residence': place_of_residence,
        'occupation': occupation,
        'source_of_income': source_of_income,
        **financial_data
    }

    dataset.append(record)

# Create DataFrame
data = pd.DataFrame(dataset)
data = data.round(2)  # Round off all numeric columns to 2 decimals

# Add new columns
years = range(2012, 2023)

# a. tax_fraud_score
tax_fraud_scores = []

for _, row in data.iterrows():
    tax_fraud_score = []
    for year in years:
        tax_to_pay_col = f"{year}_tax_to_pay"
        tax_actually_paid_col = f"{year}_tax_actually_paid"

        tax_fraud_score.append(row[tax_to_pay_col] - row[tax_actually_paid_col])

    # Calculate average tax fraud score over all years
    average_tax_fraud_score = np.mean(tax_fraud_score)
    tax_fraud_scores.append(average_tax_fraud_score)

data['tax_fraud_score'] = tax_fraud_scores

# b. target
threshold = 0.5  # You can adjust this threshold as needed
data['target'] = np.where((data['tax_fraud_score'] > (average_tax_fraud_score + 0.5 * np.std(tax_fraud_scores))) & (np.random.rand(len(data)) > 0.75), 1, 0)

# Save the updated data to CSV
data.to_csv("sample_data_with_targets.csv", index=False)
