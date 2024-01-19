import numpy as np
import pandas as pd
data = pd.read_csv('income_tax_dataset.csv')

# print(data.columns[data.isna().any()])

data['2001_tax_score'] = data['2001_tax_to_pay'] - data['2001_tax_actually_paid']
data['2002_tax_score'] = data['2002_tax_to_pay'] - data['2002_tax_actually_paid']
data['2003_tax_score'] = data['2003_tax_to_pay'] - data['2003_tax_actually_paid']
data['2004_tax_score'] = data['2004_tax_to_pay'] - data['2004_tax_actually_paid']
data['2005_tax_score'] = data['2005_tax_to_pay'] - data['2005_tax_actually_paid']
data['2006_tax_score'] = data['2006_tax_to_pay'] - data['2006_tax_actually_paid']
data['2007_tax_score'] = data['2007_tax_to_pay'] - data['2007_tax_actually_paid']
data['2008_tax_score'] = data['2008_tax_to_pay'] - data['2008_tax_actually_paid']
data['2009_tax_score'] = data['2009_tax_to_pay'] - data['2009_tax_actually_paid']
data['2010_tax_score'] = data['2010_tax_to_pay'] - data['2010_tax_actually_paid']
data['2011_tax_score'] = data['2011_tax_to_pay'] - data['2011_tax_actually_paid']
data['2012_tax_score'] = data['2012_tax_to_pay'] - data['2012_tax_actually_paid']
data['2013_tax_score'] = data['2013_tax_to_pay'] - data['2013_tax_actually_paid']
data['2014_tax_score'] = data['2014_tax_to_pay'] - data['2014_tax_actually_paid']
data['2015_tax_score'] = data['2015_tax_to_pay'] - data['2015_tax_actually_paid']

place_of_birth_country_unique_list = data['place_of_birth_country'].unique().tolist()
place_of_birth_country_index = np.arange(1,len(place_of_birth_country_unique_list)+1,1)
place_of_birth_country_text_to_num = {i:j for i,j in zip(place_of_birth_country_unique_list,place_of_birth_country_index)}
place_of_birth_country_num_to_text = {j:i for i,j in zip(place_of_birth_country_unique_list,place_of_birth_country_index)}

place_of_residence_unique_list = data['place_of_residence'].unique().tolist()
place_of_residence_index = np.arange(1,len(place_of_residence_unique_list)+1,1)
place_of_residence_text_to_num = {i:j for i,j in zip(place_of_residence_unique_list,place_of_residence_index)}
place_of_residence_num_to_text = {j:i for i,j in zip(place_of_residence_unique_list,place_of_residence_index)}

occupation_unique_list = data['occupation'].unique().tolist()
occupation_index = np.arange(1,len(occupation_unique_list)+1,1)
occupation_text_to_num = {i:j for i,j in zip(occupation_unique_list,occupation_index)}
occupation_num_to_text = {j:i for i,j in zip(occupation_unique_list,occupation_index)}

source_of_income_unique_list = data['source_of_income'].unique().tolist()
source_of_income_index = np.arange(1,len(source_of_income_unique_list)+1,1)
source_of_income_text_to_num = {i:j for i,j in zip(source_of_income_unique_list,source_of_income_index)}
source_of_income_num_to_text = {j:i for i,j in zip(source_of_income_unique_list,source_of_income_index)}

data['place_of_birth_country'] = data['place_of_birth_country'].map(place_of_birth_country_text_to_num)
data['place_of_residence'] = data['place_of_residence'].map(place_of_residence_text_to_num)
data['occupation'] = data['occupation'].map(occupation_text_to_num)
data['source_of_income'] = data['source_of_income'].map(source_of_income_text_to_num)

x = data[['place_of_birth_country', 'place_of_residence', 'occupation', 'source_of_income',
          '2001_net_income', '2001_amount_after_exemptions', '2001_tax_actually_paid',
          '2002_net_income', '2002_amount_after_exemptions', '2002_tax_actually_paid',
          '2003_net_income', '2003_amount_after_exemptions', '2003_tax_actually_paid',
          '2004_net_income', '2004_amount_after_exemptions', '2004_tax_actually_paid',
          '2005_net_income', '2005_amount_after_exemptions', '2005_tax_actually_paid',
          '2006_net_income', '2006_amount_after_exemptions', '2006_tax_actually_paid',
          '2007_net_income', '2007_amount_after_exemptions', '2007_tax_actually_paid',
          '2008_net_income', '2008_amount_after_exemptions', '2008_tax_actually_paid',
          '2009_net_income', '2009_amount_after_exemptions', '2009_tax_actually_paid',
          '2010_net_income', '2010_amount_after_exemptions', '2010_tax_actually_paid',
          '2011_net_income', '2011_amount_after_exemptions', '2011_tax_actually_paid',
          '2012_net_income', '2012_amount_after_exemptions', '2012_tax_actually_paid',
          '2013_net_income', '2013_amount_after_exemptions', '2013_tax_actually_paid',
          '2014_net_income', '2014_amount_after_exemptions', '2014_tax_actually_paid',
          '2015_net_income', '2015_amount_after_exemptions', '2015_tax_actually_paid']]

# print(x.columns[x.isna().any()])
# print(data.columns[data.isna().any()])

pan_number_input = str(input('enter pan number : '))
filter = data['pan_number'] == pan_number_input
print(data[filter])