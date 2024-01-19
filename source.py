import yaml
from streamlit_authenticator import Authenticate
from yaml.loader import SafeLoader
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import plotly.express as px

with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

name, authentication_status, username = authenticator.login('Login', 'sidebar')

if authentication_status:
    from PIL import Image
    image_logo = Image.open('logo.jpg')
    st.sidebar.image(image_logo)

    st.sidebar.success(f'Welcome **{name}**')
    st.sidebar.warning("Income tax is a direct tax imposed on the earnings of individuals, businesses, and other entities by the government. It is a **key source** of revenue for governments worldwide and is used to **fund public services** and **infrastructure**.")
    st.sidebar.error(" The basic principle behind income tax is that individuals and entities with **higher incomes** contribute a **larger share** of their earnings to support **public expenditures.**")

    authenticator.logout('Logout', 'sidebar')  # sidebar or main

    st.set_option('deprecation.showPyplotGlobalUse', False)


    st.title("TAX FRAUD DETECTION")

    data = pd.read_csv('sample_data_with_targets.csv')
    citizen_new_old = st.selectbox("SELECT CITIZEN TYPE",options=['REGISTERED','UNREGISTERED'])

    if citizen_new_old == 'REGISTERED':
        st.markdown('---')
        pan_number_input = st.text_input('ENTER PAN ID: ')
        pan_number_input = pan_number_input.upper()

        submit_button = st.button("SUBMIT")

        if submit_button:
            filter = data['pan_number'] == pan_number_input
            data_filtered = data[filter]

            st.subheader("Citizen Details")
            st.dataframe(data_filtered[['pan_number', 'name', 'place_of_birth_country',
                                        'place_of_residence', 'occupation', 'source_of_income']].T, width=750)

            # Define variables based on filtered data
            income = data_filtered[['2012_net_income', '2013_net_income', '2014_net_income', '2015_net_income',
                                    '2016_net_income', '2017_net_income', '2018_net_income', '2019_net_income',
                                    '2020_net_income', '2021_net_income', '2022_net_income']].values[0]
            exempted_amount = data_filtered[['2012_amount_after_exemptions', '2013_amount_after_exemptions',
                                             '2014_amount_after_exemptions', '2015_amount_after_exemptions',
                                             '2016_amount_after_exemptions', '2017_amount_after_exemptions',
                                              '2018_amount_after_exemptions', '2019_amount_after_exemptions',
                                             '2020_amount_after_exemptions', '2021_amount_after_exemptions',
                                             '2022_amount_after_exemptions']].values[0]
            tax_to_pay = data_filtered[['2012_tax_to_pay', '2013_tax_to_pay', '2014_tax_to_pay', '2015_tax_to_pay',
                                        '2016_tax_to_pay', '2017_tax_to_pay', '2018_tax_to_pay', '2019_tax_to_pay',
                                        '2020_tax_to_pay', '2021_tax_to_pay', '2022_tax_to_pay']].values[0]
            tax_actually_paid = data_filtered[['2012_tax_actually_paid', '2013_tax_actually_paid',
                                               '2014_tax_actually_paid', '2015_tax_actually_paid',
                                               '2016_tax_actually_paid', '2017_tax_actually_paid',
                                               '2018_tax_actually_paid', '2019_tax_actually_paid',
                                               '2020_tax_actually_paid', '2021_tax_actually_paid',
                                               '2022_tax_actually_paid']].values[0]

            # Create a time variable (years)
            years = range(2012, 2023)

            st.markdown("### Income and Exempted Amount")
            # Plotting income and exempted amount using Plotly Express
            fig1 = px.line(x=years, y=[income, exempted_amount], labels={'x': 'Year', 'value': 'Amount'},
                           markers=True, line_shape='linear',
                           line_dash_sequence=['solid', 'solid'], color_discrete_sequence=['red', 'orange'])

            fig1.update_layout(legend_title='Legend')
            fig1.update_traces(fill='tonexty', fillcolor='rgba(255, 165, 0, 0.2)')

            fig1.for_each_trace(lambda t: t.update(name='Income' if t.name == 'wide_variable_0' else 'Exempted Amount'))
            st.plotly_chart(fig1)

            # Display the data used for plots as dataframes
            st.dataframe(pd.DataFrame({'Year': years, 'Income': income, 'Exempted Amount': exempted_amount}),width=750)

            st.markdown("### Tax to Pay and Tax Actually Paid")
            # Plotting tax to pay and tax actually paid using Plotly Express
            fig2 = px.line(x=years, y=[tax_to_pay, tax_actually_paid], labels={'x': 'Year', 'value': 'Amount'},
                           markers=True, line_shape='linear',
                           line_dash_sequence=['solid', 'solid'], color_discrete_sequence=['red', 'brown'])

            fig2.update_layout(legend_title='Legend')
            fig2.update_traces(fill='tonexty', fillcolor='rgba(255, 0, 165, 0.2)')

            fig2.for_each_trace(
                lambda t: t.update(name='Tax to Pay' if t.name == 'wide_variable_0' else 'Tax Actually Paid'))
            st.plotly_chart(fig2)

            st.dataframe(pd.DataFrame({'Year': years, 'Tax to Pay': tax_to_pay, 'Tax Actually Paid': tax_actually_paid}),width=750)

            st.markdown("### Tax Fraud Score")
            tax_fraud_scores = []

            for year in years:
                tax_to_pay_col = f"{year}_tax_to_pay"
                tax_actually_paid_col = f"{year}_tax_actually_paid"

                tax_fraud_score = data_filtered[tax_to_pay_col] - data_filtered[tax_actually_paid_col]
                tax_fraud_scores.append(tax_fraud_score)

            # Calculate average tax fraud score over all years
            average_tax_fraud_score = np.mean(tax_fraud_scores)
            st.info(f"**Tax Fraud Score** Over All Years: {average_tax_fraud_score}")

            st.header("DETECTION")
            if data_filtered['target'].any() == 0:
                st.success("NO FRAUD")
            elif data_filtered['target'].any() == 1:
                st.error("FRAUD")

    if citizen_new_old == 'UNREGISTERED':
        st.markdown('---')

        name_input = st.text_input("ENTER NAME : ")
        occupation_input = st.selectbox("ENTER OCCUPATION : ",options=['Manager',
                                                                        'Teacher',
                                                                        'Engineer',
                                                                        'Doctor',
                                                                        'Entrepreneur',
                                                                        'Artist',
                                                                        'Writer',
                                                                        'Lawyer',
                                                                        'Accountant',
                                                                        'Scientist'])

        col1, col2 = st.columns(2)
        with col1:
            tax_to_pay_2012 = st.number_input("ENTER TAX TO PAY 2012", step=1)
            tax_to_pay_2013 = st.number_input("ENTER TAX TO PAY 2013", step=1)
            tax_to_pay_2014 = st.number_input("ENTER TAX TO PAY 2014", step=1)
            tax_to_pay_2015 = st.number_input("ENTER TAX TO PAY 2015", step=1)
            tax_to_pay_2016 = st.number_input("ENTER TAX TO PAY 2016", step=1)
            tax_to_pay_2017 = st.number_input("ENTER TAX TO PAY 2017", step=1)
            tax_to_pay_2018 = st.number_input("ENTER TAX TO PAY 2018", step=1)
            tax_to_pay_2019 = st.number_input("ENTER TAX TO PAY 2019", step=1)
            tax_to_pay_2020 = st.number_input("ENTER TAX TO PAY 2020", step=1)
            tax_to_pay_2021 = st.number_input("ENTER TAX TO PAY 2021", step=1)
            tax_to_pay_2022 = st.number_input("ENTER TAX TO PAY 2022", step=1)

        with col2:
            tax_paid_2012 = st.number_input("ENTER TAX PAID 2012", step=1)
            tax_paid_2013 = st.number_input("ENTER TAX PAID 2013", step=1)
            tax_paid_2014 = st.number_input("ENTER TAX PAID 2014", step=1)
            tax_paid_2015 = st.number_input("ENTER TAX PAID 2015", step=1)
            tax_paid_2016 = st.number_input("ENTER TAX PAID 2016", step=1)
            tax_paid_2017 = st.number_input("ENTER TAX PAID 2017", step=1)
            tax_paid_2018 = st.number_input("ENTER TAX PAID 2018", step=1)
            tax_paid_2019 = st.number_input("ENTER TAX PAID 2019", step=1)
            tax_paid_2020 = st.number_input("ENTER TAX PAID 2020", step=1)
            tax_paid_2021 = st.number_input("ENTER TAX PAID 2021", step=1)
            tax_paid_2022 = st.number_input("ENTER TAX PAID 2022", step=1)

        unregistered_citizen_button = st.button("CHECK")


        if unregistered_citizen_button:
            features = [[tax_to_pay_2012, tax_paid_2012,
                tax_to_pay_2013, tax_paid_2013,
                tax_to_pay_2014, tax_paid_2014,
                tax_to_pay_2015, tax_paid_2015,
                tax_to_pay_2016, tax_paid_2016,
                tax_to_pay_2017, tax_paid_2017,
                tax_to_pay_2018, tax_paid_2018,
                tax_to_pay_2019, tax_paid_2019,
                tax_to_pay_2020, tax_paid_2020,
                tax_to_pay_2021, tax_paid_2021,
                tax_to_pay_2022, tax_paid_2022]]



            # Create a dictionary with user input
            user_input_dict = {
                "Name": [name_input],
                "Occupation": [occupation_input],
                "Tax to Pay 2012": [tax_to_pay_2012],
                "Tax Paid 2012": [tax_paid_2012],
                "Tax to Pay 2013": [tax_to_pay_2013],
                "Tax Paid 2013": [tax_paid_2013],
                "Tax to Pay 2014": [tax_to_pay_2014],
                "Tax Paid 2014": [tax_paid_2014],
                "Tax to Pay 2015": [tax_to_pay_2015],
                "Tax Paid 2015": [tax_paid_2015],
                "Tax to Pay 2016": [tax_to_pay_2016],
                "Tax Paid 2016": [tax_paid_2016],
                "Tax to Pay 2017": [tax_to_pay_2017],
                "Tax Paid 2017": [tax_paid_2017],
                "Tax to Pay 2018": [tax_to_pay_2018],
                "Tax Paid 2018": [tax_paid_2018],
                "Tax to Pay 2019": [tax_to_pay_2019],
                "Tax Paid 2019": [tax_paid_2019],
                "Tax to Pay 2020": [tax_to_pay_2020],
                "Tax Paid 2020": [tax_paid_2020],
                "Tax to Pay 2021": [tax_to_pay_2021],
                "Tax Paid 2021": [tax_paid_2021],
                "Tax to Pay 2022": [tax_to_pay_2022],
                "Tax Paid 2022": [tax_paid_2022],
            }

            st.markdown('### UNREGISTERED CITIZEN DETAILS')
            user_input_df = pd.DataFrame(user_input_dict)
            st.dataframe(user_input_df.T,width=750)

            data = pd.read_csv('sample_data_with_targets.csv')
            x = data[['2012_tax_to_pay', '2012_tax_actually_paid',
                      '2013_tax_to_pay', '2013_tax_actually_paid',
                      '2014_tax_to_pay', '2014_tax_actually_paid',
                      '2015_tax_to_pay', '2015_tax_actually_paid',
                      '2016_tax_to_pay', '2016_tax_actually_paid',
                      '2017_tax_to_pay', '2017_tax_actually_paid',
                      '2018_tax_to_pay', '2018_tax_actually_paid',
                      '2019_tax_to_pay', '2019_tax_actually_paid',
                      '2020_tax_to_pay', '2020_tax_actually_paid',
                      '2021_tax_to_pay', '2021_tax_actually_paid',
                      '2022_tax_to_pay', '2022_tax_actually_paid']].values
            y = data['target'].values

            from sklearn.model_selection import train_test_split
            x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=100)

            tab1, tab2, tab3 = st.tabs(['SVM','RANDOM FOREST','XGBOOST'])
            with tab1:
                from sklearn.svm import SVC
                svm = SVC(probability=True)
                svm.fit(x_train,y_train)
                svm_pred = svm.predict(x_test)
                svm_result = svm.predict(features)[0]
                
                st.subheader("SVM DIAGNOSIS")
                if svm_result == 0:
                    st.success("NO FRAUD")
                if svm_result == 1:
                    st.error("FRAUD")

                st.subheader("SVM MODEL PARAMETERS")
                from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
                st.success(f"svm accuracy : {accuracy_score(y_test,svm_pred)}")
                st.info(f"svm precision  : {precision_score(y_test,svm_pred,pos_label=0)}")
                st.error(f"svm recall : {recall_score(y_test,svm_pred,pos_label=0)}")
                st.warning(f"svm f1 score : {f1_score(y_test,svm_pred,pos_label=0)}")

            with tab2:
                from sklearn.ensemble import RandomForestClassifier
                rf = RandomForestClassifier()
                rf.fit(x_train,y_train)
                rf_pred = rf.predict(x_test)

                rf_result = rf.predict(features)[0]

                st.subheader("RANDOM FOREST DIAGNOSIS")
                if rf_result == 0:
                    st.success("NO FRAUD")
                if rf_result == 1:
                    st.error("FRAUD")

                st.subheader("RANDOM FOREST MODEL PARAMETERS")
                from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
                st.success(f"random forest accuracy : {accuracy_score(y_test,rf_pred)}")
                st.info(f"random forest precision  : {precision_score(y_test,rf_pred,pos_label=0)}")
                st.error(f"random forest recall : {recall_score(y_test,rf_pred,pos_label=0)}")
                st.warning(f"random forest f1 score : {f1_score(y_test,rf_pred,pos_label=0)}")
                
                with tab3:
                    from xgboost import XGBClassifier

                    xgb = XGBClassifier()
                    xgb.fit(x_train, y_train)
                    xgb_pred = xgb.predict(x_test)
                    xgb_result = xgb.predict(features)[0]

                    st.subheader("XGBOOST DIAGNOSIS")
                    if xgb_result == 0:
                        st.success("NO FRAUD")
                    if xgb_result == 1:
                        st.error("FRAUD")

                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                    st.subheader("XGBOOST PARAMETERS")
                    st.success(f"xgboost accuracy : {accuracy_score(y_test, xgb_pred)}")
                    st.info(f"xgboost precision  : {precision_score(y_test, xgb_pred, pos_label=0)}")
                    st.error(f"xgboost recall : {recall_score(y_test, xgb_pred, pos_label=0)}")
                    st.warning(f"xgboost f1 score : {f1_score(y_test, xgb_pred, pos_label=0)}")


elif authentication_status == False:
    st.error('Username/password is incorrect')

elif authentication_status is None:
    st.warning('Please enter your username and password')