import streamlit_authenticator as stauth

hashed_passwords = stauth.Hasher(['bangalore']).generate()
print(hashed_passwords)
