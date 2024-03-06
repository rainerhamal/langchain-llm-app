import langchain_helper as lch
import streamlit as st

# Using streamlit to build an instamt frontend UI
st.title("Pets name generator")

user_animal_type = st.sidebar.selectbox("What type of pet do you have?", ("Cat", "Dog", "Cow", "Hamster"))

if user_animal_type == "Cat":
    pet_color = st.sidebar.text_area(label="What color is your cat?", max_chars=15)

if user_animal_type == "Dog":
    pet_color = st.sidebar.text_area(label="What color is your dog?", max_chars=15)

if user_animal_type == "Cow":
    pet_color = st.sidebar.text_area(label="What color is your cow?", max_chars=15)

if user_animal_type == "Hamster":
    pet_color = st.sidebar.text_area(label="What color is your hamster?", max_chars=15)

if pet_color:
    response = lch.generate_pet_name(user_animal_type, pet_color)
    st.text(response['pet_name'])