import streamlit as st
import requests

st.title("Кредитная карта")
st.write("Мгоновенное одобрение кредитной карты")

with st.form("Подать заявку"):
    age = st.number_input("Ваш возраст", min_value=18)
    income = st.number_input("Ваш доход в тысячах рублей", min_value=0)
    education = st.checkbox("У меня есть высше образование")
    work = st.checkbox("У меня есть стабильная работа")
    car = st.checkbox("У меня есть автомобиль")
    submit = st.form_submit_button("Подать заявку")

if submit:
    data = {"age": age, "income": income, "education": education, "work": work, "car": car}
    response = requests.post("http://127.0.0.1:8000/score", json=data)
    if response.json()["approved"]:
        st.success("Поздравляем, заявка одобрена!")
    else:
        st.success("Подобрали для вас альтернативную карту")