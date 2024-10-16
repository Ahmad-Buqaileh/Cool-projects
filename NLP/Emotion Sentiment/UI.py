import streamlit as st
from emotion_model import Model
import psycopg2


def insert_output(User_Input, Result):
    cur.execute('INSERT INTO sentemint(user_text,label) VALUES (%s,%s)', (user_input, Result))
    connection.commit()


connection = psycopg2.connect(host='localhost', # I used a local host
                              database='data_base_name_here',
                              user='user_name_here',
                              password='your_password_here'
                              )
cur = connection.cursor()
connection.commit()

trained_model = Model()

st.title("Emotion Detection")
st.markdown("<br><br>", unsafe_allow_html=True)

user_input = st.text_area("Text to analyze")

st.markdown("<br><br>", unsafe_allow_html=True)

if st.button('Analyze'):
    if len(user_input) > 24:
        if user_input:
            prediction = trained_model.predict(user_input)
            output = trained_model.get_sentiment_label(prediction)
            st.write(f"## Emotion: {output}")
            insert_output(user_input, output)
        else:
            st.markdown("## Please enter some text")
    else:
        st.markdown("## Please enter more the 24 characters")
