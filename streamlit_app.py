import pandas as pd
import streamlit as st

@st.cache_data
def load_data():
    return pd.read_pickle('all_episodes_similarity_df.pkl')

def filter_df(df:pd.DataFrame, title:str):
    return df[["Rank", title]]

st.title('Stuff You Should Know Episode Similarity App')
st.write("""
         This recomendation app was built with natural language processing
         and topic modeling, check out more about how this app was built
         [here](https://jmoro0408.github.io/project/podcast-recommender).""")

df = load_data()

modification_container = st.container()
with modification_container:
    to_compare = df.drop('Rank', axis = 1).columns.to_list()
    to_filter_columns = st.selectbox("Choose episode to compare", to_compare)

st.table(filter_df(df, to_filter_columns))