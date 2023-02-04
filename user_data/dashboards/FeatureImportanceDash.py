import streamlit as st
import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Connect to the SQLite database
conn = sqlite3.connect("litmus.sqlite")
# Query the data
data = pd.read_sql_query("SELECT * FROM feature_importance_history", conn)
# Disconnect from the database
conn.close()


# Print
st.dataframe(data)

# Boxplot
data = data[data["pair"] == "SOL/USDT:USDT"]
ranks = data.groupby("feature_id")["importance"].mean().sort_values()[::-1].index

sns.set_theme(style="ticks")

fig, ax = plt.subplots(figsize=(7, 200))
sns.boxplot(x="importance", y="feature_id", data=data,
            whis=[0, 100], width=.6, palette="vlag", order=ranks)
ax.xaxis.grid(True)
ax.set(ylabel="")
sns.despine(trim=True, left=True)
st.pyplot(fig)
