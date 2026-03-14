import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
from google import genai
import json
import os
from dotenv import load_dotenv

# ---------------- LOAD ENV ----------------

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

st.title("Conversational AI Business Intelligence Dashboard")

st.write("Ask questions about your dataset using natural language")

# ---------------- DATASET SELECTION ----------------

uploaded_file = st.file_uploader("Upload a CSV dataset (optional)", type=["csv"])

if uploaded_file is not None:

    df_uploaded = pd.read_csv(uploaded_file)

    conn = sqlite3.connect(":memory:")
    df_uploaded.to_sql("data", conn, index=False, if_exists="replace")

    dataset_name = "data"
    columns = ", ".join(df_uploaded.columns)

    st.success("Using uploaded dataset")

    st.write("Dataset shape:", df_uploaded.shape)

    st.subheader("Dataset Preview")
    st.dataframe(df_uploaded.head(10))

else:

    conn = sqlite3.connect("customers.db")

    df_temp = pd.read_sql_query("SELECT * FROM customers LIMIT 1", conn)

    dataset_name = "customers"
    columns = ", ".join(df_temp.columns)

    st.info("Using default dataset")

# ---------------- QUESTION INPUT ----------------

question = st.text_input("Enter your question")

# ---------------- GEMINI CACHE ----------------

@st.cache_data(ttl=3600)
def ask_gemini(prompt):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response


# ---------------- DATABASE CACHE ----------------

@st.cache_data
def run_query(sql_query):
    return pd.read_sql_query(sql_query, conn)


# ---------------- MAIN LOGIC ----------------

if st.button("Generate Dashboard"):

    if question.strip() == "":
        st.warning("Please enter a question.")
        st.stop()

    prompt = f"""
You are a data analyst.

Database table: {dataset_name}

Columns:
{columns}

User question:
{question}

Return ONLY valid JSON.

The SQL must return TWO columns:
1) category column
2) numeric value column

Example:
SELECT city_tier, COUNT(*) AS value FROM {dataset_name} GROUP BY city_tier

Format:

{{
"sql":"SQL_QUERY",
"chart":"bar|line|pie"
}}
"""

    response = None

    try:

        response = ask_gemini(prompt)

        text = response.text.strip()

        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        json_text = text[json_start:json_end]

        result = json.loads(json_text)

        sql = result["sql"]
        chart = result["chart"]

        st.subheader("Generated SQL")
        st.code(sql)

        df = run_query(sql)

        if df.empty:
            st.warning("Query returned no results.")
            st.stop()

        st.subheader("Data Preview")
        st.dataframe(df)

        # ---------------- CHART GENERATION ----------------

        if len(df.columns) < 2:

            st.warning("Not enough columns for visualization.")
            st.dataframe(df)

        else:

            if chart == "bar":
                fig = px.bar(df, x=df.columns[0], y=df.columns[1])

            elif chart == "line":
                fig = px.line(df, x=df.columns[0], y=df.columns[1])

            elif chart == "pie":
                fig = px.pie(df, names=df.columns[0], values=df.columns[1])

            else:
                st.warning("Unknown chart type. Showing table.")
                st.dataframe(df)
                st.stop()

            st.subheader("Visualization")
            st.plotly_chart(fig)

    except Exception as e:

        st.error("Something went wrong.")

        if response:
            st.write("Gemini response:")
            st.write(response.text)

        st.write("Error details:", e)