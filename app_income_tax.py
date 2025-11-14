import streamlit as st
import pandas as pd


@st.cache_data
def load_income_tax_data():
    df = pd.read_csv(
        "data/taxationbydistrict.csv",
        sep=";",
        encoding="ISO-8859-1",
        skiprows=7,
    )

    df.columns = [
        "Year",
        "Region_Code",
        "Region_Name",
        "Taxpayer_Count",
        "Total_Income_KEuros",
        "Total_Taxes_KEuros",
    ]

    numeric_cols = ["Taxpayer_Count", "Total_Income_KEuros", "Total_Taxes_KEuros"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    df["Tax_per_Taxpayer"] = (
            df["Total_Taxes_KEuros"] * 1000 / df["Taxpayer_Count"]
    ).round(2)

    return df

#Here is our streamlit UI

st.title("Income Tax by District & Political Impact")

st.write("This dataset contains income tax information by region in Germany.")

income_tax_df = load_income_tax_data()

st.subheader("Clean Data Preview")
st.dataframe(income_tax_df)

st.write("Number of rows:", income_tax_df.shape[0])
st.write("Number of columns:", income_tax_df.shape[1])

if st.checkbox("Show summary (describe)"):
    st.write(income_tax_df.describe(include="all"))
