import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


@st.cache_data
def load_income_tax_data():
    """Load and clean the German income tax dataset."""
    df = pd.read_csv(
        "data/taxationbydistrict.csv",
        sep=";",
        encoding="ISO-8859-1",
        skiprows=7,  # skip metadata lines at the top
    )

    # Rename columns to English
    df.columns = [
        "Year",
        "Region_Code",
        "Region_Name",
        "Taxpayer_Count",
        "Total_Income_KEuros",
        "Total_Taxes_KEuros",
    ]

    # Ensure numeric columns are actually numbers
    numeric_cols = ["Taxpayer_Count", "Total_Income_KEuros", "Total_Taxes_KEuros"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # Tax paid per taxpayer (convert from thousands of euros to euros)
    df["Tax_per_Taxpayer"] = (
        df["Total_Taxes_KEuros"] * 1000 / df["Taxpayer_Count"]
    ).round(2)

    return df


# ----------------- STREAMLIT UI -----------------

st.title("Income Tax & Political Impact")

st.write(
    "This dataset contains income tax information by region in Germany, "
    "and later will be compared with voting patterns."
)

# Load cleaned data once
income_tax_df = load_income_tax_data()

# ---- Data preview ----
st.subheader("Districts in Germany by Taxpayer & Total Income")
st.dataframe(income_tax_df)

st.write("Number of rows:", income_tax_df.shape[0])
st.write("Number of columns:", income_tax_df.shape[1])

if st.checkbox("Show summary (describe)"):
    st.write(income_tax_df.describe(include="all"))

# ---- Summary: min / avg / max of Tax_per_Taxpayer ----
if st.checkbox("Show Tax per Taxpayer Summary"):

    summary_df = pd.DataFrame({
        "Statistic": ["Minimum", "Average", "Maximum"],
        "Tax_per_Taxpayer": [
            income_tax_df["Tax_per_Taxpayer"].min(),
            income_tax_df["Tax_per_Taxpayer"].mean(),
            income_tax_df["Tax_per_Taxpayer"].max(),
        ],
    })

    st.subheader("Tax per Taxpayer – Summary")

# Bar chart: three bars side-by-side
    st.bar_chart(summary_df.set_index("Statistic"))

# Optional: show exact values below, nicely formatted
    st.table(summary_df.style.format({"Tax_per_Taxpayer": "{:,.2f}"}))

# ---- Top 10 / Bottom 10 districts using Plotly GO ----

st.subheader("Top & Bottom Districts by Tax per Taxpayer")

# Drop rows with missing values
tax_df = income_tax_df.dropna(subset=["Tax_per_Taxpayer"])

# Compute Top 10 and Bottom 10
top10 = tax_df.nlargest(10, "Tax_per_Taxpayer")
bottom10 = tax_df.nsmallest(10, "Tax_per_Taxpayer")

# -------- TOP 10 FIGURE --------
fig_top = go.Figure(
    data=[
        go.Bar(
            x=top10["Region_Name"],
            y=top10["Tax_per_Taxpayer"],
            text=top10["Tax_per_Taxpayer"].round(2),
            textposition="auto",
            marker_color="green",
        )
    ]
)

fig_top.update_layout(
    title="Top 10 Districts – Highest Tax per Taxpayer",
    xaxis_title="District",
    yaxis_title="Tax per taxpayer (€)",
    template="plotly_white",
    height=500,
)

st.plotly_chart(fig_top, use_container_width=True)

# -------- BOTTOM 10 FIGURE --------
fig_bottom = go.Figure(
    data=[
        go.Bar(
            x=bottom10["Region_Name"],
            y=bottom10["Tax_per_Taxpayer"],
            text=bottom10["Tax_per_Taxpayer"].round(2),
            textposition="auto",
            marker_color="crimson",
        )
    ]
)

fig_bottom.update_layout(
    title="Bottom 10 Districts – Lowest Tax per Taxpayer",
    xaxis_title="District",
    yaxis_title="Tax per taxpayer (€)",
    template="plotly_white",
    height=500,
)

st.plotly_chart(fig_bottom, use_container_width=True)

