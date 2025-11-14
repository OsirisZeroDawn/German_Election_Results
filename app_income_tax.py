import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt


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


@st.cache_data
def load_voting_data():
    """
    Load the harmonised federal municipal election data and keep only
    the top 6 parties, with CDU + CSU combined.
    """
    df = pd.read_csv("data/federal_muni_harm_21.csv")

    # Focus on the 2021 Bundestag election
    df = df[df["election_year"] == 2021].copy()

    # Ensure we have a combined CDU/CSU column
    if "cdu_csu" not in df.columns:
        df["cdu_csu"] = df["cdu"].fillna(0) + df["csu"].fillna(0)

    # Column with absolute valid votes (check this name in your CSV)
    vote_count_col = "valid_votes"

    # Top 6 parties we care about (vote shares, 0–1)
    party_cols = ["cdu_csu", "spd", "gruene", "fdp", "linke_pds", "afd"]

    # Keep region code at municipal and county level + election/votes/parties
    keep_cols = ["ags", "county", "election_year", vote_count_col] + party_cols
    df = df[keep_cols]

    return df, vote_count_col, party_cols


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

    summary_df = pd.DataFrame(
        {
            "Statistic": ["Minimum", "Average", "Maximum"],
            "Tax_per_Taxpayer": [
                income_tax_df["Tax_per_Taxpayer"].min(),
                income_tax_df["Tax_per_Taxpayer"].mean(),
                income_tax_df["Tax_per_Taxpayer"].max(),
            ],
        }
    )

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


# ---- Voting background: top 6 parties (CDU/CSU combined) ----

st.subheader("Voting Background – Top 6 Parties (Bundestag 2021)")

voting_top6_df, vote_count_col, party_cols = load_voting_data()

# For display, show party shares as percentages
voting_display = voting_top6_df.copy()
voting_display[party_cols] = (voting_display[party_cols] * 100).round(2)

st.write("Rows:", voting_display.shape[0], " | Columns:", voting_display.shape[1])
st.dataframe(voting_display.head(50))


# ---- Total votes by party (top 6) in millions using Plotly GO ----

st.subheader("Total Votes by Party (Bundestag 2021)")

voting_df, vote_count_col, party_cols = load_voting_data()

party_info = {
    "cdu_csu": ("CDU/CSU", "#000000"),      # black
    "spd": ("SPD", "#E3000F"),             # red
    "gruene": ("Greens", "#1FA12E"),       # green
    "fdp": ("FDP", "#FFED00"),             # yellow
    "linke_pds": ("Die Linke", "#800000"), # maroon
    "afd": ("AfD", "#00B2FF"),             # light blue
}

# Compute total votes per party: sum(share * valid_votes)
total_votes = []
labels = []
colors = []

for col in party_cols:
    label, color = party_info[col]
    votes_abs = (voting_df[col] * voting_df[vote_count_col]).sum()
    total_votes.append(votes_abs)
    labels.append(label)
    colors.append(color)

votes_millions = (np.array(total_votes) / 1_000_000).round(2)

fig_votes = go.Figure(
    data=[
        go.Bar(
            x=labels,
            y=votes_millions,
            marker_color=colors,
            text=votes_millions,
            texttemplate="%{text:.2f} M",
            textposition="outside",
        )
    ]
)

fig_votes.update_layout(
    title="Total Votes by Party (Millions, Bundestag 2021)",
    xaxis_title="Party",
    yaxis_title="Votes (millions)",
    template="plotly_white",
    yaxis=dict(tickformat=".1f"),
)

st.plotly_chart(fig_votes, use_container_width=True)


# ---- MERGE TAX DATA WITH VOTING DATA ----

st.subheader("Merged Dataset: Tax & Voting Information")

voting_df, vote_count_col, party_cols = load_voting_data()

# 1. Convert join keys to numeric safely
tax_for_merge = income_tax_df.copy()
vote_for_merge = voting_df.copy()

tax_for_merge["Region_Code_num"] = pd.to_numeric(
    tax_for_merge["Region_Code"], errors="coerce"
)
vote_for_merge["county_num"] = pd.to_numeric(
    vote_for_merge["county"], errors="coerce"
)

# 2. Drop rows where conversion failed
tax_for_merge = tax_for_merge.dropna(subset=["Region_Code_num"])
vote_for_merge = vote_for_merge.dropna(subset=["county_num"])

# 3. Merge on the cleaned numeric codes
merged_df = tax_for_merge.merge(
    vote_for_merge,
    left_on="Region_Code_num",
    right_on="county_num",
    how="inner",
)

st.write("Merged rows:", merged_df.shape[0])
st.write("Merged columns:", merged_df.shape[1])
st.dataframe(merged_df.head())


# ---- CREATE ANALYSIS DATAFRAME ----

analysis_cols = ["Tax_per_Taxpayer"] + party_cols
analysis_df = merged_df[analysis_cols].dropna()

st.subheader("Analysis DataFrame (Correlation Inputs)")
st.dataframe(analysis_df.head())

# ---- SCATTER PLOTS FOR EACH PARTY ----

st.subheader("Tax per Taxpayer vs Party Vote Share")

party_colors = {
    "cdu_csu": "#000000",
    "spd": "#E3000F",
    "gruene": "#1FA12E",
    "fdp": "#FFED00",
    "linke_pds": "#800000",
    "afd": "#00B2FF",
}

# Dropdown to choose the party to visualize
party_choice = st.selectbox("Choose a party:", party_cols)

fig_scatter = go.Figure()

fig_scatter.add_trace(go.Scatter(
    x=analysis_df["Tax_per_Taxpayer"],
    y=analysis_df[party_choice],
    mode="markers",
    marker=dict(color=party_colors[party_choice], size=8),
    name=party_choice
))

fig_scatter.update_layout(
    xaxis_title="Tax per Taxpayer (€)",
    yaxis_title=f"{party_choice} Vote Share",
    template="plotly_white",
    height=500,
)

st.plotly_chart(fig_scatter, use_container_width=True)

import plotly.express as px

if st.checkbox("Show regression line"):
    fig_reg = px.scatter(
        analysis_df,
        x="Tax_per_Taxpayer",
        y=party_choice,
        trendline="ols",
        opacity=0.6,
        hover_data=["Tax_per_Taxpayer"],
    )
    fig_reg.update_layout(
        xaxis_title="Tax per Taxpayer (€)",
        yaxis_title=f"{party_choice} Vote Share",
        template="plotly_white"
    )
    st.plotly_chart(fig_reg, use_container_width=True)

st.subheader("All Parties: Tax-per-Taxpayer Relationship")

rows = []
for col in party_cols:
    rows.append(go.Scatter(
        x=analysis_df["Tax_per_Taxpayer"],
        y=analysis_df[col],
        mode="markers",
        name=col,
        marker=dict(color=party_colors[col], size=6)
    ))

fig_multi = go.Figure(rows)
fig_multi.update_layout(
    template="plotly_white",
    xaxis_title="Tax per Taxpayer (€)",
    yaxis_title="Vote Share",
)
st.plotly_chart(fig_multi, use_container_width=True)




# ---- Vote Share (2021) by Income Bracket (with labels) ----

st.subheader("Vote Share by Income Bracket")

# 1. Create 5 quantile bins of Tax_per_Taxpayer
analysis_df["TaxBin"] = pd.qcut(
    analysis_df["Tax_per_Taxpayer"],
    5,
    labels=False
)

# 2. Median tax per taxpayer for each bin → used as x-axis labels
bin_labels = (
    analysis_df.groupby("TaxBin")["Tax_per_Taxpayer"]
    .median()
    .round(0)
    .astype(int)
)

# 3. Mean vote share for each party in each bin
mean_by_bin = analysis_df.groupby("TaxBin")[party_cols].mean()

# 4. Convert from fractions (0–1) to percentages
mean_by_bin_percent = (mean_by_bin * 100).round(1)

# 5. Build stacked bar chart with readable labels
fig_bins = go.Figure()

for party in party_cols:
    fig_bins.add_trace(go.Bar(
        x=bin_labels.index,                    # internal bin index 0–4
        y=mean_by_bin_percent[party],
        name=party,
        marker_color=party_colors.get(party, "#666666"),
    ))

fig_bins.update_layout(
    barmode="stack",
    template="plotly_white",
    height=500,
    xaxis=dict(
        title="Income Group (Median Tax per Taxpayer in €)",
        tickmode="array",
        tickvals=bin_labels.index,             # 0–4
        ticktext=[f"€{v:,.0f}" for v in bin_labels],  # human labels e.g. €5,300
    ),
    yaxis=dict(
        title="Average Vote Share (%)",
        ticksuffix="%",
        range=[0, 100],
    ),
    legend_title="Party",
)

st.plotly_chart(fig_bins, use_container_width=True)









st.subheader("Vote Share Heatmap by Tax Level")
fig_heat = px.imshow(
    mean_by_bin,
    labels=dict(x="Party", y="Income Bin", color="Vote Share"),
    text_auto=True,
    color_continuous_scale="RdBu"
)
st.plotly_chart(fig_heat, use_container_width=True)
