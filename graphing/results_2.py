import pandas as pd
import plotly.express as px

df = pd.read_csv("base_aug.csv")

fig = px.line(
    df,
    x="epoch",
    y=["base_dataset", "augmented_dataset"],  # <== multiple y-columns
    labels={"value": "F1 Score", "variable": "Dataset", "epoch": "Epoch"},
    title="F1 Score per Epoch for Base vs Augmented Dataset"
)

fig.show()