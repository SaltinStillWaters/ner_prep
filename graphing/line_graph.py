import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

df = pd.read_csv("grid_results.csv")

fig = px.line(
    df,
    x="epoch",
    y=["model_a_f1", "model_b_f1"],
    labels={"value": "F1 Score", "epoch": "Epoch", "variable": "Model"},
    title="F1 Score per Epoch for Two Models"
)

fig.show()