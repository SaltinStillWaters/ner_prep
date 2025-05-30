import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load your data (adjust path or paste data directly)
df = pd.read_csv("grid_results.csv")

# Create parallel coordinates plot
fig = px.parallel_coordinates(
    df,
    dimensions=['batch_size', 'learning_rate', 'epoch', 'weight_decay', 'f1'],
    color='f1',
    color_continuous_scale=px.colors.sequential.Plasma,
    labels={
        'batch_size': 'Batch Size',
        'learning_rate': 'Learning Rate',
        'epoch': 'Epoch',
        'weight_decay': 'Weight Decay',
        'f1': 'F1 Score'
    }
)

fig.show()
