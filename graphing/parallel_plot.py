import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load your data (adjust path or paste data directly)
root_dir = 'graphing/data'
file_name = 'bayesian_all'
df = pd.read_csv(f'{root_dir}/{file_name}.csv')

# Create parallel coordinates plot
fig = px.parallel_coordinates(
    df,
    dimensions=['per_device_train_batch_size', 'learning_rate', 'weight_decay', 'best_epoch', 'best_f1'],
    color='best_f1',
    color_continuous_scale=px.colors.sequential.Plasma,
    labels={
        'per_device_train_batch_size': 'Batch Size',
        'learning_rate': 'Learning Rate',
        'weight_decay': 'Weight Decay',
        # 'num_train_epochs': 'Epoch',
        'best_epoch': 'Best Epoch',
        'best_f1': 'F1 Score',
    }
)

fig.show()
