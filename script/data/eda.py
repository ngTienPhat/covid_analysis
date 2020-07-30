import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# distribution of number of active cases:
def distribution_active_cases(df):
    fig=px.bar(df, x='Day', y='Active')
    fig.update_layout(title="Distribution of Number of Active Cases",
                      xaxis_title="Date",yaxis_title="Number of Cases",)
    fig.show()
