import plotly.express as px
import numpy as np
df = px.data.gapminder().query("year == 2007")
fig = px.treemap(df, path=[px.Constant("world"), 'continent', 'country'], values='pop',
                  color='pop',
                  color_continuous_scale='viridis'
)
fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
fig.write_html('figure_teste.html', auto_open=True)