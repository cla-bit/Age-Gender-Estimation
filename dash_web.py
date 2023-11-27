import plotly.express as px
from dash import Dash, html, dcc, dash_table
from dash.dependencies import Input, Output
from pathlib import Path
from age_gender_estimation1 import FaceDataset

# get the dataset
dataset = FaceDataset(Path('dataset/UTKFace/'))

# Create the Dash app
app = Dash(__name__)

# bar chart of age distribution
age_bar_figure = px.bar(
    x=dataset.df['AGE GROUP'].value_counts().sort_index().values,
    y=dataset.df['AGE GROUP'].value_counts().sort_index().index,
    orientation='h',
    title='Age Distribution',
    color=dataset.df['AGE GROUP'].value_counts().sort_index().index,
    labels={'x': 'Count', 'y': 'Age Group'},
    height=500,
)

# bar chart of gender distribution
gender_bar_figure = px.pie(
    values=dataset.df['GENDER'].value_counts().values,
    names=dataset.df['GENDER'].value_counts().map({0: 'MALE', 1: 'FEMALE'}).index,
    title=f'Gender Distribution: MALE is 0 and FEMALE is 1',
    color=dataset.df['GENDER'].value_counts().map({0: 'MALE', 1: 'FEMALE'}).index,
    height=600,
    labels={'values': 'Count', 'names': 'Gender'},
)

# image grid columns and rows
image_grid_figure = dataset.plot_image_grid(row_no=5, cols_no=5)

# dataset layout table
dataset_layout = dash_table.DataTable(
    id='dataset-table',
    columns=[{"id": i, "name": i} for i in dataset.df.columns],
    data=dataset.df.to_dict('records'),
    style_table={'overflowY': 'scroll', 'height': '450px', 'width': 'auto',
                 'border': '1px solid black', 'textAlign': 'left', 'font-family': 'Arial', 'font-size': '20px'
                 },
    style_cell={'textAlign': 'left', 'font-family': 'Arial', 'font-size': '20px'},
    style_header={'backgroundColor': 'black', 'color': 'white', 'fontWeight': 'bold', 'textAlign': 'center',
                  'padding': '10px'},
    page_size=15,
)

# age group chart layout
age_group_chart_layout = dcc.Graph(
    id='age-group-chart',
    figure=age_bar_figure,
    style={'width': '100%', 'height': '100%'}
)

# gender recognition layout
gender_recognition_layout = dcc.Graph(
    id='gender-recognition',
    figure=gender_bar_figure
)

# plot image grid layout
image_grid_layout = html.Div(
    id='image-grid',
    children=[
        html.H1('Image Grid', style={
            'textAlign': 'center', 'font-size': '40px',
        }),
        dcc.Graph(
            id='image-grid-graph',
            style={'width': '100%', 'height': '100%'},
            figure=image_grid_figure
        )
    ]
)

# set the layout of the app
app.layout = html.Div(
    children=[
        html.H1('AGE GROUP AND GENDER RECOGNITION', style={
            'textAlign': 'center', 'font-size': '40px',
            'font-family': 'Arial', 'text-transform': 'uppercase'}),

        dcc.Tabs(id='tabs', value='dataset', children=[
            dcc.Tab(label='Dataset', value='dataset-table', children=[dataset_layout]),
            dcc.Tab(label='Age Group Chart', value='age-group-chart', children=[age_group_chart_layout]),
            dcc.Tab(label='Gender Recognition', value='gender-recognition', children=[gender_recognition_layout]),
            dcc.Tab(label='Image Grid', value='image-grid', children=[image_grid_layout]),
        ])
    ]
)


# callbacks for the app to switch between the layouts based on the button clicks
@app.callback(
    Output('dataset-table', 'children'),
    Output('age-group-chart', 'figure'),
    Output('gender-recognition', 'figure'),
    Output('image-grid', 'children'),
    Input('tabs', 'value')
)
def switch_layout(tab):
    if tab == 'dataset-table':
        return dataset_layout
    elif tab == 'age-group-chart':
        return age_group_chart_layout
    elif tab == 'gender-recognition':
        return gender_recognition_layout
    elif tab == 'image-grid':
        return image_grid_layout


if __name__ == '__main__':
    app.run_server(debug=True)
