# Import necessary libraries
import pandas as pd 
from dtype_diet import report_on_dataframe, optimize_dtypes
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd
# Read data for November
df_nov = pd.read_csv('./data/precos-gasolina-etanol-11.csv', delimiter=';')

# Clean and preprocess columns in the November dataframe
df_nov['Valor de Venda'] = df_nov['Valor de Venda'].str.replace(',', '.').astype('float') 
df_nov['Data da Coleta'] = pd.to_datetime(df_nov['Data da Coleta'], format="%d/%m/%Y")
df_nov['Dia da Coleta'] = df_nov['Data da Coleta'].dt.day

# Generate and apply optimized data types for the November dataframe
proposed_df = report_on_dataframe(df_nov, unit="MB")
df_nov = optimize_dtypes(df_nov, proposed_df)

# Read data for December
df_dez = pd.read_csv('./data/precos-gasolina-etanol-12.csv', delimiter=';')

# Clean and preprocess columns in the December dataframe
df_dez['Valor de Venda'] = df_dez['Valor de Venda'].str.replace(',', '.').astype('float') 
df_dez['Data da Coleta'] = pd.to_datetime(df_dez['Data da Coleta'], format="%d/%m/%Y")
df_dez['Dia da Coleta'] = df_dez['Data da Coleta'].dt.day

# Generate and apply optimized data types for the December dataframe
proposed_df = report_on_dataframe(df_dez, unit="MB")
df_dez = optimize_dtypes(df_dez, proposed_df)

# Define a function to filter DataFrame and apply aggregation
def operation_df(df, column1, operation, product=None, column2=None):
    """
    Filter DataFrame and apply aggregation.

    Parameters: df, column1, product, column2, operation

    Returns: Aggregation result.
    """
    if (product and column2) is not None:
        # Apply aggregation to a specific product and column2 (if provided)
        result = df[df[column1] == product][column2].agg(operation)
        return result
    else:
        # Apply aggregation to the entire column1 (if product and column2 are not provided)
        result = df[column1].agg(operation)
        return result

# List of products for which aggregation will be performed
product = ['GASOLINA', 'GASOLINA ADITIVADA', 'ETANOL']

# 1. Como se comportaram o preço dos combustíveis durante os dois meses citados? Os valores do 
# etanol e da gasolina tiveram uma tendência de queda ou diminuição?

# Agrupar por produto e dia da coleta, calculando o preço médio para novembro
df_nov1 = df_nov.groupby(['Produto','Dia da Coleta'])['Valor de Venda'].mean().reset_index()

# Agrupar por produto e dia da coleta, calculando o preço médio para dezembro
df_dez1 = df_dez.groupby(['Produto', 'Dia da Coleta'])['Valor de Venda'].mean().reset_index()

data_gas_nov = df_nov1[df_nov1['Produto']==product[0]]
data_gas_dez = df_dez1[df_dez1['Produto']==product[0]]

data_etn_nov = df_nov1[df_nov1['Produto']==product[2]]
data_etn_dez = df_dez1[df_dez1['Produto']==product[2]]

# Concatenar dois dataframes df_nov1 e df_dez1 em data_mimmax
data_mimmax = pd.concat([df_nov1, df_dez1])

# Converter a coluna 'Valor de Venda' para o tipo float
data_mimmax['Valor de Venda'].astype('float')

# Filtrar dados para cada tipo de produto (armazenado na lista 'product')
data_mimmax_gas = data_mimmax[data_mimmax['Produto']==product[0]]
max_gas_line = data_mimmax_gas['Valor de Venda'].max()
max_gas_line = [max_gas_line] * len(data_gas_nov['Dia da Coleta'])
max_gas_marker = data_mimmax_gas.nlargest(1, 'Valor de Venda')

min_gas_line = data_mimmax_gas['Valor de Venda'].min()
min_gas_line = [min_gas_line] * len(data_gas_nov['Dia da Coleta'])
min_gas_marker = data_mimmax_gas.nsmallest(1, 'Valor de Venda')


data_mimmax_etn = data_mimmax[data_mimmax['Produto']==product[2]]
max_etn_line = data_mimmax_etn['Valor de Venda'].max()
max_etn_line = [max_etn_line] * len(data_etn_nov['Dia da Coleta'])
max_etn_marker = data_mimmax_etn.nlargest(1, 'Valor de Venda')

min_etn_line = data_mimmax_etn['Valor de Venda'].min()
min_etn_line = [min_etn_line] * len(data_etn_nov['Dia da Coleta'])
min_etn_marker = data_mimmax_etn.nsmallest(1, 'Valor de Venda')

# Calcular os preços médios para cada dia de coleta em novembro
data_all_nov = df_nov.groupby(['Dia da Coleta'])['Valor de Venda'].mean().reset_index()

# Calcular os preços médios para cada dia de coleta em dezembro
data_all_dez = df_dez.groupby(['Dia da Coleta'])['Valor de Venda'].mean().reset_index()

# Mesclar os preços médios de novembro e dezembro no dia da coleta
data_all = pd.merge(data_all_nov, data_all_dez, on='Dia da Coleta', how='inner')

# Concatenar os preços médios de novembro e dezembro em data_all_minmax
data_all_minmax = pd.concat([data_all_dez, data_all_nov])

# Converter a coluna 'Valor de Venda' para o tipo float para data_all_minmax
all_max_y = data_all_minmax['Valor de Venda'].max()
all_max_y = [all_max_y] * len(data_all['Dia da Coleta'])
all_min_y = data_all_minmax['Valor de Venda'].min()
all_min_y = [all_min_y] * len(data_all['Dia da Coleta'])

all_max_marker = data_all_minmax.nlargest(1, 'Valor de Venda')
all_min_marker = data_all_minmax.nsmallest(1, 'Valor de Venda')

all_fuels =[
    go.Scatter(
        x=data_all['Dia da Coleta'], y=data_all['Valor de Venda_x'], 
        mode='lines', name='Novembro', 
        line=dict(color='#0057e7', width=6),
        connectgaps=True,
        legend='legend1',
        legendgroup='group1'
    ),

    go.Scatter(
        x=data_all['Dia da Coleta'], y=data_all['Valor de Venda_y'], 
        mode='lines', name='Dezembro', 
        line=dict(color='#ffa700', width=6),
        connectgaps=True,
        legend='legend1',
        legendgroup='group1'
    ),

    go.Scatter(
        x=data_all['Dia da Coleta'], y=all_max_y,
        mode='lines',
        line=dict(color='#008744', width=4, dash='dashdot'),
        showlegend=False,
        legendgroup='group1'
    ),

    go.Scatter(
        x=data_all['Dia da Coleta'], y=all_min_y,
        mode='lines',
        line=dict(color='#d62d20', width=4, dash='dashdot'),
        showlegend=False,
        legendgroup='group1'
    ),

    go.Scatter(
        mode='markers',
        x=all_max_marker['Dia da Coleta'], y=all_max_marker['Valor de Venda'],
        marker=dict(symbol='circle', size=12, color='#008744'),
        showlegend=False,
        legend='legend1',
        legendgroup='group1'
    ),

    go.Scatter(
        mode='markers',
        x=all_min_marker['Dia da Coleta'], y=all_min_marker['Valor de Venda'],
        marker=dict(symbol='circle', size=12, color='#d62d20'),
        showlegend=False,
        legend='legend1',
        legendgroup='group1'
    )
]

gas =[   
    go.Scatter(
        x=data_gas_nov['Dia da Coleta'], y=data_gas_nov['Valor de Venda'],
        mode='lines', name= 'Novembro',
        line=dict(color='#0057e7', width=6),
        legend='legend2',
        legendgroup='group2'
    ),

    go.Scatter(
        x=data_gas_dez['Dia da Coleta'], y=data_gas_dez['Valor de Venda'],
        mode='lines', name='Dezembro',
        line=dict(color='#ffa700', width=6),
        legend='legend2',
        legendgroup='group2' 
    ),

    go.Scatter(
        x=data_gas_nov['Dia da Coleta'], y=max_gas_line,
        mode='lines', 
        line=dict(color='#008744', width=6, dash='dashdot'),
        legend='legend2',
        showlegend=False,
        legendgroup='group2'
    ),

    go.Scatter(
        x=data_gas_nov['Dia da Coleta'], y=min_gas_line,
        mode='lines', 
        line=dict(color='#d62d20', width=6, dash='dashdot'),
        showlegend=False,
        legend='legend2',
        legendgroup='group2'
    ),

    go.Scatter(
        x=max_gas_marker['Dia da Coleta'], y=max_gas_marker['Valor de Venda'],
        mode='markers',
        marker=dict(symbol='circle', size=12, color='#008744'),
        showlegend=False,
        legend='legend2',
        legendgroup='group2'
    ),

    go.Scatter(
        x=min_gas_marker['Dia da Coleta'], y=min_gas_marker['Valor de Venda'],
        mode='markers',
        marker=dict(symbol='circle', size=12, color='#d62d20'),
        showlegend=False,
        legend='legend2',
        legendgroup='group2'
    )
]

etn = [
    go.Scatter(
        x=data_etn_nov['Dia da Coleta'], y=data_etn_nov['Valor de Venda'],
        mode='lines', name= 'Novembro',
        line=dict(color='#0057e7', width=6),
        legend='legend3',
        legendgroup='group3'
    ),

    go.Scatter(
        x=data_etn_dez['Dia da Coleta'], y=data_etn_dez['Valor de Venda'],
        mode='lines', name='Dezembro',
        line=dict(color='#ffa700', width=6),
        legend='legend3',
        legendgroup='group3'
    ),

    go.Scatter(
        x=data_etn_nov['Dia da Coleta'], y=max_etn_line,
        mode='lines', 
        line=dict(color='#008744', width=6, dash='dashdot'),
        legend='legend3',
        legendgroup='group3',
        showlegend=False
    ),

    go.Scatter(
        x=data_etn_nov['Dia da Coleta'], y=min_etn_line,
        mode='lines', 
        line=dict(color='#d62d20', width=6, dash='dashdot'),
        showlegend=False,
        legend='legend3',
        legendgroup='group3'
    ),

    go.Scatter(
        x=max_etn_marker['Dia da Coleta'], y=max_etn_marker['Valor de Venda'],
        mode='markers',
        marker=dict(symbol='circle', size=12, color='#008744'),
        showlegend=False,
        legend='legend3',
        legendgroup='group3'
    ),

    go.Scatter(
        x=min_etn_marker['Dia da Coleta'], y=min_etn_marker['Valor de Venda'],
        mode='markers',
        marker=dict(symbol='circle', size=12, color='#d62d20'),
        showlegend=False,
        legend='legend3',
        legendgroup='group3'
    )
]

fig = go.Figure()

for trace in gas:
    fig.add_trace(trace)

for trace in all_fuels:
    fig.add_trace(trace)

for trace in etn:
    fig.add_trace(trace)

fig.update_layout(
    legend1=dict(title='Todos os Combustíveis', xref='container', yref='container', x=1, y=0.65),
    legend2=dict(title='Gasolina', xref='container', yref='container',x=0.958, y=0.55),
    legend3=dict(title='Etanol', xref='container', yref='container', x=0.958, y=0.45)
)

fig.write_html('html/figure1.html', auto_open=True)

# 2. Qual o preço médio da gasolina e do etanol nesses dois meses?

# Criar um DataFrame para os valores médios e contagens de cada produto em novembro
data_nov = pd.DataFrame({"Produto": product, "Valor Médio": '', "Mês": "Novembro"})

# Calcular os valores médios para cada produto em novembro
data_nov['Valor Médio'] = [operation_df(df_nov, 'Produto', 'mean', p, 'Valor de Venda') for p in product]

# Calcular o valor médio global para todos os combustíveis em novembro
data_nov['Valor Médio de Todos'] = operation_df(df_nov, 'Valor de Venda', 'mean')

# Criar um DataFrame para os valores médios e contagens de cada produto em dezembro
data_dez = pd.DataFrame({"Produto": product, "Valor Médio": '', "Mês": "Dezembro"})

# Calcular os valores médios para cada produto em dezembro
data_dez['Valor Médio'] = [operation_df(df_dez, 'Produto', 'mean', p, 'Valor de Venda') for p in product]

# Calcular as contagens para cada produto em novembro e dezembro
data_nov['Quantidade'] = [operation_df(df_nov, 'Produto', 'count', p, 'Valor de Venda') for p in product]
data_dez['Quantidade'] = [operation_df(df_dez, 'Produto', 'count', p, 'Valor de Venda') for p in product]

# Mesclar os dados de novembro e dezembro para análises adicionais
data_merged = pd.merge(data_nov.drop(columns=['Mês']), data_dez.drop(columns=['Mês']), on='Produto', how='inner')
data_merged = data_merged.drop(data_merged[data_merged['Produto']==product[1]].index)

data_nov = data_nov.drop(data_nov[data_nov['Produto']==product[1]].index)
data_dez = data_dez.drop(data_dez[data_dez['Produto']==product[1]].index)

fig = make_subplots(
    cols=2, rows=1,
    subplot_titles=['Média de Preços de Novembro até Dezembro', 'Distribuição de Combustíveis nos Dois Meses'],
    specs=[[{'type': 'Bar'}, {'type': 'Pie'}]]
)

# Gráfico de Barras
bar_trace_nov = go.Bar(
    x=data_nov['Produto'], y=data_nov['Valor Médio'],
    name='Novembro', marker_color='#0057e7',
    text=data_nov['Valor Médio'],
    legendgroup='group1',
    legendgrouptitle=dict(text='Preço Médio'),
    legend='legend1'
)

bar_trace_dez = go.Bar(
    x=data_dez['Produto'], y=data_dez['Valor Médio'],
    name='Dezembro', marker_color='#ffa700',
    text=data_dez['Valor Médio'],
    legendgroup='group1'
)

fig.add_trace(bar_trace_nov, col=1, row=1)
fig.add_trace(bar_trace_dez, col=1, row=1)

fig.data[0].update(texttemplate='%{text:.3s}', textposition='outside')
fig.data[1].update(texttemplate='%{text:.3s}', textposition='outside')
fig.update_layout(barmode='group')

# Gráfico de Pizza
pie_trace = go.Pie(
    labels=data_merged['Produto'],
    values=data_merged[['Quantidade_x', 'Quantidade_y']].sum(axis=1),
    legend='legend2',
    legendgroup='group2',
    legendgrouptitle=dict(text='Distribuição Combustíveis')
)

fig.add_trace(pie_trace, col=2, row=1)

fig.data[2].update(hoverinfo='label+percent', textinfo='percent', textfont_size=20,
                marker=dict(colors=['#5d5d5d', '#2a623d'], line=dict(color='#ffcf40', width=2)))

fig.update_layout(
    legend=dict(x=0.1, y=0.8, xref="container", yref="container", traceorder='grouped'),
    legend2=dict(x=1, y=0.8, xref="container", yref="container", traceorder='grouped'),
)

fig.write_html('html/figure2.html', auto_open=True)

# 3. Quais os 5 estados com o preço médio da gasolina e do etanol mais caros?

# Concatene os dataframes de novembro e dezembro
df = pd.concat([df_nov, df_dez], axis=0, ignore_index=True)
df = df.drop(df[df['Produto']=='GASOLINA ADITIVADA'].index)

# Certifique-se de que a coluna 'Estado - Sigla' seja tratada como string
df['Estado - Sigla'] = df['Estado - Sigla'].astype(str) 

# Agrupe por estado e produto, calcule o preço médio e ordene os valores em ordem decrescente
df_gas = df[df['Produto']==product[0]]
df_gas = df_gas.groupby(['Municipio', 'Estado - Sigla'])['Valor de Venda'].mean().reset_index()
df_gas = df_gas.groupby('Estado - Sigla').apply(lambda x: x.nlargest(5, 'Valor de Venda')).reset_index(drop=True)
df_gas_filter = df_gas.groupby('Estado - Sigla')['Valor de Venda'].mean().nlargest(5).reset_index()
df_gas = df_gas[df_gas['Estado - Sigla'].isin(df_gas_filter['Estado - Sigla'].tolist())]

df_eta = df[df['Produto']==product[2]]
df_eta = df_eta.groupby(['Municipio', 'Estado - Sigla'])['Valor de Venda'].mean().reset_index()
df_eta = df_eta.groupby('Estado - Sigla').apply(lambda x: x.nlargest(5, 'Valor de Venda')).reset_index(drop=True)
df_eta_filter = df_eta.groupby('Estado - Sigla')['Valor de Venda'].mean().nlargest(5).reset_index()
df_eta = df_eta[df_eta['Estado - Sigla'].isin(df_eta_filter['Estado - Sigla'].tolist())]

df_concat = pd.concat([df_gas, df_eta], axis=0, ignore_index=True)

# Criar treemaps
figs = [
    px.treemap(df_gas,
        path=[px.Constant('Estados'), 'Estado - Sigla', 'Municipio'], 
        values='Valor de Venda',
        custom_data=['Valor de Venda', 'Municipio'],
        title='Top 5 Estados com Maior Preço Médio de Gasolina',
        labels={'Valor de Venda': 'Preço Médio'}
    ),

    px.treemap(df_eta,
        path=[px.Constant('Estados'), 'Estado - Sigla', 'Municipio'], 
        values='Valor de Venda', 
        custom_data='Valor de Venda',
        title='Top 5 Estados com Maior Preço Médio de Etanol',
        labels={'Valor de Venda': 'Preço Médio'}
    )
]

# Criar subplot com treemaps para gasolina e etanol
fig = make_subplots(rows=1, cols=len(figs),
                    subplot_titles=['Gasolina', 'Etanol'],
                    specs=[[{"type":'treemap'}, {'type':'treemap'}]])

# Adicionar treemaps ao subplot
for i, figure in enumerate(figs):
    for trace in range(len(figure["data"])):
        fig.append_trace(figure['data'][trace], row=1, col=i+1)

fig.update_traces(root_color='lightgrey')
fig.update_traces(maxdepth=2)
fig.update_traces(textinfo='label+value',
                texttemplate=
                '<b>%{label}</b>\
                <br><b><span style="font-size:16px">%{value:,.2f}</span></b>',
                hovertemplate='Preço Médio: <b>%{customdata[0]:,.2f}</b>')

fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))

# Atualizar layout com título e legenda
fig.update_layout(title_text='<b>Top 5 Estados com os Maiores Preços Médios de Gasolina e Etanol</b>', 
                title_x=0.5)  # centralizar o título

# Salvar o arquivo HTML
fig.write_html('html/figure3.html', auto_open=True)

# 4. Qual o preço médio da gasolina e do etanol por estado?

# Concatene os dataframes df_nov e df_dez ao longo do eixo 0, ignorando os índices existentes
df = pd.concat([df_nov, df_dez], axis=0, ignore_index=True)

# Converta a coluna 'Estado - Sigla' para o tipo de dado string
df['Estado - Sigla'] = df['Estado - Sigla'].astype(str)

# Agrupe os dados e calcule a média do 'Valor de Venda' para cada combinação de 'Regiao - Sigla', 'Estado - Sigla' e 'Produto'
df_data = df.groupby(['Regiao - Sigla', 'Estado - Sigla', 'Produto'])['Valor de Venda'].mean()

# Remova valores nulos e redefina o índice do dataframe resultante
data = df_data.dropna().reset_index()

# Defina as cores a serem usadas nos gráficos e as regiões correspondentes
cores = ['#d11141', '#00b159', '#00aedb', '#f37735', '#ffc425']
regioes = ['N', 'S', 'SE', 'NE', 'CO']

# Crie dicionários para armazenar os gráficos de gasolina e etanol
figure_gasolina = {}
figure_etanol = {}

# Inicialize os dataframes específicos para gasolina e etanol
gasolina = data[data['Produto'] == 'GASOLINA']
etanol = data[data['Produto'] == 'ETANOL']

# Loop para criar os gráficos para cada região e produto
for idx_regiao, regiao in enumerate(regioes):
    regiao_data_gas = gasolina[gasolina['Regiao - Sigla'] == regiao]
    regiao_data_etn = etanol[etanol['Regiao - Sigla'] == regiao]
    
    # Crie um gráfico de dispersão para cada estado na região para gasolina
    trace_gas = go.Scatter(
        x=regiao_data_gas['Estado - Sigla'],
        y=regiao_data_gas['Valor de Venda'],
        mode='markers',
        marker_color=cores[idx_regiao],
        text=regiao_data_gas['Estado - Sigla'],
        name=regiao,
        legend=f'legend{1}'
    )
    
    # Crie um gráfico de dispersão para cada estado na região para etanol
    trace_etn = go.Scatter(
        x=regiao_data_etn['Estado - Sigla'],
        y=regiao_data_etn['Valor de Venda'],
        mode='markers',
        marker_color=cores[idx_regiao],
        text=regiao_data_etn['Estado - Sigla'],
        name=regiao,
        legend=f'legend{2}'
    )
    
    # Adicione os traces aos dicionários correspondentes, garantindo que não seja None
    chave_figura_gas = f'fig_gasolina_{regiao.lower()}'
    chave_figura_etn = f'fig_etanol_{regiao.lower()}'
    
    figure_gasolina[chave_figura_gas] = trace_gas
    figure_etanol[chave_figura_etn] = trace_etn

# Crie um layout para as legendas
layout = {'legend1': dict(y=0.81, x=0.1, xref='container', yref='container', title='Gasolina'),
        'legend2': dict(y=0.808, x=0.9, xref='container', yref='container', title='Etanol')}

# Crie as figuras com subgráficos para gasolina e etanol
fig = make_subplots(
    cols=2, rows=1,
    subplot_titles=['Preço Médio da Gasolina por Estado', 'Preço Médio do Etanol por Estado'],
    specs=[[{"type": 'Scatter'}, {'type': 'Scatter'}]]
)

# Adicione os gráficos de gasolina e etanol aos subgráficos
fig.add_traces(list(figure_gasolina.values()), cols=1, rows=1)
fig.add_traces(list(figure_etanol.values()), cols=2, rows=1)

# Atualize o layout com as legendas
fig.update_layout(layout)

# Salve o arquivo HTML com um nome descritivo
fig.write_html('figure4.html', auto_open=True)

# 5. Qual o município que possui o menor preço para a gasolina e para o etanol?

# Print the information for the municipality with the lowest Gasolina (Gasoline) price in November
print('Gasolina November\n', df_nov[df_nov['Produto']==product[0]].nsmallest(1, 'Valor de Venda')[['Estado - Sigla', 'Municipio', 'Valor de Venda']].iloc[0], '\n')

# Print the information for the municipality with the lowest Gasolina (Gasoline) price in December
print('Gasolina December\n', df_dez[df_dez['Produto']==product[0]].nsmallest(1, 'Valor de Venda')[['Estado - Sigla', 'Municipio', 'Valor de Venda']].iloc[0], '\n')

# Print the information for the municipality with the lowest Etanol (Ethanol) price in November
print('Etanol November\n', df_nov[df_nov['Produto']==product[2]].nsmallest(1, 'Valor de Venda')[['Estado - Sigla', 'Municipio', 'Valor de Venda']].iloc[0], '\n')

# Print the information for the municipality with the lowest Etanol (Ethanol) price in December
print('Etanol December\n', df_dez[df_dez['Produto']==product[2]].nsmallest(1, 'Valor de Venda')[['Estado - Sigla', 'Municipio', 'Valor de Venda']].iloc[0])

# 6. Qual o município que possui o maior preço para a gasolina e para o etanol?

# Print the information for the municipality with the highest Gasolina (Gasoline) price in November
print('Gasolina November\n', df_nov[df_nov['Produto']==product[0]].nlargest(1, 'Valor de Venda')[['Estado - Sigla', 'Municipio', 'Valor de Venda']].iloc[0], '\n')

# Print the information for the municipality with the highest Gasolina (Gasoline) price in December
print('Gasolina December\n', df_dez[df_dez['Produto']==product[0]].nlargest(1, 'Valor de Venda')[['Estado - Sigla', 'Municipio', 'Valor de Venda']].iloc[0], '\n')

# Print the information for the municipality with the highest Etanol (Ethanol) price in November
print('Etanol November\n', df_nov[df_nov['Produto']==product[2]].nlargest(1, 'Valor de Venda')[['Estado - Sigla', 'Municipio', 'Valor de Venda']].iloc[0], '\n')

# Print the information for the municipality with the highest Etanol (Ethanol) price in December
print('Etanol December\n', df_dez[df_dez['Produto']==product[2]].nlargest(1, 'Valor de Venda')[['Estado - Sigla', 'Municipio', 'Valor de Venda']].iloc[0])

# 7. Qual a região que possui o maior valor médio da gasolina?

# Group by region and product, calculate mean price for Gasolina (Gasoline) in November
data_regiao = df_nov.groupby(['Regiao - Sigla', 'Produto'])['Valor de Venda'].mean().reset_index()
print('Gasolina November\n', data_regiao[data_regiao['Produto']==product[0]].nlargest(1,'Valor de Venda')[['Regiao - Sigla', 'Valor de Venda']].iloc[0], '\n')

# Group by region and product, calculate mean price for Gasolina (Gasoline) in December
data_regiao = df_dez.groupby(['Regiao - Sigla', 'Produto'])['Valor de Venda'].mean().reset_index()
print('Gasolina December\n', data_regiao[data_regiao['Produto']==product[0]].nlargest(1,'Valor de Venda')[['Regiao - Sigla', 'Valor de Venda']].iloc[0])

# 8. Qual a região que possui o menor valor médio do etanol?

# Group by region and product, calculate mean price for Etanol (Ethanol) in November
data_regiao = df_nov.groupby(['Regiao - Sigla', 'Produto'])['Valor de Venda'].mean().reset_index()
print('Etanol November\n', data_regiao[data_regiao['Produto']==product[2]].nsmallest(1,'Valor de Venda')[['Regiao - Sigla', 'Valor de Venda']].iloc[0], '\n')

# Group by region and product, calculate mean price for Etanol (Ethanol) in December
data_regiao = df_dez.groupby(['Regiao - Sigla', 'Produto'])['Valor de Venda'].mean().reset_index()
print('Etanol December\n', data_regiao[data_regiao['Produto']==product[2]].nsmallest(1,'Valor de Venda')[['Regiao - Sigla', 'Valor de Venda']].iloc[0])

# # 9. Há alguma correlação entre o valor do combustível (gasolina e etanol) e a região onde ele é vendido?


# # 10. Há alguma correlação entre o valor do combustível (gasolina e etanol) e a bandeira que vende ele?