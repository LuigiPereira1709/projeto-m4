# Import necessary libraries
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from dtype_diet import report_on_dataframe, optimize_dtypes

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

# Group by product and day of collection, calculating the mean price for November
df_nov1 = df_nov.groupby(['Produto','Dia da Coleta'])['Valor de Venda'].mean().reset_index()

# Group by product and day of collection, calculating the mean price for December
df_dez1 = df_dez.groupby(['Produto', 'Dia da Coleta'])['Valor de Venda'].mean().reset_index()

# Merge the November and December dataframes on the day of collection
data = pd.merge(df_dez1, df_nov1, on='Dia da Coleta', how='inner').reset_index()

# Calculate the overall mean prices for all fuels in November and December
data_all_nov = df_nov.groupby(['Dia da Coleta'])['Valor de Venda'].mean()
data_all_dez = df_dez.groupby(['Dia da Coleta'])['Valor de Venda'].mean()

# Merge the overall mean prices for November and December on the day of collection
data_all = pd.merge(data_all_nov, data_all_dez, on='Dia da Coleta', how='inner')

# Create a 2x2 subplot grid
fig, axes = plt.subplots(2,2, figsize=(10,6))

# Plot line chart for overall mean prices of all fuels in November and December
sns.lineplot(x='Dia da Coleta', y='Valor de Venda_y', data=data_all, label='December', color='orange', ax=axes[0,0])
sns.lineplot(x='Dia da Coleta', y='Valor de Venda_x', data=data_all, label='November', color='blue', ax=axes[0,0])
axes[0,0].set_title('All Fuels')
axes[0,0].set_xlabel('Days')
axes[0,0].set_ylabel('Average Price')
axes[0,0].legend()

# Plot line chart for Gasolina Comum (Regular Gasoline) prices in November and December
sns.lineplot(x='Dia da Coleta', y='Valor de Venda_x', data=data[data['Produto_x'] == product[0]], label='December', color='blue', marker='o', ax=axes[0,1])
sns.lineplot(x='Dia da Coleta', y='Valor de Venda_y', data=data[data['Produto_y'] == product[0]], label='November', color='purple', marker='s', ax=axes[0,1])
axes[0,1].set_title('Regular Gasoline')
axes[0,1].set_xlabel('Days')
axes[0,1].set_ylabel('Average Price')
axes[0,1].legend()

# Plot line chart for Gasolina Aditivada (Premium Gasoline) prices in November and December
sns.lineplot(x='Dia da Coleta', y='Valor de Venda_x', data=data[data['Produto_x'] == product[1]], label='December', color='red', marker='s', ax=axes[1,0])
sns.lineplot(x='Dia da Coleta', y='Valor de Venda_y', data=data[data['Produto_y'] == product[1]], label='November', color='green', marker='o', ax=axes[1,0])
axes[1,0].set_title('Premium Gasoline')
axes[1,0].set_xlabel('Days')
axes[1,0].set_ylabel('Average Price')
axes[1,0].legend()

# Plot line chart for Etanol (Ethanol) prices in November and December
sns.lineplot(x='Dia da Coleta', y='Valor de Venda_x', data=data[data['Produto_x'] == product[2]], label='December', color='yellow', ax=axes[1,1])
sns.lineplot(x='Dia da Coleta', y='Valor de Venda_y', data=data[data['Produto_y'] == product[2]], label='November', color='pink', ax=axes[1,1])
axes[1,1].set_title('Ethanol')
axes[1,1].set_xlabel('Days')
axes[1,1].set_ylabel('Average Price')
axes[1,1].legend()

# Set the overall title for the entire plot
fig.suptitle('Average Price Comparison Between November and December')
plt.tight_layout()
plt.show()

# 2. Qual o preço médio da gasolina e do etanol nesses dois meses?

# Create a DataFrame for November mean values and counts for each product
data_nov = pd.DataFrame({"Product": product, "Mean_Value": '', "Month": "November"})

# Calculate mean values for each product in November
data_nov['Mean_Value'] = [operation_df(df_nov, 'Produto', 'mean', p, 'Valor de Venda') for p in product]

# Calculate overall mean value for all fuels in November
data_nov['Mean_All_Fuels'] = operation_df(df_nov, 'Valor de Venda', 'mean')

# Create a DataFrame for December mean values and counts for each product
data_dez = pd.DataFrame({"Product": product, "Mean_Value": '', "Month": "December"})

# Calculate mean values for each product in December
data_dez['Mean_Value'] = [operation_df(df_dez, 'Produto', 'mean', p, 'Valor de Venda') for p in product]

# Calculate counts for each product in November and December
data_nov['Qtd'] = [operation_df(df_nov, 'Produto', 'count', p, 'Valor de Venda') for p in product]
data_dez['Qtd'] = [operation_df(df_dez, 'Produto', 'count', p, 'Valor de Venda') for p in product]

# Concatenate November and December data
data = pd.concat([data_dez, data_nov])

# Merge November and December data for further analysis
data_merged = pd.merge(data_nov.drop(columns=['Month']), data_dez.drop(columns=['Month']), on='Product', how='inner')

# Create a subplot with two plots side by side
fig, axes = plt.subplots(1,2, figsize=(10, 6))

# Plot a bar chart showing mean values for each product in November and December
sns.barplot(x='Product', y='Mean_Value', hue="Month", data=data, color='purple', ax=axes[0])
axes[0].set_title("Mean of Products Between November and December")
axes[0].set_ylabel("Mean_Value")
axes[0].set_xlabel("Products")
axes[0].legend()

# Plot a pie chart showing the distribution of fuel types based on counts in November and December
axes[1].pie(data_merged[['Qtd_x', 'Qtd_y']].sum(axis=1), colors=sns.color_palette('bright'), autopct="%.0f%%")
axes[1].set_title('Distribution of fuel types')
axes[1].legend(labels=data_merged['Product'], loc='lower center')

# Adjust layout for better visualization
plt.tight_layout()
plt.show()

# 3. Quais os 5 estados com o preço médio da gasolina e do etanol mais caros?

# Concatenate dataframes for November and December
df = pd.concat([df_nov, df_dez], axis=0, ignore_index=True)

# Ensure 'Estado - Sigla' column is treated as a string
df['Estado - Sigla'] = df['Estado - Sigla'].astype(str) 

# Group by state and product, calculate mean price, and sort values in descending order
df_per_product = df.groupby(['Estado - Sigla', 'Produto'])['Valor de Venda'].mean().reset_index()
df_per_product = df_per_product.sort_values('Valor de Venda', ascending=False)

# Group by state, calculate mean price, and select the top 5 states with the highest average prices
df_all = df.groupby('Estado - Sigla')['Valor de Venda'].mean().reset_index()
df_all = df_all.sort_values('Valor de Venda', ascending=False).head(5).reset_index(drop=True)

# Create a subplot with three plots side by side
fig, axes = plt.subplots(1,3, figsize=(10,6))

# Plot bar chart for the top 5 states with the highest average prices for all fuels
sns.barplot(x='Estado - Sigla', y='Valor de Venda', data=df_all, palette='bright', ax=axes[1])
axes[1].set_title('All Fuels')
axes[1].set_xlabel('States')
axes[1].set_ylabel('Average Price')

# Plot bar chart for the top 5 states with the highest average prices for Gasolina (Gasoline)
sns.barplot(x='Estado - Sigla', y='Valor de Venda', data=df_per_product[df_per_product['Produto']==product[0]].head(5), palette='bright', ax=axes[0])
axes[0].set_title('Gasolina')
axes[0].set_xlabel('States')
axes[0].set_ylabel('Average Price')

# Plot bar chart for the top 5 states with the highest average prices for Etanol (Ethanol)
sns.barplot(x='Estado - Sigla', y='Valor de Venda', data=df_per_product[df_per_product['Produto']==product[2]].head(5), palette='bright', ax=axes[2])
axes[2].set_title('Ethanol')
axes[2].set_xlabel('States')
axes[2].set_ylabel('Average Price')

# Set the overall title for the entire plot
plt.suptitle('Top 5 States with the Highest Average Prices')
plt.tight_layout()
plt.show()

# # 4. Qual o preço médio da gasolina e do etanol por estado?
# df_data = df.groupby(['Estado - Sigla', 'Produto'])['Valor de Venda'].mean().reset_index()

# fig, axes = plt.subplots(1,2, figsize=(10,6))

# sns.scatterplot(y='Estado - Sigla', x='Valor de Venda', data=df_data[df_data['Produto']==product[0]], ax=axes[0])
# axes[0].set_title('Gasolina')
# axes[0].set_xlabel('Preco medio')
# axes[0].set_ylabel('Estados')

# sns.scatterplot(y='Estado - Sigla', x='Valor de Venda', data=df_data[df_data['Produto']==product[2]], color='orange', ax=axes[1])
# axes[1].set_title('Etanol')
# axes[1].set_xlabel('Preco medio')
# axes[1].set_ylabel('=='*20)
# axes[1].set_yticks([])

# plt.suptitle('Preco Medio por Estado')
# plt.tight_layout()
# plt.show()


# # 5. Qual o município que possui o menor preço para a gasolina e para o etanol?
# print('Gasolina Novembro\n', df_nov[df_nov['Produto']==product[0]].nsmallest(1, 'Valor de Venda')[['Estado - Sigla', 'Municipio', 'Valor de Venda']].iloc[0], '\n')
# print('Gasolina Novembro\n', df_nov[df_nov['Produto']==product[0]].nsmallest(1, 'Valor de Venda')[['Estado - Sigla', 'Municipio', 'Valor de Venda']].iloc[0], '\n')
# print('Etanol Novembro\n', df_nov[df_nov['Produto']==product[2]].nsmallest(1, 'Valor de Venda')[['Estado - Sigla', 'Municipio', 'Valor de Venda']].iloc[0], '\n')
# print('Etanol Dezembro\n', df_dez[df_dez['Produto']==product[2]].nsmallest(1, 'Valor de Venda')[['Estado - Sigla', 'Municipio', 'Valor de Venda']].iloc[0])

# # 6. Qual o município que possui o maior preço para a gasolina e para o etanol?
# print('Gasolina Novembro\n', df_nov[df_nov['Produto']==product[0]].nlargest(1, 'Valor de Venda')[['Estado - Sigla', 'Municipio', 'Valor de Venda']].iloc[0], '\n')
# print('Gasolina Dezembro\n', df_dez[df_dez['Produto']==product[0]].nlargest(1, 'Valor de Venda')[['Estado - Sigla', 'Municipio', 'Valor de Venda']].iloc[0], '\n')
# print('Etanol Novembro\n', df_nov[df_nov['Produto']==product[2]].nlargest(1, 'Valor de Venda')[['Estado - Sigla', 'Municipio', 'Valor de Venda']].iloc[0], '\n')
# print('Etanol Dezembro\n', df_dez[df_dez['Produto']==product[2]].nlargest(1, 'Valor de Venda')[['Estado - Sigla', 'Municipio', 'Valor de Venda']].iloc[0])

# # 7. Qual a região que possui o maior valor médio da gasolina?
# data_regiao = df_nov.groupby(['Regiao - Sigla', 'Produto'])['Valor de Venda'].mean().reset_index()
# print('Gasolina Novembro\n', data_regiao[data_regiao['Produto']==product[0]].nlargest(1,'Valor de Venda')[['Regiao - Sigla', 'Valor de Venda']].iloc[0], '\n')
# data_regiao = df_dez.groupby(['Regiao - Sigla', 'Produto'])['Valor de Venda'].mean().reset_index()
# print('Gasolina Dezembro\n', data_regiao[data_regiao['Produto']==product[0]].nlargest(1,'Valor de Venda')[['Regiao - Sigla', 'Valor de Venda']].iloc[0])

# # 8. Qual a região que possui o menor valor médio do etanol?
# data_regiao = df_nov.groupby(['Regiao - Sigla', 'Produto'])['Valor de Venda'].mean().reset_index()
# print('Etanol Novembro\n', data_regiao[data_regiao['Produto']==product[2]].nsmallest(1,'Valor de Venda')[['Regiao - Sigla', 'Valor de Venda']].iloc[0], '\n')
# data_regiao = df_dez.groupby(['Regiao - Sigla', 'Produto'])['Valor de Venda'].mean().reset_index()
# print('Etanol Dezembro\n', data_regiao[data_regiao['Produto']==product[2]].nsmallest(1,'Valor de Venda')[['Regiao - Sigla', 'Valor de Venda']].iloc[0])


# # 9. Há alguma correlação entre o valor do combustível (gasolina e etanol) e a região onde ele é vendido?


# # 10. Há alguma correlação entre o valor do combustível (gasolina e etanol) e a bandeira que vende ele?
