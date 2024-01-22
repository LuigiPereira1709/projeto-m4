import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from dtype_diet import report_on_dataframe, optimize_dtypes

df_nov = pd.read_csv('./data/precos-gasolina-etanol-11.csv', delimiter=';')
df_nov['Valor de Venda'] = df_nov['Valor de Venda'].str.replace(',', '.').astype('float') 
df_nov['Data da Coleta'] = pd.to_datetime(df_nov['Data da Coleta'], format="%d/%m/%Y")
df_nov['Dia da Coleta'] = df_nov['Data da Coleta'].dt.day
proposed_df = report_on_dataframe(df_nov, unit="MB")
df_nov= optimize_dtypes(df_nov, proposed_df)

df_dez = pd.read_csv('./data/precos-gasolina-etanol-12.csv', delimiter=';')
df_dez['Valor de Venda'] = df_dez['Valor de Venda'].str.replace(',', '.').astype('float') 
df_dez['Data da Coleta'] = pd.to_datetime(df_dez['Data da Coleta'], format="%d/%m/%Y")
df_dez['Dia da Coleta'] = df_dez['Data da Coleta'].dt.day
proposed_df = report_on_dataframe(df_dez, unit="MB")
df_dez= optimize_dtypes(df_dez, proposed_df)

def operation_df(df, column1, operation, product=None, column2=None):
    """
    Filter DataFrame and apply aggregation.

    Parameters: df, column1, product, column2, operation

    Returns: Aggregation result.
    """
    if (product and column2) is not None:
        result = df[df[column1] == product][column2].agg(operation)
        return result
    else:
        result = df[column1].agg(operation)
        return result

product = ['GASOLINA', 'GASOLINA ADITIVADA', 'ETANOL']

# 1. Como se comportaram o preço dos combustíveis durante os dois meses citados? Os valores do 
# etanol e da gasolina tiveram uma tendência de queda ou diminuição?

# df_nov1 = df_nov.groupby(['Produto','Dia da Coleta'])['Valor de Venda'].mean().reset_index()

# df_dez1 = df_dez.groupby(['Produto', 'Dia da Coleta'])['Valor de Venda'].mean().reset_index()

# data = pd.merge(df_dez1, df_nov1, on='Dia da Coleta', how='inner').reset_index()

# data_all_nov = df_nov.groupby(['Dia da Coleta'])['Valor de Venda'].mean()
# data_all_dez = df_dez.groupby(['Dia da Coleta'])['Valor de Venda'].mean()

# data_all = pd.merge(data_all_nov, data_all_dez, on='Dia da Coleta', how='inner')


# fig, axes = plt.subplots(2,2, figsize=(10,6))

# sns.lineplot(x='Dia da Coleta', y='Valor de Venda_y', data=data_all, label='Dezembro', color='orange', ax=axes[0,0])
# sns.lineplot(x='Dia da Coleta', y='Valor de Venda_x', data=data_all, label='Novembro', color='blue', ax=axes[0,0])
# axes[0,0].set_title('Todos combustiveis')
# axes[0,0].set_xlabel('Dias')
# axes[0,0].set_ylabel('Preco Medio')
# axes[0,0].legend()


# sns.lineplot(x='Dia da Coleta', y='Valor de Venda_x', data=data[data['Produto_x'] == product[0]], label='Dezembro', color='blue', marker='o', ax=axes[0,1])
# sns.lineplot(x='Dia da Coleta', y='Valor de Venda_y', data=data[data['Produto_y'] == product[0]], label='Novembro', color='purple', marker='s', ax=axes[0,1])
# axes[0,1].set_title('Gasolina Comum')
# axes[0,1].set_xlabel('Dias')
# axes[0,1].set_ylabel('Preco medio')
# axes[0,1].legend()

# sns.lineplot(x='Dia da Coleta', y='Valor de Venda_x', data=data[data['Produto_x'] == product[1]], label='Dezembro', color='red', marker='s', ax=axes[1,0])
# sns.lineplot(x='Dia da Coleta', y='Valor de Venda_y', data=data[data['Produto_y'] == product[1]], label='Novembro', color='green', marker='o', ax=axes[1,0])
# axes[1,0].set_title('Gasolina Aditivada')
# axes[1,0].set_xlabel('Dias')
# axes[1,0].set_ylabel('Preco medio')
# axes[1,0].legend()

# sns.lineplot(x='Dia da Coleta', y='Valor de Venda_x', data=data[data['Produto_x'] == product[2]], label='Dezembro', color='yellow', ax=axes[1,1])
# sns.lineplot(x='Dia da Coleta', y='Valor de Venda_y', data=data[data['Produto_y'] == product[2]], label='Novembro', color='pink', ax=axes[1,1])
# axes[1,1].set_title('Etanol')
# axes[1,1].set_xlabel('Dias')
# axes[1,1].set_ylabel('Preco Medio')
# axes[1,1].legend()

# fig.suptitle('Media do preco entre novembro e dezembro')
# plt.tight_layout()
# plt.show()

# 2. Qual o preço médio da gasolina e do etanol nesses dois meses?

# data_nov = pd.DataFrame({"Product": product, "Mean_Value": '', "Month": "November"})
# data_nov['Mean_Value'] = [operation_df(df_nov, 'Produto', 'mean', p, 'Valor de Venda') for p in product]

# data_nov['Mean_All_Fuels'] = operation_df(df_nov, 'Valor de Venda', 'mean')

# data_dez = pd.DataFrame({"Product": product, "Mean_Value": '', "Month": "December"})
# data_dez['Mean_Value'] = [operation_df(df_dez, 'Produto', 'mean', p, 'Valor de Venda') for p in product]

# data_nov['Qtd'] = [operation_df(df_nov, 'Produto', 'count', p, 'Valor de Venda') for p in product]
# data_dez['Qtd'] = [operation_df(df_dez, 'Produto', 'count', p, 'Valor de Venda') for p in product]

# data = pd.concat([data_dez, data_nov])

# data_merged = pd.merge(data_nov.drop(columns=['Month']), data_dez.drop(columns=['Month']), on='Product', how='inner')

# fig, axes = plt.subplots(1,2, figsize=(10, 6))
# sns.barplot(x='Product', y='Mean_Value', hue="Month", data=data, color='purple', ax=axes[0])
# axes[0].set_title("Mean of Products Between November and December")
# axes[0].set_ylabel("Mean_Value")
# axes[0].set_xlabel("Products")
# axes[0].legend()

# axes[1].pie(data_merged[['Qtd_x', 'Qtd_y']].sum(axis=1), colors=sns.color_palette('bright'), autopct="%.0f%%")
# axes[1].set_title('Distribution of fuel types')
# axes[1].legend(labels=data_merged['Product'], loc='lower center')

# plt.tight_layout()
# plt.show()

# 3. Quais os 5 estados com o preço médio da gasolina e do etanol mais caros?
df = pd.concat([df_nov, df_dez], axis=0, ignore_index=True)
df['Estado - Sigla'] = df['Estado - Sigla'].astype(str) 

# df_per_product = df.groupby(['Estado - Sigla', 'Produto'])['Valor de Venda'].mean().reset_index()
# df_per_product = df_per_product.sort_values('Valor de Venda', ascending=False)

# df_all = df.groupby('Estado - Sigla')['Valor de Venda'].mean().reset_index()
# df_all = df_all.sort_values('Valor de Venda', ascending=False).head(5).reset_index(drop=True)

# fig, axes = plt.subplots(1,3, figsize=(10,6))

# sns.barplot(x='Estado - Sigla', y='Valor de Venda', data=df_all, palette='bright', ax=axes[1])
# axes[1].set_title('Todos os combustiveis')
# axes[1].set_xlabel('Estados')
# axes[1].set_ylabel('Preco Medio')

# sns.barplot(x='Estado - Sigla', y='Valor de Venda', data=df_per_product[df_per_product['Produto']==product[0]].head(5), palette='bright', ax=axes[0])
# axes[0].set_title('Gasolina')
# axes[0].set_xlabel('Estados')
# axes[0].set_ylabel('Preco medio')

# sns.barplot(x='Estado - Sigla', y='Valor de Venda', data=df_per_product[df_per_product['Produto']==product[2]].head(5), palette='bright', ax=axes[2])
# axes[2].set_title('Etanol')
# axes[2].set_xlabel('Estados')
# axes[2].set_ylabel('Preco medio')

# plt.suptitle('Top 5 Estados Com Preco Medio Mais Alto')
# plt.tight_layout()
# plt.show()


# 4. Qual o preço médio da gasolina e do etanol por estado?
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


# 5. Qual o município que possui o menor preço para a gasolina e para o etanol?
# print('Gasolina Novembro\n', df_nov[df_nov['Produto']==product[0]].nsmallest(1, 'Valor de Venda')[['Estado - Sigla', 'Municipio', 'Valor de Venda']].iloc[0], '\n')
# print('Gasolina Novembro\n', df_nov[df_nov['Produto']==product[0]].nsmallest(1, 'Valor de Venda')[['Estado - Sigla', 'Municipio', 'Valor de Venda']].iloc[0], '\n')
# print('Etanol Novembro\n', df_nov[df_nov['Produto']==product[2]].nsmallest(1, 'Valor de Venda')[['Estado - Sigla', 'Municipio', 'Valor de Venda']].iloc[0], '\n')
# print('Etanol Dezembro\n', df_dez[df_dez['Produto']==product[2]].nsmallest(1, 'Valor de Venda')[['Estado - Sigla', 'Municipio', 'Valor de Venda']].iloc[0])

# 6. Qual o município que possui o maior preço para a gasolina e para o etanol?
# print('Gasolina Novembro\n', df_nov[df_nov['Produto']==product[0]].nlargest(1, 'Valor de Venda')[['Estado - Sigla', 'Municipio', 'Valor de Venda']].iloc[0], '\n')
# print('Gasolina Dezembro\n', df_dez[df_dez['Produto']==product[0]].nlargest(1, 'Valor de Venda')[['Estado - Sigla', 'Municipio', 'Valor de Venda']].iloc[0], '\n')
# print('Etanol Novembro\n', df_nov[df_nov['Produto']==product[2]].nlargest(1, 'Valor de Venda')[['Estado - Sigla', 'Municipio', 'Valor de Venda']].iloc[0], '\n')
# print('Etanol Dezembro\n', df_dez[df_dez['Produto']==product[2]].nlargest(1, 'Valor de Venda')[['Estado - Sigla', 'Municipio', 'Valor de Venda']].iloc[0])

# 7. Qual a região que possui o maior valor médio da gasolina?
# data_regiao = df_nov.groupby(['Regiao - Sigla', 'Produto'])['Valor de Venda'].mean().reset_index()
# print('Gasolina Novembro\n', data_regiao[data_regiao['Produto']==product[0]].nlargest(1,'Valor de Venda')[['Regiao - Sigla', 'Valor de Venda']].iloc[0], '\n')
# data_regiao = df_dez.groupby(['Regiao - Sigla', 'Produto'])['Valor de Venda'].mean().reset_index()
# print('Gasolina Dezembro\n', data_regiao[data_regiao['Produto']==product[0]].nlargest(1,'Valor de Venda')[['Regiao - Sigla', 'Valor de Venda']].iloc[0])

# 8. Qual a região que possui o menor valor médio do etanol?
data_regiao = df_nov.groupby(['Regiao - Sigla', 'Produto'])['Valor de Venda'].mean().reset_index()
print('Etanol Novembro\n', data_regiao[data_regiao['Produto']==product[2]].nsmallest(1,'Valor de Venda')[['Regiao - Sigla', 'Valor de Venda']].iloc[0], '\n')
data_regiao = df_dez.groupby(['Regiao - Sigla', 'Produto'])['Valor de Venda'].mean().reset_index()
print('Etanol Dezembro\n', data_regiao[data_regiao['Produto']==product[2]].nsmallest(1,'Valor de Venda')[['Regiao - Sigla', 'Valor de Venda']].iloc[0])


# 9. Há alguma correlação entre o valor do combustível (gasolina e etanol) e a região onde ele é vendido?


# 10. Há alguma correlação entre o valor do combustível (gasolina e etanol) e a bandeira que vende ele?
