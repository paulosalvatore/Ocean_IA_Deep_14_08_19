#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Importando dependências

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop


# In[8]:


# Carregando os dados de treino e teste

(x_treino, y_treino), (x_teste, y_teste) = mnist.load_data()


# In[15]:


# Exibindo como estão os dados

# Quantas imagens para treino?
print("Imagens de Treino:", len(x_treino))

# Quantas imagens para teste?
print("Imagens de Teste:", len(x_teste))

# Qual o formato de uma imagem?
print("Formato da imagem:", x_treino[0].shape)

# O que a imagem x_treino[0] representa?
print("Representação da imagem x_treino[0]:", y_treino[0])

# Como são os dados de uma imagem?
print("Dados da imagem:", x_treino[0])


# In[25]:


# Exibindo a imagem

import matplotlib.pyplot as plt

# Configuração para o jupyter notebook exibir a imagem corretamente
get_ipython().run_line_magic('matplotlib', 'inline')

indice = 0

# Se quiser perguntar para o usuário um número de índice:
#indice = int(input("Digite um número válido entre 0 e 59999: "))

print("Essa imagem representa:", y_treino[indice])
plt.imshow(x_treino[indice], cmap=plt.cm.binary)
plt.show()


# In[27]:


# Achatando as matrizes de pixels e transformando em uma única lista com valores

quantidade_treino = len(x_treino) # Vai me trazer 60000
quantidade_teste = len(x_teste) # Vai me trazer 10000

tamanho_imagem = x_treino[0].shape # Vai me trazer (28, 28)
tamanho_total = tamanho_imagem[0] * tamanho_imagem[1] # Vai me trazer 784

x_treino = x_treino.reshape(quantidade_treino, tamanho_total)
x_teste = x_teste.reshape(quantidade_teste, tamanho_total)


# In[30]:


# Visualizar dados achatados

print("Novo formato dos dados:", x_treino[0].shape)
print(x_treino[0])


# In[34]:


# Normalização dos dados

# Converte todos os valores de int8 para float32
x_treino = x_treino.astype('float32')
x_teste = x_teste.astype('float32')

# Valores entre 0 e 255 ficarão entre 0 e 1
x_treino /= 255
x_teste /= 255


# In[35]:


# Visualizando os dados normalizados

print("Dados normalizados")
print(x_treino[0])


# In[42]:


# Transformando y_treino e y_teste para variáveis categóricas

valores_unicos = set(y_treino) # Irá me trazer os itens únicos: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
qtde_valores_unicos = len(valores_unicos) # Irá me trazer que são 10 itens únicos
print("Quantidade de Valores Únicos em y_treino:", qtde_valores_unicos)

# O que temos em y_treino[0]?
print("y_treino[0] antes:", y_treino[0])

# Transforma 1 em [0, 1, 0, 0, 0, 0, 0, 0, 0], 2 em [0, 0, 1, 0, 0, 0, 0, 0, 0] e assim por diante
y_treino = keras.utils.to_categorical(y_treino, qtde_valores_unicos)
y_teste = keras.utils.to_categorical(y_teste, qtde_valores_unicos)

# Como ficou y_treino[0] depois da transformação?
print("y_treino[0] depois:", y_treino[0])


# In[45]:


# Criando o modelo

model = Sequential()

# Primeira hidden layer com 30 neurônios, com função de ativação ReLU
# Na primeira camada, precisamos definir o input shape, que no caso será (784,)
model.add(Dense(30, activation='relu', input_shape=(tamanho_total,)))

# Adicionamos um regularizador. No caso, será um Dropout
model.add(Dropout(0.2))

# Segunda hidden layer com 20 neurônios, com função de ativação ReLU
model.add(Dense(20, activation='relu'))

# Mais um regularizador depois da segunda hidden layer
model.add(Dropout(0.2))

# Finalizamos com a camada de output, com a quantidade de valores únicos (no caso 10) e uma
# função de ativação Softmax
model.add(Dense(qtde_valores_unicos, activation='softmax'))

# Exibimos o resumo do modelo criado
model.summary()


# In[46]:


# Compila o modelo criado

model.compile(loss='categorical_crossentropy',
             optimizer=RMSprop(),
             metrics=['accuracy'])


# In[47]:


# Treina o modelo

history = model.fit(x_treino, y_treino,
                   batch_size=128,
                   epochs=10,
                   verbose=1,
                   validation_data=(x_teste, y_teste))


# In[52]:


# Fazendo nossas previsões

indice = 9

# Qual o valor categórico de y_teste[indice]?
print("Valor em y_teste[indice]", y_teste[indice])
# y_teste[indice] irá me trazer [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.], por tanto, deve ser um 7

# Reajustando a imagem em x_teste[indice]
imagem = x_teste[indice].reshape((1, tamanho_total))

# Fazendo minha previsão
prediction = model.predict(imagem) # Irá retornar os valores de cada posição do output
print("Previsão:", prediction)

# Ajustando a previsão para o número real
prediction_class = model.predict_classes(imagem)
print("Previsão (ajustada):", prediction_class)

(x_treino_img, y_treino_img), (x_teste_img, y_teste_img) = mnist.load_data()
plt.imshow(x_teste_img[indice], cmap=plt.cm.binary)

