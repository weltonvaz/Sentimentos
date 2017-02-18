#!/usr/bin/python3
# -*- coding: utf-8 -*-
# limpando a tela antes de começar!
import os
os.system('clear')

#Conjunto de importações
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

#Armazenamento de documentos (frases) na lista dataset
dataset = [("Eu amo este carro."),
           ("Este visual é lindo."),
           ("Eu gosto muito da noite."),
           ("Estou muito ansioso para o show."),
           ("Ela é minha grande amiga."),
           ("Eu não gosto deste carro."),
           ("Este visual é horrível."),
           ("Eu me sinto cansada nesta tarde."),
           ("Não estou ansioso para a viagem."),
           ("Ele é muito organizado."),
           ("Eu me sinto feliz hoje."),
           ("Ela é muito inteligente e muito dedicada."),
           ("Eu não confio naquele homem."),
           ("Minha casa está suja."),
           ("Seu vizinho é chato.")]

#Armazenamento das polaridades de cada documento (frase) na lista polaris
polaris = [1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1]

#Divisão dos dados das listas dataset e polaris em conjuntos de treinamento e validação
dados_treino, dados_val, pols_treino, pols_val = train_test_split(dataset, polaris, test_size=0.30)

#Print do conjunto de treinamento e suas respectivas polaridades
print("Conjunto de Treinamento")
print(dados_treino)
print("Polaridades do Conjunto de Treinamento")
print(pols_treino)
print("\n---------------------------------------------\n")
#Print do conjunto de validação e suas respectivas polaridades
print("Conjunto de Validação")
print(dados_val)
print("Polaridades do Conjunto de Validação")
print(pols_val)

#Cria uma instância para a bag-of-words
bag = CountVectorizer()

#Método fit_transform:
#fit = cria e aprende a bag
#transform = cria a matriz termo-documento
bag_treino = bag.fit_transform(dados_treino)

#Printa o vocabulário da bag-of-words
print(bag.vocabulary_)

#A função sorted() ordena o vocabulário da bag-of-words
print(sorted(bag.vocabulary_))

#Printa a bag-of-words
print(bag_treino)

#Cria a matriz termo-documento para o conjunto de validação com a bag já treinada
bag_val = bag.transform(dados_val)

#Printa a matriz termo-documento criada para o conjunto de validação
print(bag_val)
