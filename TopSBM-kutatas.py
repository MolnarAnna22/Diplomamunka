#!/usr/bin/env python
# coding: utf-8

# # TopSBM: Topic Modeling with Stochastic Block Models

import os
import pylab as plt
import graph_tool.all as gt
from sbmtm import sbmtm
import pandas as pd

# Adatok betöltése
final = pd.read_csv("~/Documents/anna/merged_df.csv", sep = ",", encoding = "utf8")

# Szövegek listába mentése
texts = list(final.clear_text)
txt = [h.split() for h in texts]

## titles
titles = range(0, len(final.clear_text))

model = sbmtm()

# Szó-dokumentum hálózat készítése
model.make_graph(txt,documents=titles)

# Modell mentése
model.save_graph(filename = 'graph.xml.gz')
model.load_graph(filename = 'graph.xml.gz')

# Fit the model
gt.seed_rng(32)
model.fit()

# Modell kirajzoltatása
model.plot(nedges=10000, filename="abra_uj.png")


# Legjellemzőbb 20 szó kiíratása topikonként
print(model.topics(l=1,n=20))
print(model.topics(l=2,n=20))
print(model.topics(l=3,n=20))

# Topikok és kulcsszavak + dokumentumokhoz topikeloszlások kiszámítása
print(model.print_topics(l=1,format='csv',path_save = ''))
print(model.print_topics(l=2,format='csv',path_save = ''))
print(model.print_topics(l=3,format='csv',path_save = ''))



