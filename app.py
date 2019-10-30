from flask import Flask, render_template
app = Flask(__name__)

###
import requests 
import csv

from pandas import DataFrame
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg 

import os
import re 
import numpy as np
from sklearn import linear_model
from scipy.stats import zscore
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import seaborn as sns
#%matplotlib inline
from mlxtend.frequent_patterns import apriori, association_rules
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:90% !important; }</style>"))
import collections
from sklearn.metrics import f1_score
from sklearn import tree







from collections import Counter
#from colour import Color
import json
from operator import itemgetter
import pandas as pd
from scipy import stats
import six
from statsmodels.graphics.mosaicplot import mosaic
import sys
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import StratifiedKFold, ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer
#from lime.lime_tabular import LimeTabularExplainer
#import tensorflow as tf
####

@app.route('/')
@app.route('/dashboard')
def homepage():
    return render_template('index.html')

@app.route('/analyze_data')
def rundata_click():
    return render_template('analyze_data.html')

@app.route('/gender')
def gender_barplot():
	df = pd.read_csv('prescription_data.csv', sep=',', low_memory=False)
	series = df['gender']
	fig_size=(10,5)
	fig = plt.figure(figsize=(10,5))
	ax = plt.subplot()
	counts = series.value_counts()
	vals = counts.values
	pers = vals / vals.sum()
	ax = plt.subplot()
	counts.plot(title=series.name, kind='barh', figsize=fig_size, ax=ax, color = 'slateblue')
	plt.style.use('seaborn-whitegrid')
	nudged_vals = vals * 1.01  
	for i, val in enumerate(nudged_vals):
		formatted_per = '{:0.01%}'.format(pers[i])
		ax.text(nudged_vals[i], i, formatted_per)
	ax.set_xlim((0, max(counts.values)*1.2))
	fig.savefig('static/gender.png')
	return render_template('gender.html')

@app.route('/gender_vs_specialty')
def gender_specialty_graph():

	def target_mosaic(cat1, cat2, figsize=(18, 4)):
		xtab = pd.crosstab(cat1, cat2).unstack()
		# Bas colors:
		colors = ['#0499CC', '#4D8951', '#FDBA58', '#876DB5', '#32A8B4', '#9BB8D7', '#839A8A']
		color_count = len(colors)
		# These need to be strings for `mosaic`
		cat1_levels = list(map(str, cat1.value_counts().keys().values))
		cat2_levels = list(map(str, cat2.value_counts().keys().values))
		# Display only the y-axis category inside the box:
		lab = (lambda key : key[1])
		fig, _ = mosaic(xtab, gap=0.01, labelizer=lab, properties=prop)
		figwidth, figheight = figsize
		fig.set_figwidth(figwidth)
		fig.set_figheight(figheight)
		fig.savefig('static/gender_vs_specialty.png')
		
		
	ys = pd.read_csv('prescription_data.csv', sep=',', low_memory=False)
	cat2='gender'
	specialty_mincount=1500
	specialty_maxcount=None
	cat2 = ys[cat2]
	specialty = ys['specialty']
	dist = specialty.value_counts()
	if specialty_maxcount == None:
		specialty_maxcount = dist.max()
	subset = [s for s,c in dist.items()
				if c >= specialty_mincount and c <= specialty_maxcount]
	specialty = specialty[specialty.isin(subset)]
	target_mosaic(cat2, specialty)
	
	def prop(key):
		c2, c1 = key
		cat1_index = cat1_levels.index(c1)
		cat2_index = cat2_levels.index(c2)
		base_color = colors[cat2_index % color_count]
		adjusted = increase_luminance(base_color, multiplier=cat1_index)
		return adjusted
		
	return render_template('gender_vs_specialty.html')
	
	
	
@app.route('/settlement_type')
def show_settlement_graph():
	df = pd.read_csv('prescription_data.csv', sep=',', low_memory=False)
	series = df['settlement_type']
	fig_size=(10,5)
	fig = plt.figure(figsize=(10,5))
	ax = plt.subplot()
	counts = series.value_counts()
	vals = counts.values
	pers = vals / vals.sum()
	ax = plt.subplot()
	counts.plot(title=series.name, kind='barh', figsize=fig_size, ax=ax, color = 'slateblue')
	plt.style.use('seaborn-whitegrid')
	nudged_vals = vals * 1.01  
	for i, val in enumerate(nudged_vals):
		formatted_per = '{:0.01%}'.format(pers[i])
		ax.text(nudged_vals[i], i, formatted_per)
	ax.set_xlim((0, max(counts.values)*1.2))
	fig.savefig('static/settlement_type.png')
	return render_template('settlement_type.html')


@app.route('/specialty')
def show_specialty_graph():
	df = pd.read_csv('prescription_data.csv', sep=',', low_memory=False)
	series = df['specialty']
	fig_size=(10,15)
	fig = plt.figure(figsize=(10,15))
	ax = plt.subplot()
	counts = series.value_counts()
	vals = counts.values
	pers = vals / vals.sum()
	ax = plt.subplot()
	counts.plot(title=series.name, kind='barh', figsize=fig_size, ax=ax, color = 'slateblue')
	plt.style.use('seaborn-whitegrid')
	nudged_vals = vals * 1.01  
	for i, val in enumerate(nudged_vals):
		formatted_per = '{:0.01%}'.format(pers[i])
		ax.text(nudged_vals[i], i, formatted_per)
	ax.set_xlim((0, max(counts.values)*1.2))
	fig.savefig('static/specialty.png')
	return render_template('specialty.html')
	

@app.route('/years_practice')
def years_practice_graph():
	df = pd.read_csv('prescription_data.csv', sep=',', low_memory=False)
	series = df['specialty']
	fig_size=(10,5)
	fig = plt.figure(figsize=(10,5))
	ax = plt.subplot()
	counts = series.value_counts()
	vals = counts.values
	pers = vals / vals.sum()
	ax = plt.subplot()
	counts.plot(title=series.name, kind='barh', figsize=fig_size, ax=ax, color = 'slateblue')
	plt.style.use('seaborn-whitegrid')
	nudged_vals = vals * 1.01  
	for i, val in enumerate(nudged_vals):
		formatted_per = '{:0.01%}'.format(pers[i])
		ax.text(nudged_vals[i], i, formatted_per)
	ax.set_xlim((0, max(counts.values)*1.2))
	fig.savefig('static/years_practice.png')
	return render_template('years_practice.html')








@app.route('/rules')
def show_rules():
    def encode_text_dummy(df, name):
        dummies = pd.get_dummies(df[name])
        for x in dummies.columns:
            dummy_name = "{}-{}".format(name, x)
            df[dummy_name] = dummies[x]
        df.drop(name, axis=1, inplace=True)
    
    # Read data csv in
    df = pd.read_csv('prescription_data.csv', sep=',', low_memory=False)
    # Create subset of data with only a few columns used for association analysis
    data = df[['gender', 'specialty', 'settlement_type']]
    encode_text_dummy(data, 'gender')
    encode_text_dummy(data, 'specialty')
    encode_text_dummy(data, 'settlement_type')
    #data.head()
    # Get frequent itemsets
    freq_items1 = apriori(data, min_support=0.009, use_colnames=True, verbose=1)
    freq_items1
    # Get the rules
    rules1 = association_rules(freq_items1, metric="confidence", min_threshold=0.2)
    #rules1
    #Test 1 Visualization
    plt.scatter(rules1['support'], rules1['confidence'], alpha=0.5)
    plt.xlabel('Support')
    plt.ylabel('Confidence')
    plt.title('Support vs Confidence')
    #plt.show()
    # Only grab needed columns from rule results
    rules1_results = rules1[['antecedents', 'consequents', 'confidence']]
    #rules1_results.head()
    #rules1_results['confidence'].values
    # Filter rules based on a relatively high confidence level - 90% 
    results = rules1_results[rules1_results['confidence'].values >= .9]

    results1 = results['antecedents']
    

    antecedents = ([list(x) for x in results1])
    length = len(antecedents)

    results2 = results['consequents']

    consequents = ([list(x) for x in results2])


    confidence = results['confidence'].tolist()

    return  render_template('rules.html', antecedents = antecedents, consequents = consequents, confidence = confidence, length = length)

    
@app.route('/region')
def target_barplot():
	df = pd.read_csv('prescription_data.csv', sep=',', low_memory=False)
	series = df['region']
	fig_size=(10,5)
	fig = plt.figure(figsize=(10,5))
	ax = plt.subplot()
	counts = series.value_counts()
	vals = counts.values
	pers = vals / vals.sum()
	ax = plt.subplot()
	counts.plot(title=series.name, kind='barh', figsize=fig_size, ax=ax, color = 'slateblue')
	plt.style.use('seaborn-whitegrid')
	nudged_vals = vals * 1.01  
	for i, val in enumerate(nudged_vals):
		formatted_per = '{:0.01%}'.format(pers[i])
		ax.text(nudged_vals[i], i, formatted_per)
	ax.set_xlim((0, max(counts.values)*1.2))
	fig.savefig('static/region.png')
	return render_template('region.html')


#if __name__ == '__main__':
#   app.run(debug=True, use_reloader=True)

#set FLASK_APP=app.py     python -m flask run