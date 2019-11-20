
#

from flask import Flask, render_template, redirect, url_for, request
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
from colour import Color
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

df = pd.read_csv('fulldata.csv', sep=',', low_memory=False)
data2 = df[['brand_name_rx_count', 'gender', 'generic_rx_count', 'region', 'settlement_type', 'specialty', 'years_practicing']]


	


@app.route('/')
@app.route('/landing')
def landingpage():
	if os.path.exists("static/gender.svg"):
		os.remove("static/gender.svg")
	if os.path.exists("static/gender_vs_region.svg"):
		os.remove("static/gender_vs_region.svg")
	if os.path.exists("static/gender_vs_settlement.svg"):
		os.remove("static/gender_vs_settlement.svg")
	if os.path.exists("static/gender_vs_specialty.svg"):
		os.remove("static/gender_vs_specialty.svg")
	if os.path.exists("static/gender_vs_years.svg"):
		os.remove("static/gender_vs_years.svg")
		
	if os.path.exists("static/region.svg"):
		os.remove("static/region.svg")
	if os.path.exists("static/region_vs_settlement.svg"):
		os.remove("static/region_vs_settlement.svg")
	
	if os.path.exists("static/settlement_type.svg"):
		os.remove("static/settlement_type.svg")
	if os.path.exists("static/settlement_vs_specialty.svg"):
		os.remove("static/settlement_vs_specialty.svg")
	
	if os.path.exists("static/specialty.svg"):
		os.remove("static/specialty.svg")
	
	if os.path.exists("static/years_practice.svg"):
		os.remove("static/years_practice.svg")
		
	return render_template('landing.html')
	
	
@app.route('/terms')
def termspage():
	return render_template('terms.html')

	
@app.route('/index')
@app.route('/dashboard')
def index_page():
	return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
	user_name_array = ["test1","test2"]
	password_array = ["user1","user2"]
	error = None
	if request.method == 'POST':
		if(request.form['username'] in user_name_array):	#if username in array
			temp_index = user_name_array.index(request.form['username'])	#get index of username
			#check if password in passowrd array and at same index of username
			if((request.form['password'] in password_array) and password_array.index(request.form['password']) == temp_index):
				return redirect(url_for('index_page'))
				#if password not in password array
			elif (request.form['password'] not in password_array): 
				error = 'Invalid Credentials. Please try again'
				#can not mixmatch credentials
			if((request.form['password'] in password_array) and password_array.index(request.form['password']) != temp_index):
				error = 'Invalid Credentials. Please try again'
		# if username not in username_array
		else:
			error = 'Invalid Credentials. Please try again'

	return render_template('login.html', error=error)

@app.route('/drug_analyze')
def drug_analyze_click():
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = 'Top 20 Drugs', 'Other'
    sizes = [30, 70]
    explode = (0.1, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
    colors = ['mediumspringgreen', 'mediumslateblue']
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, colors= colors, autopct='%1.1f%%', shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.savefig('static/pie_chart.svg')
    return render_template('graph_files/drug_analyze.html')
	
@app.route('/analyze_data')
def rundata_click():
    return render_template('analyze_data.html')

@app.route('/pull_data')
def pulldata_click():
    df_drugs = df.drop(['brand_name_rx_count', 'gender', 'generic_rx_count', 'region', 'settlement_type', 'specialty', 'years_practicing'], axis=1)
    total_records = data2.shape[0]
    region = data2['region'].value_counts()
    south = region[0]
    
    return render_template('pull_data_API.html', total_records = total_records, south = south)

@app.route('/drugs')
def drugs_click():
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = 'Top 20 Drugs', 'Other'
    sizes = [30, 70]
    explode = (0.1, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
    colors = ['mediumspringgreen', 'mediumslateblue']
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, colors= colors, autopct='%1.1f%%', shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.savefig('static/pie_chart.svg')
    return render_template('drugs.html')

@app.route('/gender')
def gender_barplot():
	
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
	fig.savefig('static/gender.svg')
	return render_template('graph_files/gender.html')

@app.route('/gender_vs_specialty')
def gender_specialty_graph():

	def increase_luminance(color_str, multiplier=0):
		c = Color(color_str)
		lum = 0.8 - np.repeat(0.1, multiplier).sum()
		c.luminance = lum
		return {'color': str(c)}


	def target_mosaic(cat1, cat2, figsize=(18, 4)):
		xtab = pd.crosstab(cat1, cat2).unstack()
		# Bas colors:
		colors = ['#0499CC', '#4D8951', '#FDBA58', '#876DB5', '#32A8B4', '#9BB8D7', '#839A8A']
		color_count = len(colors)
		# These need to be strings for `mosaic`
		cat1_levels = list(map(str, cat1.value_counts().keys().values))
		cat2_levels = list(map(str, cat2.value_counts().keys().values))
		
		def prop(key):
			c2, c1 = key
			cat1_index = cat1_levels.index(c1)
			cat2_index = cat2_levels.index(c2)
			base_color = colors[cat2_index % color_count]
			adjusted = increase_luminance(base_color, multiplier=cat1_index)
			return adjusted
		# Display only the y-axis category inside the box:
		lab = (lambda key : key[1])
		fig, _ = mosaic(xtab, gap=0.01, labelizer=lab, properties=prop)
		figwidth, figheight = figsize
		fig.set_figwidth(figwidth)
		fig.set_figheight(figheight)
		fig.savefig('static/gender_vs_specialty.svg')
		
	cat2='gender'
	specialty_mincount=8000
	specialty_maxcount=50000
	cat2 = df[cat2]
	specialty = df['specialty']
	dist = specialty.value_counts()
	if specialty_maxcount == None:
		specialty_maxcount = dist.max()
	subset = [s for s,c in dist.items()
				if c >= specialty_mincount and c <= specialty_maxcount]
	specialty = specialty[specialty.isin(subset)]
	target_mosaic(cat2, specialty)
	return render_template('graph_files/gender_vs_specialty.html')
	
@app.route('/gender_vs_settlement')	
def gender_settlement_graph(): 
	
	def increase_luminance(color_str, multiplier=0):
		c = Color(color_str)
		lum = 0.8 - np.repeat(0.1, multiplier).sum()
		c.luminance = lum
		return {'color': str(c)}

	def target_mosaic(cat1, cat2, figsize=(18, 4)):
		xtab = pd.crosstab(cat1, cat2).unstack()
		# Bas colors:
		colors = ['#0499CC', '#4D8951', '#FDBA58', '#876DB5', '#32A8B4', '#9BB8D7', '#839A8A']
		color_count = len(colors)
		# These need to be strings for `mosaic`
		cat1_levels = list(map(str, cat1.value_counts().keys().values))
		cat2_levels = list(map(str, cat2.value_counts().keys().values))
		
		def prop(key):
			c2, c1 = key
			cat1_index = cat1_levels.index(c1)
			cat2_index = cat2_levels.index(c2)
			base_color = colors[cat2_index % color_count]
			adjusted = increase_luminance(base_color, multiplier=cat1_index)
			return adjusted
		# Display only the y-axis category inside the box:
		lab = (lambda key : key[1])
		fig, _ = mosaic(xtab, gap=0.01, labelizer=lab, properties=prop)
		figwidth, figheight = figsize
		fig.set_figwidth(figwidth)
		fig.set_figheight(figheight)
		fig.savefig('static/gender_vs_settlement.svg')
		
		
	
	cat2='gender'
	settlement_mincount=1500
	settlement_maxcount=None
	cat2 = data2[cat2]
	settlement = data2['settlement_type']
	dist = settlement.value_counts()
	if settlement_maxcount == None:
		settlement_maxcount = dist.max()
	subset = [s for s,c in dist.items()
				if c >= settlement_mincount and c <= settlement_maxcount]
	settlement = settlement[settlement.isin(subset)]
	target_mosaic(cat2, settlement)
	return render_template('graph_files/gender_vs_settlement.html')
	
@app.route('/gender_vs_region')
def gender_region_graph():

	def increase_luminance(color_str, multiplier=0):
		c = Color(color_str)
		lum = 0.8 - np.repeat(0.1, multiplier).sum()
		c.luminance = lum
		return {'color': str(c)}


	def target_mosaic(cat1, cat2, figsize=(18, 4)):
		xtab = pd.crosstab(cat1, cat2).unstack()
		# Bas colors:
		colors = ['#0499CC', '#4D8951', '#FDBA58', '#876DB5', '#32A8B4', '#9BB8D7', '#839A8A']
		color_count = len(colors)
		# These need to be strings for `mosaic`
		cat1_levels = list(map(str, cat1.value_counts().keys().values))
		cat2_levels = list(map(str, cat2.value_counts().keys().values))
		
		def prop(key):
			c2, c1 = key
			cat1_index = cat1_levels.index(c1)
			cat2_index = cat2_levels.index(c2)
			base_color = colors[cat2_index % color_count]
			adjusted = increase_luminance(base_color, multiplier=cat1_index)
			return adjusted
		# Display only the y-axis category inside the box:
		lab = (lambda key : key[1])
		fig, _ = mosaic(xtab, gap=0.01, labelizer=lab, properties=prop)
		figwidth, figheight = figsize
		fig.set_figwidth(figwidth)
		fig.set_figheight(figheight)
		fig.savefig('static/gender_vs_region.svg')
		
		
	#ys = pd.read_csv('prescription_data.csv', sep=',', low_memory=False)
	cat2='gender'
	region_mincount=1500
	region_maxcount=None
	cat2 = data2[cat2]
	region = data2['region']
	dist = region.value_counts()
	if region_maxcount == None:
		region_maxcount = dist.max()
	subset = [s for s,c in dist.items()
				if c >= region_mincount and c <= region_maxcount]
	region = region[region.isin(subset)]
	target_mosaic(cat2, region)
	return render_template('graph_files/gender_vs_region.html')


@app.route('/gender_vs_years')
def gender_years_graph():

	def increase_luminance(color_str, multiplier=0):
		c = Color(color_str)
		lum = 0.8 - np.repeat(0.1, multiplier).sum()
		c.luminance = lum
		return {'color': str(c)}


	def target_mosaic(cat1, cat2, figsize=(18, 4)):
		xtab = pd.crosstab(cat1, cat2).unstack()
		# Bas colors:
		colors = ['#0499CC', '#4D8951', '#FDBA58', '#876DB5', '#32A8B4', '#9BB8D7', '#839A8A']
		color_count = len(colors)
		# These need to be strings for `mosaic`
		cat1_levels = list(map(str, cat1.value_counts().keys().values))
		cat2_levels = list(map(str, cat2.value_counts().keys().values))
		
		def prop(key):
			c2, c1 = key
			cat1_index = cat1_levels.index(c1)
			cat2_index = cat2_levels.index(c2)
			base_color = colors[cat2_index % color_count]
			adjusted = increase_luminance(base_color, multiplier=cat1_index)
			return adjusted
		# Display only the y-axis category inside the box:
		lab = (lambda key : key[1])
		fig, _ = mosaic(xtab, gap=0.01, labelizer=lab, properties=prop)
		figwidth, figheight = figsize
		fig.set_figwidth(figwidth)
		fig.set_figheight(figheight)
		fig.savefig('static/gender_vs_years.svg')
		
		
	#ys = pd.read_csv('prescription_data.csv', sep=',', low_memory=False)
	cat2='gender'
	years_mincount=7000
	years_maxcount=120000
	cat2 = data2[cat2]
	years = data2['years_practicing']
	dist = years.value_counts()
	if years_maxcount == None:
		years_maxcount = dist.max()
	subset = [s for s,c in dist.items()
				if c >= years_mincount and c <= years_maxcount]
	years = years[years.isin(subset)]
	target_mosaic(cat2, years)
	return render_template('graph_files/gender_vs_years.html')
	
@app.route('/settlement_type')
def show_settlement_graph():
	#df = pd.read_csv('prescription_data.csv', sep=',', low_memory=False)
	series = data2['settlement_type']
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
	fig.savefig('static/settlement_type.svg')
	return render_template('graph_files/settlement_type.html')


@app.route('/region_vs_settlement')
def region_settlement_graph():

	def increase_luminance(color_str, multiplier=0):
		c = Color(color_str)
		lum = 0.8 - np.repeat(0.1, multiplier).sum()
		c.luminance = lum
		return {'color': str(c)}


	def target_mosaic(cat1, cat2, figsize=(18, 4)):
		xtab = pd.crosstab(cat1, cat2).unstack()
		# Bas colors:
		colors = ['#0499CC', '#4D8951', '#FDBA58', '#876DB5', '#32A8B4', '#9BB8D7', '#839A8A']
		color_count = len(colors)
		# These need to be strings for `mosaic`
		cat1_levels = list(map(str, cat1.value_counts().keys().values))
		cat2_levels = list(map(str, cat2.value_counts().keys().values))
		
		def prop(key):
			c2, c1 = key
			cat1_index = cat1_levels.index(c1)
			cat2_index = cat2_levels.index(c2)
			base_color = colors[cat2_index % color_count]
			adjusted = increase_luminance(base_color, multiplier=cat1_index)
			return adjusted
		# Display only the y-axis category inside the box:
		lab = (lambda key : key[1])
		fig, _ = mosaic(xtab, gap=0.01, labelizer=lab, properties=prop)
		figwidth, figheight = figsize
		fig.set_figwidth(figwidth)
		fig.set_figheight(figheight)
		fig.savefig('static/region_vs_settlement.svg')
		
	#ys = pd.read_csv('prescription_data.csv', sep=',', low_memory=False)
	cat2='settlement_type'
	region_mincount=1500
	region_maxcount=None
	cat2 = data2[cat2]
	region = data2['region']
	dist = region.value_counts()
	if region_maxcount == None:
		region_maxcount = dist.max()
	subset = [s for s,c in dist.items()
				if c >= region_mincount and c <= region_maxcount]
	region = region[region.isin(subset)]
	target_mosaic(cat2, region)
	return render_template('graph_files/region_vs_settlement.html')


@app.route('/settlement_vs_specialty')
def settlement_specialty_graph():

	def increase_luminance(color_str, multiplier=0):
		c = Color(color_str)
		lum = 0.8 - np.repeat(0.1, multiplier).sum()
		c.luminance = lum
		return {'color': str(c)}


	def target_mosaic(cat1, cat2, figsize=(18, 4)):
		xtab = pd.crosstab(cat1, cat2).unstack()
		# Bas colors:
		colors = ['#0499CC', '#4D8951', '#FDBA58', '#876DB5', '#32A8B4', '#9BB8D7', '#839A8A']
		color_count = len(colors)
		# These need to be strings for `mosaic`
		cat1_levels = list(map(str, cat1.value_counts().keys().values))
		cat2_levels = list(map(str, cat2.value_counts().keys().values))
		
		def prop(key):
			c2, c1 = key
			cat1_index = cat1_levels.index(c1)
			cat2_index = cat2_levels.index(c2)
			base_color = colors[cat2_index % color_count]
			adjusted = increase_luminance(base_color, multiplier=cat1_index)
			return adjusted
		# Display only the y-axis category inside the box:
		lab = (lambda key : key[1])
		fig, _ = mosaic(xtab, gap=0.01, labelizer=lab, properties=prop)
		figwidth, figheight = figsize
		fig.set_figwidth(figwidth)
		fig.set_figheight(figheight)
		fig.savefig('static/settlement_vs_specialty.svg')
		
		
	
	cat2='settlement_type'
	specialty_mincount=8000
	specialty_maxcount=50000
	cat2 = data2[cat2]
	specialty = data2['specialty']
	dist = specialty.value_counts()
	if specialty_maxcount == None:
		specialty_maxcount = dist.max()
	subset = [s for s,c in dist.items()
				if c >= specialty_mincount and c <= specialty_maxcount]
	specialty = specialty[specialty.isin(subset)]
	target_mosaic(cat2, specialty)
	return render_template('graph_files/settlement_vs_specialty.html')
	

@app.route('/specialty')
def show_specialty_graph():
	#df = pd.read_csv('prescription_data.csv', sep=',', low_memory=False)
	
	df_specialty = df[['specialty']]
	list2 = ['General Practice', 'Family', 'Psychiatry', 'Cardiovascular Disease', 'Medical', 'Gastroenterology', 'Neurology', 'Adult Health', 'Nephrology', 'Hematology & Oncology', 'Pulmonary Disease', 'Endocrinology, Diabetes & Metabolism', 'Oral and Maxillofacial Surgery']           
	specialty_data = data2[data2['specialty'].isin(list2)]
	
	series = specialty_data['specialty']
	fig_size=(18,15)
	fig = plt.figure(figsize=(18,15))
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
	fig.savefig('static/specialty.svg')
	return render_template('graph_files/specialty.html')
	

@app.route('/years_practice')
def years_practice_graph():
	#df = pd.read_csv('prescription_data.csv', sep=',', low_memory=False)
	series = data2['years_practicing']
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
	fig.savefig('static/years_practice.svg')
	return render_template('graph_files/years_practice.html')


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

    return  render_template('rules/rules.html', antecedents = antecedents, consequents = consequents, confidence = confidence, length = length)

    
@app.route('/region')
def target_barplot():
	#df = pd.read_csv('prescription_data.csv', sep=',', low_memory=False)
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
	fig.savefig('static/region.svg')
	return render_template('graph_files/region.html')
    
 
@app.route('/drug_demo_graph_generation')
def graph_generate():
    def encode_text_dummy(df, name):
        dummies = pd.get_dummies(df[name])
        for x in dummies.columns:
            dummy_name = "{}-{}".format(name, x)
            df[dummy_name] = dummies[x]
        df.drop(name, axis=1, inplace=True)

    # Convert a Pandas dataframe to the x,y inputs that TensorFlow needs
    def to_xy(df, target):
        result = []
        for x in df.columns:
            if x != target:
                result.append(x)
        # find out the type of the target column. 
        target_type = df[target].dtypes
        target_type = target_type[0] if isinstance(target_type, Sequence) else target_type
        # Encode to int for classification, float otherwise. TensorFlow likes 32 bits.
        if target_type in (np.int64, np.int32):
            # Classification
            dummies = pd.get_dummies(df[target])
            return df[result].values.astype(np.float32), dummies.values.astype(np.float32)
        else:
            # Regression
            return df[result].values.astype(np.float32), df[target].values.astype(np.float32)
    dataset_filename = 'data.jsonl'
    def iter_dataset():
        with open(dataset_filename, 'rt') as f:
            for line in f:
                ex = json.loads(line)
                yield (ex['cms_prescription_counts'],
                       ex['provider_variables'])

    def merge_dicts(*dicts: dict):
        merged_dict = dict()
        for dictionary in dicts:
            merged_dict.update(dictionary)
        return merged_dict

    data = [merge_dicts(x, y) for x, y in iter_dataset()]
    df = pd.DataFrame(data)
    df.fillna(0, inplace=True)

    df.drop(columns='gender', inplace=True)
    df.drop(columns='region', inplace=True)
    df.drop(columns='settlement_type', inplace=True)
    df.drop(columns='years_practicing', inplace=True)

    drugName = "specialty-Pulmonary Diagnostics"

    encode_text_dummy(df, 'specialty')
    # Encode to a 2D matrix for training
    x,y = to_xy(df, drugName)

    # Split into train/test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=45) 

    #regressor = LinearRegression()
    regressor = LinearRegression()

    # Fit/train linear regression
    regressor.fit(x_train,y_train)

    # Predict
    pred = regressor.predict(x_test)

    # Measure RMSE error.  RMSE is common for regression.
    score = np.sqrt(metrics.mean_squared_error(pred,y_test))
    
    names.remove(drugName)


    def report_coef(names,coef,intercept):
        r = pd.DataFrame( { 'coef': coef, 'positive': coef>0.4  }, index = names )
        r = r.sort_values(by=['coef'])
        
        badRows = r[(r['positive'] == False)].index
        #badCoef = r[(r['coef'] <= 1.00e-02)].index
        r.drop(badRows, inplace=True)
        #r.drop(badCoef, inplace=True)
        
        display(r)
        print("Intercept: {}".format(intercept))
        r['coef'].plot(kind='barh', color=r['positive'].map({True: 'b', False: 'r'}))
        
        
    report_coef(names, (regressor.coef_ * 1.9), regressor.intercept_)

#if __name__ == '__main__':
#   app.run(debug=True, use_reloader=True)

#set FLASK_APP=app.py     python -m flask run