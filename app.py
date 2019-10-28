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
####

@app.route('/')
def homepage():
    return render_template('index.html')

@app.route('/run_data')
def rundata_click():
    return render_template('run_data.html')

@app.route('/gender')
def show_gener_graph():
    # Read data csv in
    df = pd.read_csv('prescription_data.csv', sep=',', low_memory=False)
    #graphing    
    fig = plt.figure(figsize=(10,10))
    plt.title('gender',fontdict={'fontsize':'30'})
    ax = sns.countplot(y='gender',data= df,palette='husl')
    ax.set(xlabel='Counts', ylabel='')
    plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.tight_layout()
    #plt.show()
    #saving graphs as png
    fig.savefig('static/gender.png')
    return render_template('gender.html')



@app.route('/settlement')
def show_settlement_graph():
    # Read data csv in
    df = pd.read_csv('prescription_data.csv', sep=',', low_memory=False)
    #graphing     
    fig = plt.figure(figsize=(10,10))
    plt.title('settlement_type',fontdict={'fontsize':'30'})
    ax = sns.countplot(y='settlement_type',data= df,palette='husl')
    ax.set(xlabel='Counts', ylabel='')
    plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.tight_layout()
    #plt.show()
    #saving graphs as png
    fig.savefig('static/settlement.png')
    return render_template('settlement.html')


@app.route('/specialty')
def show_specialty_graph():
    # Read data csv in
    df = pd.read_csv('prescription_data.csv', sep=',', low_memory=False)
    #graphing     
    fig = plt.figure(figsize=(10,10))
    plt.title('specialty',fontdict={'fontsize':'30'})
    ax = sns.countplot(y='specialty',data= df,palette='Spectral')
    ax.set(xlabel='Counts', ylabel = '')
    plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.tight_layout()
   # plt.show()
    #saving graphs as png
    fig.savefig('static/specialty.png')
    return render_template('specialty.html')



@app.route('/analyze_data')

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

    return  render_template('analyze.html', antecedents = antecedents, consequents = consequents, confidence = confidence, length = length)

    




#if __name__ == '__main__':
#   app.run(debug=True, use_reloader=True)

#set FLASK_APP=app.py     python -m flask run