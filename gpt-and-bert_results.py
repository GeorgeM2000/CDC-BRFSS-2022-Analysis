# -*- coding: utf-8 -*-
"""
Created on Thu May 9 15:12:24 2024

@author: tsintzask
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix,classification_report,roc_curve,roc_auc_score
from sklearn.calibration import calibration_curve,CalibrationDisplay

with open("bert_results.json","r") as f:
    bert_results = json.load(f)
del f

with open("gpt2_results.json","r") as f:
    gpt2_results = json.load(f) 
del f

cost_matrix=[
    [0, 1],
    [1, 0]
]

def calc_cost(conf_m):
    return np.sum(conf_m.T*cost_matrix)

"""
results_showcase(bert_results,"BERT")
results_showcase(gpt2_results,"GPT2")
results_plots(bert_results,"BERT")
results_plots(gpt2_results,"GPT2")
"""

def results_showcase(results:dict, model:str, printing=True):
    conf_m = confusion_matrix(results['labels'],results['predicted'])
    report = classification_report(results['labels'],results['predicted'],digits=5)
    
    out=f"{model} confusion matrix:\n{conf_m}\n\n{model} total cost: {calc_cost(conf_m)}\n\n{model} classification report:\n{report}\n"
    
    if printing:
        print(out)
    else:
        return(out)
    
def results_plots(results:dict, model:str):
    
    probabilities = [y[1] for y in results['probabilities']]
    prob_true, prob_pred = calibration_curve(results['labels'], probabilities, n_bins=10)
    disp = CalibrationDisplay(prob_true, prob_pred, probabilities)
    
    fig = plt.figure(dpi=600)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor('#f0ecf4')
    plt.grid(color="white")
    disp.plot(ax=ax)
    plt.title(f"{model} Probability Calibration Curve")
    plt.savefig(f"{model.lower()}_calibration_curve.png")
    plt.show()
    
    fpr, tpr, _ = roc_curve(results['labels'], probabilities)
    auc = roc_auc_score(results['labels'], probabilities)
    
    fig = plt.figure(dpi=600)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(fpr,tpr,label=f"ROC Curve (AUC={'{0:,.3f}'.format(auc)})")
    ax.plot([0,1],[0,1],label="Random Guessing",color="orange",linestyle="dashed")
    ax.set_facecolor('#f0ecf4')
    plt.title(f"{model} ROC Curve")
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.grid(color="white")
    plt.legend(loc=4)
    plt.savefig(f"{model.lower()}_roc_curve.png")
    plt.show()

def combined_calibration_plot():
    bert_probabilities = [y[1] for y in bert_results['probabilities']]
    bert_prob_true, bert_prob_pred = calibration_curve(bert_results['labels'], bert_probabilities, n_bins=10)
    bert_disp = CalibrationDisplay(bert_prob_true, bert_prob_pred, bert_probabilities)
    
    gpt2_probabilities = [y[1] for y in gpt2_results['probabilities']]
    gpt2_prob_true, gpt2_prob_pred = calibration_curve(gpt2_results['labels'], gpt2_probabilities, n_bins=10)
    gpt2_disp = CalibrationDisplay(gpt2_prob_true, gpt2_prob_pred, gpt2_probabilities)
    
    fig = plt.figure(dpi=600)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor('#f0ecf4')
    plt.grid(color="white")
    bert_disp.plot(ax=ax,label="BERT model")
    gpt2_disp.plot(ax=ax,label="GPT2 model")
    plt.title("Probability Calibration Curve")
    plt.legend(loc=4)
    plt.savefig("combined_calibration_curve.png")
    plt.show()

def combined_roc_auc_curve():
    bert_probabilities = [y[1] for y in bert_results['probabilities']]
    bert_prob_true, bert_prob_pred = calibration_curve(bert_results['labels'], bert_probabilities, n_bins=10)
    
    gpt2_probabilities = [y[1] for y in gpt2_results['probabilities']]
    gpt2_prob_true, gpt2_prob_pred = calibration_curve(gpt2_results['labels'], gpt2_probabilities, n_bins=10)

    bert_fpr, bert_tpr, _ = roc_curve(bert_results['labels'], bert_probabilities)
    bert_auc = roc_auc_score(bert_results['labels'], bert_probabilities)

    gpt2_fpr, gpt2_tpr, _ = roc_curve(gpt2_results['labels'], gpt2_probabilities)
    gpt2_auc = roc_auc_score(gpt2_results['labels'], gpt2_probabilities)
    
    fig = plt.figure(dpi=600)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(bert_fpr,bert_tpr,label=f"BERT ROC Curve (AUC={'{0:,.3f}'.format(bert_auc)})")
    ax.plot(gpt2_fpr,gpt2_tpr,label=f"GPT-2 ROC Curve (AUC={'{0:,.3f}'.format(gpt2_auc)})",color="green")
    ax.plot([0,1],[0,1],label="Random Guessing",color="orange",linestyle="dashed")
    ax.set_facecolor('#f0ecf4')
    plt.title("ROC Curve")
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.grid(color="white")
    plt.legend(loc=4)
    plt.savefig("combined_roc_curve.png")
    plt.show()

def combined_micro_performance():
    fig = plt.figure(dpi=600)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_axisbelow(True)
    ax.grid()
    ax.bar([0.55,1.55],[0.06122,0.05176],width=0.3,bottom=0.7,align='edge',label='Micro Precision')
    ax.bar([0.85,1.85],[0.05649,0.05173],width=0.3,bottom=0.7,align='edge',label='Micro Recall')
    ax.bar([1.15,2.15],[0.05512,0.05169],width=0.3,bottom=0.7,align='edge',label='Micro F1')
    plt.xticks(ticks=[0.5,0.75,1,1.25,1.5,1.75,2,2.25,2.5],labels=["","","BERT","","","","GPT-2","",""])
    plt.legend(loc=8)
    plt.title("Performance of Transformer Models")
    plt.show()