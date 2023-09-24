# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 18:39:38 2023

@author: Gavin
"""

import pandas as pd
import numpy as np
from Alphas import load_alphas
from Data import DataCombiner
from Portfolio_Construction import portfolio_construction
import os
import matplotlib.pyplot as plt
from Performance import performance_calculation
import seaborn as sns
def plot_cumulative_returns(portfolio_returns_integrated, benchmark_returns_integrated, portfolio_returns_mixing):
    plt.figure(figsize=(12, 8))

    plt.plot(portfolio_returns_integrated['Analysis Date'], portfolio_returns_integrated['Portfolio Cumulative Return'], label='Integrated Portfolio', color='blue')
    plt.plot(benchmark_returns_integrated['Analysis Date'], benchmark_returns_integrated['Benchmark Cumulative Return'], label='Benchmark', color='skyblue')
    plt.plot(portfolio_returns_mixing['Analysis Date'], portfolio_returns_mixing['Portfolio Cumulative Return'], label='Mixing Portfolio', color='red')
    plt.title('Cumulative Returns Over Time')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.grid(True)
    plt.style.use('seaborn-dark-palette')
    plt.show()
    
def plot_monthly_returns_scatter(portfolio_returns, benchmark_returns, title):
    plt.figure(figsize=(12, 8))
    common_index = portfolio_returns.index.intersection(benchmark_returns.index)
    portfolio_returns = portfolio_returns.loc[common_index]
    benchmark_returns = benchmark_returns.loc[common_index]
    plt.scatter(benchmark_returns, portfolio_returns, alpha=0.6, edgecolors='w')
    plt.xlabel('Benchmark Returns')
    plt.ylabel('Portfolio Returns')
    plt.title(title)
    plt.grid(True)
    plt.style.use('seaborn-dark-palette')
    plt.show()
    
def correlation_matrix(result, title):
    corr = result.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True) # annot=True will show the correlation values
    plt.title(title)
    plt.show()
    
def plot_cumulative_excess_returns(portfolio_returns_integrated, benchmark_returns_integrated, portfolio_returns_mixing):
    plt.figure(figsize=(12, 8))
    plt.plot(portfolio_returns_integrated['Analysis Date'], portfolio_returns_integrated['Portfolio Excess Cumulative Return'], label='Integrated Portfolio', color='blue')
    plt.plot(benchmark_returns_integrated['Analysis Date'], benchmark_returns_integrated['Benchmark Excess Cumulative Return'], label='Benchmark', color='skyblue')
    plt.plot(portfolio_returns_mixing['Analysis Date'], portfolio_returns_mixing['Portfolio Excess Cumulative Return'], label='Mixing Portfolio', color='red')
    plt.title('Cumulative Excess Returns Over Time')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Excess Returns')
    plt.legend()
    plt.style.use('seaborn-dark-palette')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    pass