# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 19:29:38 2023

@author: Gavin
"""

import pandas as pd
from Data import DataCombiner
import numpy as np
from scipy import stats

from scipy.stats import mstats
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mstats

class load_alphas():
    def __init__(self,data):
        self.data = data
        self.plot_data = self.data.copy()

    def winsorize_data(self, n_std_dev):
        cols_to_winsorize = ['Book-to-Price', 'Historical Beta', 'Relative Strength', 
                             'Gross profitability', 'Dividend yield', 
                             'Earnings per share Growth rate', 
                             'Logarithm of Market Capitalization']
        grouped = self.data.groupby('Analysis Date')
        def winsorize_df(df):
            lower = norm.cdf(-n_std_dev)
            upper = norm.cdf(n_std_dev)
            limits = [lower, 1-upper]  
            for col in cols_to_winsorize:
                df[col] = mstats.winsorize(df[col], limits=limits)
            return df
        self.data = grouped.apply(winsorize_df)
        return self.data

    def replace_outliers_using_iqr(self):
        cols_to_check = ['Book-to-Price', 'Historical Beta', 'Relative Strength', 
                         'Gross profitability', 'Dividend yield', 
                         'Earnings per share Growth rate', 
                         'Logarithm of Market Capitalization']
        for col in cols_to_check:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            self.data[col] = np.where(self.data[col] < lower_bound, lower_bound, self.data[col])
            self.data[col] = np.where(self.data[col] > upper_bound, upper_bound, self.data[col])
        return self.data

    def boxplot_all_columns(self):
        cols_to_plot = ['Book-to-Price', 'Historical Beta', 'Relative Strength', 
                        'Gross profitability', 'Dividend yield', 
                        'Earnings per share Growth rate', 
                        'Logarithm of Market Capitalization']
        plt.figure(figsize=(15,10))
        for i, col in enumerate(cols_to_plot):
            plt.subplot(3,3,i+1) 
            sns.boxplot(x=self.data[col])
            plt.title(col)
            plt.tight_layout()
        plt.show()
        
    def plot_qq(self):
        cols_to_plot = ['Book-to-Price', 'Historical Beta', 'Relative Strength', 
                        'Gross profitability', 'Dividend yield', 
                        'Earnings per share Growth rate', 
                        'Logarithm of Market Capitalization']
        plt.figure(figsize=(15,10))
        for i, col in enumerate(cols_to_plot):
            plt.subplot(3,3,i+1) 
            stats.probplot(self.data[col], dist="norm", plot=plt)
            plt.title(f'QQ plot for {col}')
        plt.tight_layout()
        plt.show()

    def plot_winsorization(self,n_std_dev):
        cols_to_winsorize = ['Book-to-Price', 'Historical Beta', 'Relative Strength', 
                             'Gross profitability', 'Dividend yield', 
                             'Earnings per share Growth rate', 
                             'Logarithm of Market Capitalization']
        lower = norm.cdf(-n_std_dev)
        upper = norm.cdf(n_std_dev)
        limits = [lower, 1-upper]  
        plt.figure(figsize=(15,40)) 
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  
        for i, col in enumerate(cols_to_winsorize):
            plt.subplot(len(cols_to_winsorize), 2, 2*i+1) 
            sns.distplot(self.plot_data[col], kde=False, bins=30, color=colors[i])
            plt.title(f'Before Winsorization: {col}')
            plt.xticks(rotation=90)
            unique_dates = self.plot_data['Analysis Date'].unique()  
            for date in unique_dates:
                self.plot_data.loc[self.plot_data['Analysis Date'] == date, col] = \
                    mstats.winsorize(self.plot_data.loc[self.plot_data['Analysis Date'] == date, col], limits=limits)
            plt.subplot(len(cols_to_winsorize), 2, 2*i+2)  
            sns.distplot(self.plot_data[col], kde=False, bins=30, color=colors[i])
            plt.title(f'After Winsorization: {col}')
            plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

    def standardize_daily(self):
        df_standardized = self.data.copy()
        df_standardized['Analysis Date'] = pd.to_datetime(df_standardized['Analysis Date'], dayfirst=True,  infer_datetime_format=True)
        cols_to_standardize = ['Book-to-Price', 'Historical Beta', 'Relative Strength', 
                               'Gross profitability', 'Dividend yield', 
                               'Earnings per share Growth rate', 
                               'Logarithm of Market Capitalization']
        for date in df_standardized['Analysis Date'].unique():
            df_date = df_standardized[df_standardized['Analysis Date'] == date]
            total_market_cap = df_date['Mkt Value'].sum()
            df_date['weights'] = df_date['Mkt Value'] / total_market_cap
            for col in cols_to_standardize:
                weighted_mean = np.sum(df_date[col] * df_date['weights'])
                equal_std = np.std(df_date[col])
                df_standardized.loc[df_standardized['Analysis Date'] == date, col] = (df_date[col] - weighted_mean) / equal_std
        df_standardized['Historical Beta'] = df_standardized['Historical Beta'].apply(lambda x : -x)
        df_standardized['Logarithm of Market Capitalization'] = df_standardized['Logarithm of Market Capitalization'].apply(lambda x : -x)
        return df_standardized



# if __name__ == "__main__":

    # data = pd.read_csv("Final_Data.csv")
    # alpha_loader= load_alphas(data)
    # alpha_loader.replace_outliers_using_iqr()
    # alpha_loader.boxplot_all_columns()
    # alpha_loader.winsorize_data(2.5)
    # alpha_loader.boxplot_all_columns()
    # alpha_loader.plot_qq()
    # alpha_loader.plot_winsorization(2.5)
    # # apply the standardization function
    # df_standardized = alpha_loader.standardize_daily()

    # # df_winsorized = alpha_loader.winsorize_data(cols_to_winsorize,[0.01, 0.01])
    # df_standardized.to_csv("test.csv",index = False)
