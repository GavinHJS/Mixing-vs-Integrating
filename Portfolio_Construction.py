# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 21:34:09 2023

@author: Gavin
"""
import pandas as pd
import numpy as np
from Alphas import load_alphas
from Data import DataCombiner
import os
import matplotlib.pyplot as plt
class portfolio_construction:
    def __init__(self, N , factors_in_focus,data):
        self.N_stocks = N 
        self.factors = factors_in_focus
        self.factor_weights = 1/len(self.factors)
        self.data = data
        self.data['Analysis Date'] = pd.to_datetime(self.data['Analysis Date'], format='%Y/%m/%d')
        self.dates =self.data['Analysis Date'].unique()
        
    def mixing_portfolio(self):
        self.mixing_portfolios = pd.DataFrame()
        for date in self.dates:
            df_date = self.data[self.data['Analysis Date'] == date]
            date_portfolios = pd.DataFrame()
            for factor in self.factors:
                factor_portfolio = df_date.sort_values(factor, ascending=False)
                factor_portfolio = factor_portfolio[:int(self.N_stocks * len(df_date))]
                factor_portfolio['Weight (%)'] = 1/ len(factor_portfolio)
                date_portfolios = pd.concat([date_portfolios, factor_portfolio])
            mixing_portfolio = date_portfolios.groupby("Asset ID").sum().reset_index()
            mixing_portfolio = pd.merge(mixing_portfolio[["Asset ID" ,"Weight (%)"]] , date_portfolios[['Asset ID', 'Analysis Date', 'Asset Name', 'Holdings',
                   'Country Of Exposure', 'GICS Industry', 'Price', 'Mkt Value'
                   , 'Risk Free Rate', 'Book-to-Price', 'Historical Beta',
                   'Relative Strength', 'Gross profitability', 'Dividend yield',
                   'Earnings per share Growth rate', 'Logarithm of Market Capitalization']] , how ="left" ,on = "Asset ID")
            mixing_portfolio = mixing_portfolio.drop_duplicates()
            mixing_portfolio['Weight (%)'] = mixing_portfolio['Weight (%)'] * self.factor_weights
            mixing_portfolio['Analysis Date'] = date
            self.mixing_portfolios = pd.concat([self.mixing_portfolios, mixing_portfolio])
    
        return self.mixing_portfolios
    
    def integrated_portfolio(self):
        self.integrated_portfolios = pd.DataFrame()
        for date in self.dates:
            df_date = self.data[self.data['Analysis Date'] == date]
            df_date.loc[:, 'Integrated Signal'] = df_date[self.factors].mean(axis=1)
            integrated_portfolio = df_date.sort_values('Integrated Signal', ascending=False)
            integrated_portfolio = integrated_portfolio[:int(self.N_stocks * len(df_date) )]
            integrated_portfolio['Weight (%)'] = 1 / len(integrated_portfolio)
            integrated_portfolio['Analysis Date'] = date 
            self.integrated_portfolios = pd.concat([self.integrated_portfolios, integrated_portfolio])
        return self.integrated_portfolios
    

# if __name__ == "__main__":
#         data = pd.read_csv("Final_Data.csv")
#         alpha_loader= load_alphas(data)
#         alpha_loader.winsorize_data(2.5)
#         # apply the standardization function
#         df_standardized = alpha_loader.standardize_daily()

#         # df_winsorized = alpha_loader.winsorize_data(cols_to_winsorize,[0.01, 0.01])
#         constructor = portfolio_construction(0.1 , ["Book-to-Price" , "Relative Strength" ] ,df_standardized )
#         portfolio_mix = constructor.mixing_portfolio()

