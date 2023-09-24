# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 21:26:58 2023

@author: Gavin
"""
import pandas as pd
import numpy as np
from Alphas import load_alphas
from Data import DataCombiner
from Portfolio_Construction import portfolio_construction
import os
import matplotlib.pyplot as plt

class performance_calculation():
    def __init__(self , market_data , portfolio, factors_in_focus):
        self.portfolio = portfolio
        self.data = market_data
        self.factors_in_focus = factors_in_focus
        self.performance_data = self.data.copy()
        self.performance_data =  self.performance_data.sort_values('Analysis Date')
        self.performance_data["Weight (%)"] = self.performance_data["Weight (%)"].str.rstrip('%').astype('float') / 100
        self.performance_data["Risk Free Rate"] = self.performance_data["Risk Free Rate"].astype('float').apply(lambda x : (1 + x)**(1/12) - 1)
        
    def calculate_benchmark_returns(self):
        self.benchmark_returns = self.performance_data.copy()
        self.benchmark_returns =  self.benchmark_returns.sort_values('Analysis Date')
        self.benchmark_returns['Risk Free Rate'] = pd.to_numeric(self.benchmark_returns['Risk Free Rate'], errors='coerce')
        self.benchmark_returns['Monthly Return'] =  self.benchmark_returns.groupby('Asset ID')['Price'].pct_change()
        median = self.benchmark_returns['Monthly Return'].median()
        std = self.benchmark_returns['Monthly Return'].std()
        upper_limit = median + 2.5 * std
        self.benchmark_returns.loc[self.benchmark_returns['Monthly Return'] > upper_limit, 'Monthly Return'] = 0
        self.benchmark_returns['Monthly Excess Return'] = self.benchmark_returns['Monthly Return'] - self.benchmark_returns['Risk Free Rate']
        self.benchmark_returns['Weight (%)'] = self.benchmark_returns.groupby('Asset ID')['Weight (%)'].ffill()
        self.benchmark_returns['Weighted Return'] = self.benchmark_returns['Monthly Return'] * self.benchmark_returns['Weight (%)']
        self.benchmark_returns['Weighted Excess Return'] = self.benchmark_returns['Monthly Excess Return'] * self.benchmark_returns['Weight (%)']
        benchmark_returns = self.benchmark_returns.groupby('Analysis Date')['Weighted Return'].sum()
        benchmark_returns_df = pd.DataFrame(benchmark_returns)
        benchmark_returns_df.reset_index(inplace=True)
        benchmark_returns_df.columns = ['Analysis Date', 'Benchmark Return']
        self.benchmark_returns_only = benchmark_returns_df
        benchmark_excess_returns = self.benchmark_returns.groupby('Analysis Date')['Weighted Excess Return'].sum()
        benchmark_excess_returns_df = pd.DataFrame(benchmark_excess_returns)
        benchmark_excess_returns_df.reset_index(inplace=True)
        benchmark_excess_returns_df.columns = ['Analysis Date', 'Benchmark Excess Return']
        self.benchmark_returns_only = pd.merge(self.benchmark_returns_only, benchmark_excess_returns_df, how='left', on='Analysis Date')
        self.benchmark_returns = pd.merge(self.benchmark_returns, benchmark_returns_df, how='left', on='Analysis Date')
        return self.benchmark_returns

    
    def calculate_portfolio_returns(self):
        self.portfolio_returns_data = self.portfolio.copy()
        self.portfolio_returns = self.benchmark_returns[["Asset ID",	"Analysis Date",	"Asset Name",
                                                         "Holdings",	"Country Of Exposure",	"GICS Industry",
                                                         "Price"	,"Mkt Value",	"Risk Free Rate",
                                                         "Book-to-Price",	"Historical Beta"	,"Relative Strength",
                                                         "Gross profitability",	"Dividend yield"	,"Earnings per share Growth rate",
                                                         "Logarithm of Market Capitalization"]]
        self.portfolio_returns = self.portfolio_returns.merge(
        self.portfolio_returns_data[['Asset ID', 'Analysis Date', 'Weight (%)']],
        on=['Asset ID', 'Analysis Date'],
        how='left')
        self.portfolio_returns['Weight (%)'].fillna(0, inplace=True)

        self.portfolio_returns =  self.portfolio_returns.sort_values('Analysis Date')
        try:
            self.portfolio_returns['Risk Free Rate'] = pd.to_numeric(self.portfolio_returns['Risk Free Rate'], errors='coerce')
        except Exception:
            pass
        # self.portfolio_returns["Risk Free Rate"] = self.portfolio_returns["Risk Free Rate"].astype('float').apply(lambda x : (1 + x)**(1/12) - 1)
        self.portfolio_returns['Monthly Return'] =  self.portfolio_returns.groupby('Asset ID')['Price'].pct_change()
        # self.portfolio_returns.loc[self.portfolio_returns['Monthly Return'] > 0.4, 'Monthly Return'] = 0
        self.portfolio_returns['Monthly Excess Return'] = self.portfolio_returns['Monthly Return'] - self.portfolio_returns['Risk Free Rate']
        self.portfolio_returns['Weight (%)'] = self.portfolio_returns.groupby('Asset ID')['Weight (%)'].ffill()
        self.portfolio_returns['Weighted Return'] = self.portfolio_returns['Monthly Return'] * self.portfolio_returns['Weight (%)']
        self.portfolio_returns['Weighted Excess Return'] = self.portfolio_returns['Monthly Excess Return'] * self.portfolio_returns['Weight (%)']
        portfolio_returns = self.portfolio_returns.groupby('Analysis Date')['Weighted Return'].sum()
        portfolio_returns_df = pd.DataFrame(portfolio_returns)
        portfolio_returns_df.reset_index(inplace=True)
        portfolio_returns_df.columns = ['Analysis Date', 'Portfolio Return']
        self.portfolio_returns_only = portfolio_returns_df
        portfolio_excess_returns = self.portfolio_returns.groupby('Analysis Date')['Weighted Excess Return'].sum()
        portfolio_excess_returns_df = pd.DataFrame(portfolio_excess_returns)
        portfolio_excess_returns_df.reset_index(inplace=True)
        portfolio_excess_returns_df.columns = ['Analysis Date', 'Portfolio Excess Return']

        self.portfolio_returns_only = pd.merge(self.portfolio_returns_only, portfolio_excess_returns_df, how='left', on='Analysis Date')
        self.portfolio_returns = pd.merge(self.portfolio_returns, portfolio_returns_df, how='left', on='Analysis Date')
        return self.portfolio_returns
    
    
    def calculate_cumulative_returns(self):
        self.benchmark_returns_only['Benchmark Cumulative Return'] = (1 + self.benchmark_returns_only['Benchmark Return']).cumprod() - 1
        self.portfolio_returns_only['Portfolio Cumulative Return'] = (1 + self.portfolio_returns_only['Portfolio Return']).cumprod() - 1
        self.benchmark_returns_only['Benchmark Excess Cumulative Return'] = (1 + self.benchmark_returns_only['Benchmark Excess Return']).cumprod() - 1
        self.portfolio_returns_only['Portfolio Excess Cumulative Return'] = (1 + self.portfolio_returns_only['Portfolio Excess Return']).cumprod() - 1
        return self.benchmark_returns_only, self.portfolio_returns_only
    
    def calculate_average_annualized_returns(self):
        total_period_in_years = len(self.portfolio_returns_only) / 12
        final_portfolio_value = (1 + self.portfolio_returns_only['Portfolio Return']).cumprod().iloc[-1]
        final_benchmark_value = (1 + self.benchmark_returns_only['Benchmark Return']).cumprod().iloc[-1]
        final_portfolio_excess_value = (1 + self.portfolio_returns_only['Portfolio Excess Return']).cumprod().iloc[-1]
        final_benchmark_excess_value = (1 + self.benchmark_returns_only['Benchmark Excess Return']).cumprod().iloc[-1]
        
        portfolio_annualized_return = (final_portfolio_value ** (1 / total_period_in_years)) - 1
        benchmark_annualized_return = (final_benchmark_value ** (1 / total_period_in_years)) - 1
        portfolio_annualized_excess_return = (final_portfolio_excess_value ** (1 / total_period_in_years)) - 1
        benchmark_annualized_excess_return = (final_benchmark_excess_value ** (1 / total_period_in_years)) - 1
        
        return portfolio_annualized_return, benchmark_annualized_return, portfolio_annualized_excess_return, benchmark_annualized_excess_return

    def calculate_volatility(self,returns):
        return np.std(returns)* np.sqrt(12)

    def calculate_sharpe(self):
        return (np.mean(self.portfolio_returns_only['Portfolio Return']) * 12) / self.calculate_volatility(self.portfolio_returns_only['Portfolio Return'])


    def calculate_tracking_error(self):
        return np.std(self.portfolio_returns_only['Portfolio Return']- self.benchmark_returns_only['Benchmark Return']) * np.sqrt(12)
    
    def calculate_information_ratio(self):
        return ((np.mean(self.portfolio_returns_only['Portfolio Return'] - self.benchmark_returns_only['Benchmark Return']) * 12) / self.calculate_tracking_error())
    
    def calculate_portfolio_exposures(self):
        results = pd.DataFrame()
        portfolio_exposure = self.portfolio.copy()
        for factor in self.factors_in_focus:
            portfolio_exposure[f'{factor}_weighted'] = portfolio_exposure['Weight (%)'] * portfolio_exposure[factor]
        grouped = portfolio_exposure.groupby('Analysis Date').sum()
        for factor in self.factors_in_focus:
            results[f'{factor}_sum'] = grouped[f'{factor}_weighted']        
        average = results.mean()
        return average,results
    
    def calculate_max_drawdown(self, returns):
        cum_returns = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cum_returns)
        running_max[running_max < 1] = 1
        drawdown = (cum_returns)/running_max - 1
        return drawdown.min()
            
    def calculate_average_numberofassets(self):
        average_assets_per_date = self.portfolio.groupby('Analysis Date')['Asset ID'].count().mean()
        return int(average_assets_per_date)
    
    def calculate_all(self):
        portfolio_volatility = self.calculate_volatility(self.portfolio_returns_only['Portfolio Return'])
        portfolio_sharpe = self.calculate_sharpe()
        information_ratio = self.calculate_information_ratio()
        tracking_error = self.calculate_tracking_error()
        max_drawdown = self.calculate_max_drawdown(self.portfolio_returns_only['Portfolio Return'])
        return portfolio_volatility, portfolio_sharpe, information_ratio, tracking_error, max_drawdown

    def plot_cumulative_returns(self):
        benchmark_returns, portfolio_returns = self.calculate_cumulative_returns()
        benchmark_returns['Analysis Date'] = pd.to_datetime(benchmark_returns['Analysis Date'], infer_datetime_format=True)
        portfolio_returns['Analysis Date'] = pd.to_datetime(portfolio_returns['Analysis Date'], infer_datetime_format=True)
        plt.figure(figsize=(10,6))
        plt.plot(benchmark_returns['Analysis Date'], benchmark_returns['Benchmark Cumulative Return'], label='Benchmark')
        plt.plot(portfolio_returns['Analysis Date'], portfolio_returns['Portfolio Cumulative Return'], label='Portfolio')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.title('Cumulative Returns over Time')
        plt.show() 


# if __name__ =="__main__":
#     directory = "MSCI_WORLD Data"
#     folders =os.listdir(directory)

#     data_combiner = DataCombiner(directory, folders)
#     data_combiner.load_data()
#     data_combiner.replace_missing_data()
#     combined_data_df = data_combiner.get_combined_data()
#     # ['Book-to-Price', 'Historical Beta', 'Relative Strength', 
#     #                        'Gross profitability', 'Dividend yield', 
#     #                        'Earnings per share Growth rate', 
#     #                        'Logarithm of Market Capitalization']
#     constructor = portfolio_construction(0.1 , ["Book-to-Price" , "Relative Strength",'Historical Beta' ] , True)
#     portfolio_integrated = constructor.integrated_portfolio()
#     portfolio_mixing = constructor.mixing_portfolio()    

    
#     print(""" ########################### Integrated Approach ##############################
              
#           """)
#     x = performance_calculation(combined_data_df ,portfolio_integrated )
#     value = x.calculate_benchmark_returns()
#     value2 = x.calculate_portfolio_returns()
#     x.plot_cumulative_returns()
#     portfolio_volatility_integrated, portfolio_sharpe_integrated,portfolio_information_ratio_integrated, tracking_error_integrated = x.calculate_all()

#     print('Portfolio Volatility:', portfolio_volatility_integrated)
#     print('Portfolio Sharpe Ratio:', portfolio_sharpe_integrated)
#     print('Portfolio Information Ratio:', portfolio_information_ratio_integrated)
#     print('Tracking Error:', tracking_error_integrated)

    
#     print(""" ########################### Mixing Approach ##############################
              
#           """)
#     y = performance_calculation(combined_data_df ,portfolio_mixing )
#     value = y.calculate_benchmark_returns()
#     value2 = y.calculate_portfolio_returns()
#     y.plot_cumulative_returns()
#     portfolio_volatility_mixing, portfolio_sharpe_mixing,portfolio_information_ratio_mixing, tracking_error_mixing = y.calculate_all()

#     print('Portfolio Volatility:', portfolio_volatility_mixing)
#     print('Portfolio Sharpe Ratio:', portfolio_sharpe_mixing)
#     print('Portfolio Information Ratio:', portfolio_information_ratio_mixing)
#     print('Tracking Error:', tracking_error_mixing)

        