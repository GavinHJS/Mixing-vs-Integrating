# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 23:44:09 2023

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
from Visualization import plot_cumulative_returns,plot_monthly_returns_scatter,correlation_matrix,plot_cumulative_excess_returns

    

if __name__ =="__main__":

    
    """ Control Panel"""
    factors_in_focus =['Book-to-Price', 'Historical Beta', 'Relative Strength', 
                            'Gross profitability', 'Dividend yield', 
                            'Earnings per share Growth rate', 
                            'Logarithm of Market Capitalization']
    directory = "MSCI_WORLD Data"
    percentage_of_stock_holdings = 0.1
    winsorization_value = 2.5 #input is in std dev
    

    """End of Control Panel"""
    folders =os.listdir(directory)

    data_combiner = DataCombiner(directory, folders)
    data_combiner.load_data()
    data_combiner.replace_missing_data()
    combined_data_df = data_combiner.get_combined_data()
    alpha_loader= load_alphas(combined_data_df)
    alpha_loader.winsorize_data(winsorization_value) #input is in std deviation
    df_standardized = alpha_loader.standardize_daily()
    # results_df = pd.DataFrame(columns=['Approach', 'Lambda', 'Portfolio Annualized Return', 'Benchmark Annualized Return', 'Portfolio Annualized Excess Return', 'Benchmark Annualized Excess Return', 'Portfolio Volatility', 'Portfolio Sharpe Ratio', 'Portfolio Information Ratio', 'Tracking Error']) 
    # for i in [0.35,0.3,0.25,0.2,0.15,0.1,0.05] :

    constructor = portfolio_construction(percentage_of_stock_holdings
                                         ,factors_in_focus  , df_standardized)
    portfolio_integrated = constructor.integrated_portfolio()
    portfolio_mixing = constructor.mixing_portfolio()
    
    
    # print(""" ###########################{}###########################""".format(i))
    print(""" ########################### Integrated Approach ###########################
              
          """)
    integrated = performance_calculation(combined_data_df ,portfolio_integrated ,constructor.factors  )
    value = integrated.calculate_benchmark_returns()
    value2 = integrated.calculate_portfolio_returns()
    exposures,exposures_series_integrated = integrated.calculate_portfolio_exposures()
    avg_assets = integrated.calculate_average_numberofassets()
    integrated_benchmark_returns, integrated_portfolio_returns = integrated.calculate_cumulative_returns()
    portfolio_annualized_return, benchmark_annualized_return, portfolio_annualized_excess_return, benchmark_annualized_excess_return = integrated.calculate_average_annualized_returns()


    portfolio_volatility_integrated, portfolio_sharpe_integrated,portfolio_information_ratio_integrated, tracking_error_integrated,max_drawdown_integrated = integrated.calculate_all()
#     results_df = results_df.append({
#     'Approach': 'Integrated',
#     'Lambda': i,
#     'Portfolio Annualized Return': portfolio_annualized_return,
#     'Benchmark Annualized Return': benchmark_annualized_return,
#     'Portfolio Annualized Excess Return': portfolio_annualized_excess_return,
#     'Benchmark Annualized Excess Return': benchmark_annualized_excess_return,
#     'Portfolio Volatility': portfolio_volatility_integrated,
#     'Portfolio Sharpe Ratio': portfolio_sharpe_integrated,
#     'Portfolio Information Ratio': portfolio_information_ratio_integrated,
#     'Tracking Error': tracking_error_integrated
# }, ignore_index=True)
    print("Portfolio Annualized Return: " ,portfolio_annualized_return)
    print("Benchmark Annualized Return: ",benchmark_annualized_return) 
    print("Portfolio Annualized Excess Return: ",portfolio_annualized_excess_return) 
    print("Benchmark Annualized Excess Return: ",benchmark_annualized_excess_return) 
    for factor in factors_in_focus:
        print(f"Average of {factor}_sum: {exposures[f'{factor}_sum']}")
    print("Benchmark Cumulative Return: " ,integrated_benchmark_returns["Benchmark Cumulative Return"].iloc[-1] )
    print("Portfolio Cumulative Return: " ,integrated_portfolio_returns["Portfolio Cumulative Return"].iloc[-1] )
    print("Benchmark Excess Cumulative Return: " ,integrated_benchmark_returns["Benchmark Excess Cumulative Return"].iloc[-1] )
    print("Portfolio Excess Cumulative Return: " ,integrated_portfolio_returns["Portfolio Excess Cumulative Return"].iloc[-1] )

    print('Average Number of Assets: ' ,avg_assets )
    print('Portfolio Volatility:', portfolio_volatility_integrated)
    print('Portfolio Sharpe Ratio:', portfolio_sharpe_integrated)
    print('Portfolio Information Ratio:', portfolio_information_ratio_integrated)
    print('Tracking Error:', tracking_error_integrated)
    exposures_series_integrated = exposures_series_integrated.rename(columns = {'Book-to-Price_sum':"Value", 'Historical Beta_sum' :"Min Volatility", 'Relative Strength_sum' :"Momentum",
           'Gross profitability_sum' :"Quality", 'Dividend yield_sum' : "Yield",
           'Earnings per share Growth rate_sum' :"Growth",
           'Logarithm of Market Capitalization_sum' :"Low Size"})
    correlation_matrix(exposures_series_integrated , "Integrated Portfolio Factor Correlation Heatmap")
    
    print(""" ########################### Mixing Approach ###########################
              
          """)
    mixing = performance_calculation(combined_data_df ,portfolio_mixing,constructor.factors )
    value = mixing.calculate_benchmark_returns()
    value2 = mixing.calculate_portfolio_returns()
    exposures , exposures_series_mixing= mixing.calculate_portfolio_exposures()
    mixing_benchmark_returns, mixing_portfolio_returns = mixing.calculate_cumulative_returns()
    avg_assets = mixing.calculate_average_numberofassets()
    portfolio_annualized_return, benchmark_annualized_return, portfolio_annualized_excess_return, benchmark_annualized_excess_return = mixing.calculate_average_annualized_returns()
    portfolio_volatility_mixing, portfolio_sharpe_mixing,portfolio_information_ratio_mixing, tracking_error_mixing,max_drawdown_mixing = mixing.calculate_all()
    print("Portfolio Annualized Return: " ,portfolio_annualized_return)
    print("Benchmark Annualized Return: ",benchmark_annualized_return) 
    print("Portfolio Annualized Excess Return: ",portfolio_annualized_excess_return) 
    print("Benchmark Annualized Excess Return: ",benchmark_annualized_excess_return)
    exposures_series_mixing = exposures_series_mixing.rename(columns = {'Book-to-Price_sum':"Value", 'Historical Beta_sum' :"Min Volatility", 'Relative Strength_sum' :"Momentum",
           'Gross profitability_sum' :"Quality", 'Dividend yield_sum' : "Yield",
           'Earnings per share Growth rate_sum' :"Growth",
           'Logarithm of Market Capitalization_sum' :"Low Size"})
    correlation_matrix(exposures_series_mixing , "Mixing Portfolio Factor Correlation Heatmap")
    for factor in factors_in_focus:
        print(f"Average of {factor}_sum: {exposures[f'{factor}_sum']}")
    print("Benchmark Cumulative Return: " ,mixing_benchmark_returns["Benchmark Cumulative Return"].iloc[-1] )
    print("Portfolio Cumulative Return: " ,mixing_portfolio_returns["Portfolio Cumulative Return"].iloc[-1] )
    print("Benchmark Excess Cumulative Return: " ,mixing_benchmark_returns["Benchmark Excess Cumulative Return"].iloc[-1] )
    print("Portfolio Excess Cumulative Return: " ,mixing_portfolio_returns["Portfolio Excess Cumulative Return"].iloc[-1] )
    print('Average Number of Assets: ' ,avg_assets )
    print('Portfolio Volatility:', portfolio_volatility_mixing)
    print('Portfolio Sharpe Ratio:', portfolio_sharpe_mixing)
    print('Portfolio Information Ratio:', portfolio_information_ratio_mixing)
    print('Tracking Error:', tracking_error_mixing)
    
#         results_df = results_df.append({
#     'Approach': 'Mixing',
#     'Lambda': i,
#     'Portfolio Annualized Return': portfolio_annualized_return,
#     'Benchmark Annualized Return': benchmark_annualized_return,
#     'Portfolio Annualized Excess Return': portfolio_annualized_excess_return,
#     'Benchmark Annualized Excess Return': benchmark_annualized_excess_return,
#     'Portfolio Volatility': portfolio_volatility_mixing,
#     'Portfolio Sharpe Ratio': portfolio_sharpe_mixing,
#     'Portfolio Information Ratio': portfolio_information_ratio_mixing,
#     'Tracking Error': tracking_error_mixing
# }, ignore_index=True)

    plot_cumulative_returns(integrated_portfolio_returns, integrated_benchmark_returns, mixing_portfolio_returns)
    plot_cumulative_excess_returns(integrated_portfolio_returns, integrated_benchmark_returns, mixing_portfolio_returns)
    plot_monthly_returns_scatter(integrated_portfolio_returns['Portfolio Return'], integrated_benchmark_returns['Benchmark Return'], title='Monthly Returns - Integrated Approach')
    plot_monthly_returns_scatter(mixing_portfolio_returns['Portfolio Return'], mixing_benchmark_returns['Benchmark Return'], title='Monthly Returns - mixing Approach')