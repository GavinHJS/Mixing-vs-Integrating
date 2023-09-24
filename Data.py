# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 18:58:40 2023

@author: Gavin
"""

import pandas as pd
import sys
import os
import datetime



import pandas as pd
import os

class DataCombiner:
    def __init__(self, directory, folders):
        self.directory = directory
        self.folders = folders
        self.combined_df = pd.DataFrame()

    def load_data(self):
        for number, file in enumerate(self.folders):
            file_path = os.path.join(self.directory, file)
            temp_df = pd.read_csv(file_path, skiprows=12)
            temp_df = temp_df[temp_df["Asset Name"] != "Asset Name"]
            temp_df = temp_df.dropna(subset=['Asset Name'])
            self.combined_df = self.combined_df.append(temp_df, ignore_index=True)
        self.combined_df.rename(columns=lambda x: x.strip(), inplace=True)
        self.combined_df['Analysis Date'] = pd.to_datetime(self.combined_df['Analysis Date'], infer_datetime_format=True)
        numeric_columns = ['Holdings', 'Price', 'Mkt Value',  'Book-to-Price', 'Historical Beta',
                           'Relative Strength', 'Gross profitability', 'Dividend yield', 'Earnings per share Growth rate',
                           'Logarithm of Market Capitalization']
        self.combined_df[numeric_columns] = self.combined_df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    
    def replace_missing_data(self):
        numeric_columns = ['Book-to-Price', 'Historical Beta', 'Relative Strength', 
                           'Gross profitability', 'Dividend yield', 'Earnings per share Growth rate', 
                           'Logarithm of Market Capitalization']
        mean_dict = {}
        for column in numeric_columns:
            mean_by_date_industry = self.combined_df.groupby(['Analysis Date', 'GICS Industry'])[column].mean()
            for (date, industry), mean in mean_by_date_industry.dropna().items():
                if date not in mean_dict:
                    mean_dict[date] = {}
                if industry not in mean_dict[date]:
                    mean_dict[date][industry] = {}
                mean_dict[date][industry][column] = mean
                
        def fill_na(row):
            for column in numeric_columns:
                if pd.isna(row[column]):
                    if row['Analysis Date'] in mean_dict and row['GICS Industry'] in mean_dict[row['Analysis Date']] and column in mean_dict[row['Analysis Date']][row['GICS Industry']]:
                        row[column] = mean_dict[row['Analysis Date']][row['GICS Industry']][column]
            return row
    
        self.combined_df = self.combined_df.apply(fill_na, axis=1)
        # self.combined_df['Relative Strength'] = self.combined_df.groupby('Asset ID')['Relative Strength'].shift(1)
        # self.combined_df['Analysis Date'] = pd.to_datetime(self.combined_df['Analysis Date'])
        
        # earliest_date = self.combined_df['Analysis Date'].min()
        
        # self.combined_df = self.combined_df[self.combined_df['Analysis Date'] != earliest_date]
        # self.combined_df.to_csv("items.csv")

    def count_missing_data(self):
        numeric_columns = ['Book-to-Price', 'Historical Beta', 'Relative Strength', 
                           'Gross profitability', 'Dividend yield', 'Earnings per share Growth rate', 
                           'Logarithm of Market Capitalization']
        missing_counts = {}
        for column in numeric_columns:
            missing_counts[column] = self.combined_df[column].isna().sum()
        missing_data_df = pd.DataFrame(list(missing_counts.items()), columns=['Column', 'Missing Values'])
        return missing_data_df
            
    def get_combined_data(self):
        return self.combined_df



# if __name__ == "__main__":
    
#     directory = "MSCI_WORLD Data"
#     folders =os.listdir(directory)

#     data_combiner = DataCombiner(directory, folders)
#     data_combiner.load_data()
    
#     data_combiner.replace_missing_data()
#     value = (data_combiner.count_missing_data())
#     combined_data_df = data_combiner.get_combined_data()
#     combined_data_df.to_csv("Final_Data.csv",index = False)
