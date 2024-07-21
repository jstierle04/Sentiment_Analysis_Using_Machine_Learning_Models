import os
os.getcwd()

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
airline = pd.read_csv('Tweets.csv')

import numpy as np

# check number of rows/columns
print(airline.shape, "\n")

# check column data types
print(airline.dtypes, "\n")

# check number of duplicated rows
print(airline[airline.duplicated()], "\n")

# descriptive statistics
print(airline.describe(), "\n")

# percentage of missing values in each column
print(round(airline.isna().sum() / len(airline) * 100, 1), "\n")

# EDA profile

from ydata_profiling import ProfileReport

# Generate the report
profile = ProfileReport(airline, title = "Airline Profile")

# Save the report to .html
profile.to_file("Airline Profile.html")

# Plot of reasons for negative sentiment by airline
plt.figure(figsize=(12, 6))
sns.countplot(data = airline, x = "airline", order = airline['airline'].value_counts().index, hue = 'negativereason')
plt.xlabel('Airline')
plt.ylabel('Frequency')
plt.show()

# airline sentiment by 10 most frequently appearing timezones
plt.figure(figsize=(17, 10))
sns.countplot(data = airline, y = "user_timezone", order = airline['user_timezone'].value_counts().iloc[:10].index, 
              hue = 'airline_sentiment')
plt.rc('xtick', labelsize=21) 
plt.rc('ytick', labelsize=21) 
plt.xlabel('Frequency', fontsize = 22.5)
plt.ylabel('Time Zone', fontsize = 22.5)
plt.title('Frequency of Airline Sentiment by Common Time Zones', fontsize = 22.5)
plt.subplots_adjust(left = 0.3)
plt.show()

# frequency at which each airline is tweeted at
sns.countplot(x = 'airline', data = airline, order = airline['airline'].value_counts().index, hue = 'airline')
plt.xlabel('Airline')
plt.ylabel('Frequency')
plt.title('Frequency of Airlines Appearing in Tweets')
plt.show()

# count missing values
print(airline.loc[:].isnull().sum(), "\n")

# get frequency of each type of sentiment
sums = [0, 0, 0]
i = 0
while i < 14640:
    sentiment = airline.at[i, 'airline_sentiment']

    if sentiment == 'positive':
        sums[0] += 1
    elif sentiment == 'neutral':
        sums[1] += 1
    else:
        sums[2] += 1

    i += 1

print('Positive:', sums[0])
print('Neutral:', sums[1])
print('Negative:', sums[2])