#EDA
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


#Text pre-processing

import re
import string
import nltk

from nltk.corpus import stopwords   
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
stopWords = set(stopwords.words('english'))

white_list = ["not", "no", "won't", "isn't", "couldn't", "wasn't", "didn't", "shouldn't", 
"hasn't", "wouldn't", "haven't", "weren't", "hadn't", "shan't", "doesn't",
"mightn't", "mustn't", "needn't", "don't", "aren't", "won't"]

from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer 
from nltk import pos_tag

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
def lemmatize_text(text):
    
    lmtzr = WordNetLemmatizer().lemmatize
    text = word_tokenize(str(text))   # Init the Wordnet Lemmatizer    
    word_pos = pos_tag(text)    
    lemm_words = [lmtzr(sw[0], get_wordnet_pos(sw[1])) for sw in word_pos]
    return (' '.join(lemm_words))

airline = pd.read_csv('Tweets.csv')

#function to process the text of each tweet
def column_nlp(df):
    new_column = []

    for index, row in df.iterrows():

        #make sure text is a string
        row_text = str(row['text'])
        
        #make all text lowercase
        row_text.lower()

        #remove tagged airline
        row_text = re.sub(r'@[A-Za-z0-9]+', '', row_text)

        #remove URLs
        row_text = re.sub(r'https?://[A-Za-z0-9./]+', '', row_text)

        #remove stop words
        words = row_text.split()
        row_text = ' '.join([t for t in words if (t not in stopwords.words('english') or t in white_list)]) 

        #remove punctuation
        row_text = ''.join([t for t in row_text if t not in string.punctuation])

        #remove numeric numbers
        row_text = ''.join([t for t in row_text if not t.isdigit()])

        #lemmatize text
        row_text = lemmatize_text(row_text)

        new_column.append(row_text)

    df.insert(15, 'processed_text', new_column, allow_duplicates = True)

column_nlp(airline)

airline.to_csv('JamesStierle_workshop3.csv')

#remove emojis
def remove_emojis(data):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                    "]+", re.UNICODE)
    return re.sub(emoj, '', data)
        
airline = pd.read_csv("JamesStierle_workshop3.csv")
for text in airline['processed_text']:
    airline.replace(text, remove_emojis(str(text)))

airline.to_csv('JamesStierle_workshop3.csv')

#TF-IDF

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

vector = CountVectorizer(analyzer = 'word', ngram_range=(1, 3)) 

corpus = pd.read_csv("JamesStierle_workshop3.csv")

vectorizer = TfidfVectorizer() 
matrix = vectorizer.fit_transform(corpus['processed_text'].fillna(''))
tfidf_df = pd.DataFrame(matrix.toarray(), columns = vectorizer.get_feature_names_out())

print(tfidf_df, "\n")

output_shapes = {}
for n in range(1, 11):
    vectorizer = TfidfVectorizer(ngram_range = (1, n))
    matrix = vectorizer.fit_transform(corpus['processed_text'].fillna(''))

    output_shapes[n] = matrix.shape

print('Output shapes for different n-grams: \n')
for num, dimensions in output_shapes.items():
    print('n-gram range=(1, %d): %s' % (num, str(dimensions)))

#Data balancing

airline = pd.read_csv('JamesStierle_workshop3.csv')
airline_sub = airline.loc[:, ['tweet_id', 'airline_sentiment', 'text', 'processed_text']]
airline_sub['label'] = airline_sub['airline_sentiment'].map({'negative': -1, 'neutral': 0, 'positive': 1})
airline_sub.to_csv('JamesStierle_workshop5.csv')

sns.countplot(x = 'label', data = airline_sub, order = airline_sub['label'].value_counts().index, hue = 'label')
plt.xlabel('Sentiment')
plt.ylabel('Frequency')
plt.show()

print(airline_sub['label'].value_counts())

import random
from sklearn.model_selection import train_test_split

random.seed(1234567)

train_X, test_X, train_y, test_y = train_test_split(airline_sub, airline_sub['label'], test_size = 0.2, random_state = 101)

print('Entire dataset', airline_sub.shape)
print('Train dataset', train_X.shape)
print('Test dataset', test_X.shape)

sns.countplot(x = 'label', data = train_X, order = train_X['label'].value_counts().index, hue="label")
plt.xlabel('Sentiment')
plt.ylabel('Frequency')
plt.show()

print(train_X['label'].value_counts())

from sklearn.utils import resample
from sklearn.utils import shuffle

def oversampling(train_X):
    df_major_neg = train_X[train_X['label'] == -1]
    df_minor_neu = train_X[train_X['label'] == 0]
    df_minor_pos = train_X[train_X['label'] == 1]        
    major_count = len(df_major_neg)
 
    # oversample minority class
    df_minor_neu_oversampled = resample(df_minor_neu, 
                                 replace = True,              # sample with replacement
                                 n_samples = major_count,     # to match majority class 
                                 random_state = 1000)    

    df_minor_pos_oversampled = resample(df_minor_pos, 
                                 replace = True,             
                                 n_samples = major_count,   
                                 random_state = 1000)      
         
    trainX = pd.concat([df_major_neg, df_minor_neu_oversampled, df_minor_pos_oversampled])   # Combine majority class with oversampled minority class
    print("Train dataset class distribution: \n", trainX.label.value_counts())
    trainX = shuffle(trainX, random_state = 200) 
    return trainX

def undersampling(train_X):
    df_major_neg = train_X[train_X['label'] == -1]
    df_minor_neu = train_X[train_X['label'] == 0]
    df_minor_pos = train_X[train_X['label'] == 1]  
    minor_count = len(df_minor_pos)

    df_minor_neu_undersampled = resample(df_minor_neu, replace = True,
                                         n_samples = minor_count, random_state=1000)
    
    df_major_neg_undersampled = resample(df_major_neg, replace = True,
                                         n_samples=minor_count, random_state=1000)
    
    trainX = pd.concat([df_minor_pos, df_minor_neu_undersampled, df_major_neg_undersampled])
    print("Train dataset class distribution: \n", trainX.label.value_counts())
    trainX = shuffle(trainX, random_state = 200) 
    return trainX

train_X = undersampling(train_X)

# Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, classification_report,confusion_matrix, accuracy_score

def airline_NB(df, feature, ngram, sample_method):    
    random.seed(1234567)
    
    if feature == "TF":
        vector = CountVectorizer(analyzer = 'word', ngram_range=(1, ngram)) 
    elif feature == "TFIDF":        
        vector = TfidfVectorizer(ngram_range=(1,ngram))
        
    train_X, test_X, train_y, test_y = train_test_split(df, df['label'], test_size = 0.2, random_state = 101)
        
    if sample_method == "undersampling":
        train_X = undersampling(train_X)
    
    elif sample_method == "oversampling":    
        train_X = oversampling(train_X)   
              
    pipe = make_pipeline(vector, MultinomialNB(alpha = 1.0, fit_prior = True))
    clf = pipe.fit(train_X['processed_text'].fillna('').values.astype('U'), train_X['label'])     
    
    test_y_hat = pipe.predict(test_X['processed_text'].fillna('').values.astype('U'))
       
    df_result = test_X.copy()
    df_result['prediction'] = test_y_hat.tolist()   
    
    df_prob = pd.DataFrame(pipe.predict_proba(test_X['processed_text'].fillna('').values.astype('U')), columns = pipe.classes_)
    df_prob.index = df_result.index
    df_prob.columns = ['probability_negative', 'Probability_neutral', 'probability_positive']

    df_final = pd.concat([df_result, df_prob], axis = 1)
    
    file_name = 'NB_' + str(ngram) + '_' + sample_method 
    df_final.to_csv(file_name + '.csv')       
    
    print("-----------------------------------------")
    print("NB classification report -- ", "feature: %s/" %feature, "ngram: %d/" %ngram, "sample_method: %s/" %sample_method)
    print(pd.crosstab(test_y.to_numpy(), test_y_hat, rownames = ['True'], colnames = ['Predicted'], margins = True))      

    print("-----------------------------------------")
    print(classification_report(test_y, test_y_hat))    
    print('Macro F1 Score: {:.2f}'.format(f1_score(test_y_hat, test_y, average = 'macro')))  
    print('Weighted F1 Score: {:.2f}'.format(f1_score(test_y_hat, test_y, average = 'weighted')))     


 # Naive Bayes. Arguments: dataframe, TF/IFIDF, unigran or ngram, data-balancing method   
airline_NB(airline_sub, "TF", 1, "none")  
airline_NB(airline_sub, "TF", 1, "oversampling")  
airline_NB(airline_sub, "TF", 1, "undersampling")  

airline_NB(airline_sub, "TFIDF", 1, "none")  
airline_NB(airline_sub, "TFIDF", 1, "oversampling")  
airline_NB(airline_sub, "TFIDF", 1, "undersampling")  

airline_NB(airline_sub, "TF", 2, "none")  
airline_NB(airline_sub, "TF", 2, "oversampling")  
airline_NB(airline_sub, "TF", 2, "undersampling")  

airline_NB(airline_sub, "TFIDF", 2, "none")  
airline_NB(airline_sub, "TFIDF", 2, "oversampling")  
airline_NB(airline_sub, "TFIDF", 2, "undersampling")  
            
airline_NB(airline_sub, "TF", 3, "none")  
airline_NB(airline_sub, "TF", 3, "oversampling")  
airline_NB(airline_sub, "TF", 3, "undersampling") 

airline_NB(airline_sub, "TFIDF", 3, "none")
airline_NB(airline_sub, "TFIDF", 3, "oversampling")
airline_NB(airline_sub, "TFIDF", 3, "undersampling")

#Random forest
from sklearn.ensemble import RandomForestClassifier

def airline_RF(df, feature, ngram, sample_method):    
    random.seed(1234567)
    
    if feature == "TF":
        vector = CountVectorizer(analyzer = 'word', ngram_range=(1, ngram)) 
    elif feature == "TFIDF":        
        vector = TfidfVectorizer(ngram_range=(1,ngram))
        
    train_X, test_X, train_y, test_y = train_test_split(df, df['label'], test_size = 0.2, random_state = 101)
        
    if sample_method == "undersampling":
        train_X = undersampling(train_X)
    
    elif sample_method == "oversampling":    
        train_X = oversampling(train_X)   
              
    pipe = make_pipeline(vector, RandomForestClassifier())
    clf = pipe.fit(train_X['processed_text'].fillna('').values.astype('U'), train_X['label'])     
    
    test_y_hat = pipe.predict(test_X['processed_text'].fillna('').values.astype('U'))
       
    df_result = test_X.copy()
    df_result['prediction'] = test_y_hat.tolist()   
    
    df_prob = pd.DataFrame(pipe.predict_proba(test_X['processed_text'].fillna('').values.astype('U')), columns = pipe.classes_)
    df_prob.index = df_result.index
    df_prob.columns = ['probability_negative', 'Probability_neutral', 'probability_positive']

    df_final = pd.concat([df_result, df_prob], axis = 1)
    
    file_name = 'RF_' + str(ngram) + '_' + sample_method 
    df_final.to_csv(file_name + '.csv')       
    
    print("-----------------------------------------")
    print("RF classification report -- ", "feature: %s/" %feature, "ngram: %d/" %ngram, "sample_method: %s/" %sample_method)
    print(pd.crosstab(test_y.to_numpy(), test_y_hat, rownames = ['True'], colnames = ['Predicted'], margins = True))      

    print("-----------------------------------------")
    print(classification_report(test_y, test_y_hat))    
    print('Macro F1 Score: {:.2f}'.format(f1_score(test_y_hat, test_y, average = 'macro')))  
    print('Weighted F1 Score: {:.2f}'.format(f1_score(test_y_hat, test_y, average = 'weighted')))     


 #Random Forest. Arguments: dataframe, TF/IFIDF, unigran or ngram, data-balancing method   
airline_RF(airline_sub, "TF", 1, "none")  
airline_RF(airline_sub, "TF", 1, "oversampling")  
airline_RF(airline_sub, "TF", 1, "undersampling")  

airline_RF(airline_sub, "TFIDF", 1, "none")  
airline_RF(airline_sub, "TFIDF", 1, "oversampling")  
airline_RF(airline_sub, "TFIDF", 1, "undersampling") 

airline_RF(airline_sub, "TF", 2, "none")  
airline_RF(airline_sub, "TF", 2, "oversampling")  
airline_RF(airline_sub, "TF", 2, "undersampling")  

airline_RF(airline_sub, "TFIDF", 2, "none")  
airline_RF(airline_sub, "TFIDF", 2, "oversampling")  
airline_RF(airline_sub, "TFIDF", 2, "undersampling") 

airline_RF(airline_sub, "TF", 3, "none")  
airline_RF(airline_sub, "TF", 3, "oversampling")  
airline_RF(airline_sub, "TF", 3, "undersampling") 
            
airline_RF(airline_sub, "TFIDF", 3, "none")
airline_RF(airline_sub, "TFIDF", 3, "oversampling")
airline_RF(airline_sub, "TFIDF", 3, "undersampling")

#SVM
from sklearn.svm import SVC

def airline_SVM(df, feature, ngram, sample_method):    
    random.seed(1234567)
    
    if feature == "TF":
        vector = CountVectorizer(analyzer = 'word', ngram_range=(1, ngram)) 
    elif feature == "TFIDF":        
        vector = TfidfVectorizer(ngram_range=(1,ngram))
        
    train_X, test_X, train_y, test_y = train_test_split(df, df['label'], test_size = 0.2, random_state = 101)
        
    if sample_method == "undersampling":
        train_X = undersampling(train_X)
    
    elif sample_method == "oversampling":    
        train_X = oversampling(train_X)   
              
    pipe = make_pipeline(vector, SVC(probability=True))
    clf = pipe.fit(train_X['processed_text'].fillna('').values.astype('U'), train_X['label'])     
    
    test_y_hat = pipe.predict(test_X['processed_text'].fillna('').values.astype('U'))
       
    df_result = test_X.copy()
    df_result['prediction'] = test_y_hat.tolist()   
    
    df_prob = pd.DataFrame(pipe.predict_proba(test_X['processed_text'].fillna('').values.astype('U')), columns = pipe.classes_)
    df_prob.index = df_result.index
    df_prob.columns = ['probability_negative', 'Probability_neutral', 'probability_positive']

    df_final = pd.concat([df_result, df_prob], axis = 1)
    
    file_name = 'SVM_' + str(ngram) + '_' + sample_method 
    df_final.to_csv(file_name + '.csv')       
    
    print("-----------------------------------------")
    print("SVM classification report -- ", "feature: %s/" %feature, "ngram: %d/" %ngram, "sample_method: %s/" %sample_method)
    print(pd.crosstab(test_y.to_numpy(), test_y_hat, rownames = ['True'], colnames = ['Predicted'], margins = True))      

    print("-----------------------------------------")
    print(classification_report(test_y, test_y_hat))    
    print('Macro F1 Score: {:.2f}'.format(f1_score(test_y_hat, test_y, average = 'macro')))  
    print('Weighted F1 Score: {:.2f}'.format(f1_score(test_y_hat, test_y, average = 'weighted')))     


 #SVM. Arguments: dataframe, TF/IFIDF, unigran or ngram, data-balancing method   
airline_SVM(airline_sub, "TF", 1, "none")  
airline_SVM(airline_sub, "TF", 1, "oversampling")  
airline_SVM(airline_sub, "TF", 1, "undersampling")

airline_SVM(airline_sub, "TFIDF", 1, "none")  
airline_SVM(airline_sub, "TFIDF", 1, "oversampling")  
airline_SVM(airline_sub, "TFIDF", 1, "undersampling") 

airline_SVM(airline_sub, "TF", 2, "none")  
airline_SVM(airline_sub, "TF", 2, "oversampling")  
airline_SVM(airline_sub, "TF", 2, "undersampling")

airline_SVM(airline_sub, "TFIDF", 2, "none")  
airline_SVM(airline_sub, "TFIDF", 2, "oversampling")  
airline_SVM(airline_sub, "TFIDF", 2, "undersampling")  

airline_SVM(airline_sub, "TF", 3, "none")  
airline_SVM(airline_sub, "TF", 3, "oversampling")  
airline_SVM(airline_sub, "TF", 3, "undersampling")
            
airline_SVM(airline_sub, "TFIDF", 3, "none")
airline_SVM(airline_sub, "TFIDF", 3, "oversampling")
airline_SVM(airline_sub, "TFIDF", 3, "undersampling") 

