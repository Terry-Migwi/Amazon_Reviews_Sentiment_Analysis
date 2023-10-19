## Amazon Reviews Sentiment Analysis 
### Project Overview
This research performs analysis on CDs and Vinyl reviews on Amazon with the objective of identifying the reviewer’s sentiments on the products. This project leverages Natural Language Processing for sentiment analysis by implementing a pre-trained model, machine learning, and deep learning algorithms for classification. 

### Installation and Setup
* Editor: Google Colaboratory
* Python Version: Python 3

### Packages Used
* General purpose packages: itertools
* Data manipulation packages: pyspark, pandas, re, NLTK
* Data Visualization: seaborn, matplotlb
* Machine Learning: sklearn, TensorFlow

### Data
The dataset was obtained from https://jmcauley.ucsd.edu/data/amazon/index_2014.html. The data was downloaded as a gzip file and was loaded into colab using json. The data preprocessing steps involved removing special characters, white spaces, stop words, count vectorization, and Term Frequency -Inverse Document Frequency to create a corpus. 

### Code Structure
This project has four notebooks that have been detailed below:
* ```Sentiment_Analysis_CDs & Vinyl.ipynb``` - this notebook classifies the reviews using the pre-trained VADER sentiment analyzer.
* ```EDA_CDs_Vinyl.ipynb``` - this notebook explores the review data to find the most frequently used words and visualize them in bar charts, word cloud, and bigram networks.
* ```ML_CDs_Vinyl.ipynb ``` - this notebook performs sentiment classification using Logistic Regression, Random Forest, Naive Bayes, and Support Vector Machines algorithms.
* ```NN_CDs_Vinyl.ipynb ``` - this notebook contains the Neural Network Algorithm used to predict the reviews.

### Results and Evaluation
The VADER sentiment analyzer classified the reviews into positive, neutral, and negative, with the positive reviews being the most. The results of the machine learning are summarized in the graph below. 
