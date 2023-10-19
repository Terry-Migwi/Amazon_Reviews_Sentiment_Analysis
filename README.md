## Amazon Reviews Sentiment Analysis 


### Project Overview
This research performs analysis on CDs and Vinyl reviews on Amazon with the objective of identifying the reviewer’s sentiments on the products. This project leverages Natural Language Processing for sentiment analysis by implementing a pre-trained model, machine learning, and deep learning algorithms for classification. 

### Installation and Setup
clone the repository: 

### Resources Used
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
The VADER sentiment analyzer classified the reviews into positive, neutral, and negative as visualized in the graph below. As shown in the graph, the number of positive reviews on the CDs and Vinyl products is three times more than the negative and neutral reviews. 

![image](https://github.com/Terry-Migwi/Amazon_Reviews_Sentiment_Analysis/assets/65303250/1edec14a-32d0-4777-b9a2-63a8355c09c5)

Further, the machine learning algorithms were compared using the accuracy scores and are visualized in the graph below. Linear models gave a better accuracy as compared to the non-linear models. 

![Evaluation of Model Performance](https://github.com/Terry-Migwi/Amazon_Reviews_Sentiment_Analysis/assets/65303250/af12cdd4-2f55-4121-8a14-91e0ffe56950)


