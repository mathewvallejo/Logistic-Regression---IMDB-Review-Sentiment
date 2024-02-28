# Logistic-Regression---IMDB-Review-Sentiment
Tokenizing and vectorizing text data for binary sentiment predictions

# Assignment 2: Binary and Multiclass Classification with Textual Data
*Mat Vallejo, Bonaventure Dossou, Minjae Kim*  
*February 25, 2024*

## Abstract
This paper explores the capabilities of logistic regression and multilinear regression in predicting outcomes for binary data and multi-class data respectively. The logistic regression model is used to predict positive or negative sentiment from IMDB movie reviews, while the multilinear regression is used to match news clippings to their correct news category from the 20 News dataset. This process involves heavy data preparation and cleaning, along with carefully executed feature selection to maximize predictive results. The models are compared against built- in machine learning models from the Scikit-learn machine learning library to test for robustness. We find strong predictive capabilities from both models that regularly outperform Scikit-learn models with no hyperparameter tuning.

## 1 Introduction
This exploration involves a variety of cascading tasks for both datasets and model implementations. To begin, there is a necessary data preprocessing stage that organizes textual data into a workable format. Notably, the use of the Scikit-learn vectorizer function achieves the goal of indexing word frequency in the 20 News dataset. In contrast, a proprietary word counter is used in the IMDB implementation. Following the preprocessing step is feature selection, done using the union of calculated mutual information in the multilinear regression and using linear regression in the IMDB implementation. This helps us discern which words contribute most heavily to their respective outcomes. Once feature extraction has been accomplished, the models can be implemented using gradient descent to minimize cross-entropy loss, designed to terminate after N number of iterations or once convergence is reached with a validation set. The results are then reported in both numerical and graphical formats.

### 1.1 Data Preprocessing
To begin, we set up a new function for the IMDB dataset that will allow us to download the dataset and organize the reviews by their index, the general sentiment (positive or negative), and the rating given. Following this we organize the reviews into positive and negative groups within larger groups of training and testing data. This reflects the way the data is organized from the download itself with 50% of the data in the train folder and 50% of the data in the test folder. We use a counter to track word count and filter out rare and stopwords. For the 20 News dataset, the vectorizer function is used to calculate word count followed by the TF-IDF of the features, which is a measure of total frequency against inverse document frequency.

## 2 Methods

### 2.1 Linear Regression
We use a Linear Regression function designed to extract features with the greatest contribution to both positive and negative sentiment scores. We fit the training data to the model, make predictions on the test set, and calculate the Mean Squared Error (MSE). We then use the coefficient weights to determine the top 100 features for each class and use a sorting function to report the results. 

The listed terms contributing to positive/negative sentiments largely make intuitive sense such as “great” and “excellent” for the positive sentiment, with terms like “worst” and “waste” weighing highly for negative sentiment.

### 2.2 Multiclass Regression
The features (words) selection process for the 20 News dataset is comprised of the following: deleting all newline characters from strings, removing digits, removing punctuations, splitting split into words to remove spaces, removing stop words, and removing non-English words. Since this process already ensures that we have useful and non-misleading words, we only at a later stage removed rare words (appearing in less than 1% of the entire corpus). This being done, we selected the top 250 of each class and united it to have 1000 features overall. Some of the selected features are the following: [’commander’, ’ripping’, ’lot’, ’aside’, ’gap’, ’groff’, ’express’, ’wait’, ’encouragement’, ’regional’, ’crusade’, ’julio’, ’burnt’, ’number’, ’carcinoma’, ’coca’, ’put’, ’belly’, ’picket’, ’sampler’]. Quite well related to the news domain.
We use a multilinear regression for the multiclass classification task. After calculating mutual information to determine feature importance we implement the model on top selected features. The class is defined using cross entropy as the loss function and a gradient descent loop to maximize accuracy in predictions. The gradient descent loop is coded to terminate after either a maximum number of iterations (5000 in this test) is reached or convergence is met between the training and validation sets such that the calculated loss exceeds that of the prior iteration. We then checked the gradient with a small perturbation test which was calculated to be 5.328255059620041e-13 while the best validation loss from gradient descent was measured at 0.4887632494975844.

## 3 Implementation of Logistic and Multiclass Classifiers
For this task, we implement our defined Logistic Regression model on the IMDB dataset with a sigmoid logistic function, cross-entropy as our cost function, and a gradient descent function. The implementation we conducted was limited to a maximum of 10,000 iterations (1e4 as opposed to the coded default 1e5) to offset long processing times and defaults to a learning rate of 0.1. We test our results against a variety of sklearn models with no hyperparameter tuning.

Model | AUROC
--- | ---
Our Logistic Regression | 0.93
Sklearn Logistic Regression | 0.92
Sklearn Decision Tree | 0.67
Sklearn KNN | 0.60

We use a multilinear regression for the multiclass classification task. After calculating mutual information to determine feature importance we implement the model on top selected features. The class is defined using cross entropy as the loss function and a gradient descent loop to maximize accuracy in predictions. The gradient descent loop is coded to terminate after either a maximum number of iterations (5000 in this test) is reached or convergence is met between the training and validation sets such that the calculated loss exceeds that of the prior iteration. We then checked the gradient with a small perturbation test which was calculated to be 5.328255059620041e-13 while the best validation loss from gradient descent was measured at 0.4887632494975844.

### 3.1 Small Perturbation Test
We check our logistic regression gradient descent with a small perturbation test. We can see the results of the relative error are in line with what we would expect (a very small number, in this case, 4.752217356242678e-14) where the norm of gradient descent at the final iteration was 4.159e-03.
#### Logistic Regression
Test Type | Result
--- | ---
Analytical Gradient | -520681.094165622
Approximated Gradient | -520681.0823179607
Relative Error | 1.2943796138003165e-16
Gradient Descent Norm (Final Iteration) | 0.004159

## 4 Run Experiments
### Method comparison using ROC curve

For our logistic regression, we determine the top 10 features contributing to both positive and negative sentiment, similar to the linear regression above. However, we return features with much more perceived correlation to their respective sentiments and less noise than in the simple linear regression implementation.

### 4.1 Logistic Regression Feature Importance
#### 4.1.1 Top 10 features contributing to positive and negative reviews:
See .pdf report for figures.

## 5 Results
See .pdf for full results section.

## 6 Conclusion
In conclusion, we find the logistic regression model implementation to be very robust in predicting sentiment for the IMDB movie review data. The experimentation shows the benefit of strong data cleaning and preprocessing practices, as well as the effects of hyperparameters like feature selection, maximum iterations in gradient descent, and learning rate that contribute to accuracy and/or overfitting. The multilinear regression model was also highly successful in its multi-class classification objective. It highly outperformed the Scikit-learn decision tree model (with no hyperparameter tuning), and benefitted strongly from data preprocessing and feature selection, including the removal of noisy data such as punctuation.

## 7 Statement of Contribution
Logistic Regression, Multiclass Regression, Report - Mat Vallejo | Multiclass Regression, Report - Bonaventure Dossou |
Review and Report - Minjae Kim

