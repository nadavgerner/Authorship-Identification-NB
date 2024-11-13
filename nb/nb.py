import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

class NaiveBayes():
    def __init__(self, alpha):
        """
        Initalizing the NaiveBayes Classifier
        :param alpha: the value of alpha for lidstone smoothing 
        """
        self.alpha = alpha
        # Initalizing following vars to make them accessable 
        self.vocabulary = None
        self.priors = None
        self.likelihoods = None

    def train_nb(self, df):
        """
        Training Naive Bayes classifier with own implementation 
        :param df: the pandas DataFrame that contains the auther and the text 
        """

        # Creating vocabulary
        vocabulary = {word: index for index, word in enumerate(set(' '.join(df['text']).split()))}
        
        n_docs = df.shape[0]
        n_classes = df['author'].nunique()

        # Creating priors using the value_counts
        priors = df['author'].value_counts(normalize= True).to_dict()

        training_matrix = np.zeros(shape=(n_docs, len(vocabulary)))

        for index, doc in enumerate(df['text']):
            for token in doc.split():
                if token in vocabulary: # Should always be true
                    training_matrix[index, vocabulary[token]] += 1
            
        class_word_counts = {author: np.zeros(len(vocabulary)) for author in df['author'].unique()}
        for index, author in enumerate(df['author']):
            class_word_counts[author] += training_matrix[index]

        likelihoods = np.zeros((n_classes, len(vocabulary)))

        class_labels = list(class_word_counts.keys())
        total_words_per_class = {author: class_word_counts[author].sum() for author in class_labels}
        
        for i, author in enumerate(class_word_counts):
            likelihoods[i, :] = (class_word_counts[author] + self.alpha) / (total_words_per_class[author] + self.alpha * len(vocabulary))

        # Updating self
        self.vocabulary = vocabulary
        self.priors = priors
        self.likelihoods = likelihoods

    def sklearn_nb(self, train_df, test_df):
        """
        Using Scikit-learn's MultinomialNB
        :param train_df: The training pandas DataFrame with the 
        """
        vectorizer = CountVectorizer()
        vectorizer.fit(train_df['text'])

        training_data = vectorizer.transform(train_df['text'])
        test_data = vectorizer.transform(test_df['text'])

        nb_classifier = MultinomialNB()
        nb_classifier.fit(training_data, train_df['author'])

        return nb_classifier.predict(test_data)

    def test(self, df):
        """
        Testing the classifier on the pandas DataFrame representing the disputed Federalist
        Papers using the vocabulary, priors, and likelihoods.
        :param: pandas DataFrame
        :return: a numpy array of predictions
        """
        class_predictions = []
        class_labels = sorted(self.priors.keys())

        priors_array = np.array([self.priors[author] for author in class_labels])
        
        for text in df['text']:
            test_vector = np.zeros(shape=(len(self.vocabulary)))
            
            for word in text.split():
                if word in self.vocabulary:
                    test_vector[self.vocabulary[word]] += 1
                     
            log_probs = np.log(priors_array) + np.dot(test_vector, np.log(self.likelihoods.T))

            yhat = np.argmax(log_probs)

            class_predictions.append(yhat)

        return class_predictions
        

    def evaluate(self, true_labels, predictions):
        """
        Evaluate the model's performance
        :param true_labels: array-like object
        :param preds: array-like object
        :return: accuracy, F1 score, and confusion matrix
        """

        accuracy = metrics.accuracy_score(true_labels, predictions)
        f1_score = metrics.f1_score(true_labels, predictions)
        conf_matrix = metrics.confusion_matrix(true_labels, predictions)

        return accuracy, f1_score, conf_matrix

    @staticmethod
    def plot_confusion_matrix(conf_matrix_data, labels):
        """
        Method to plot a confusion matrix
        :param conf_matrix_data: array for the confusion matrix
        :param labels: class labels
        """
        plt.title("Confusion matrix")
        axis = sns.heatmap(conf_matrix_data, annot=True, cmap='Blues')
        axis.set_xticklabels(labels)
        axis.set_yticklabels(labels)
        axis.set(xlabel="Predicted", ylabel="True")
        #plt.savefig("conf.jpg")
        plt.show()
