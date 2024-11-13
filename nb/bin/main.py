import argparse
from nb.nb import NaiveBayes
from nb.utils.load_data import build_dataframe

def main():
    # Parse argumment; we can add an alpha parameter too
    parser = argparse.ArgumentParser(description="Naive Bayes Algorithm")
    parser.add_argument("-f", "--indir", required=True, help="Data directory")
    # I added an alpha parameter
    parser.add_argument("-a", "--alpha", type=float, default=0.5, help="Smoothing parameter (alpha)")
    args = parser.parse_args()

    # Loading data
    training_df, test_df = build_dataframe(args.indir)

    # Initializing and training
    nb_classifier = NaiveBayes(alpha=args.alpha)
    nb_classifier.train_nb(training_df)

    # Using nb classifier 
    class_predictions = nb_classifier.test(test_df)

    # Evaluating classifier
    acc, f1, conf = nb_classifier.evaluate(test_df['author'], class_predictions)
    print("Using the Naive Bayes classifier we created, these are the metrics for our predictions:")
    print({"Accuracy":acc, "F1": f1})

    # Plotting confusion matrix
    NaiveBayes.plot_confusion_matrix(conf, [0, 1])

    # Comparing with sklearn classifier
    sklearn_preds = nb_classifier.sklearn_nb(training_df, test_df)
    sklearn_metrics = nb_classifier.evaluate(test_df['author'], sklearn_preds)
    
    # Evaluating sklearn classifie
    print("\nUsing the Scikit-learn library for Naive Bayes classifier, these are the metrics for the predictions:")
    print({"Accuracy":sklearn_metrics[0], "F1": sklearn_metrics[1]})

    # Plotting sklearn confusion matrix
    NaiveBayes.plot_confusion_matrix(sklearn_metrics[2], [0, 1])

if __name__ == "__main__":
    main()