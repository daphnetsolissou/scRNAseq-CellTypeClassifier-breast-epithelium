import os
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report



def build_final_model(X, y, best_params):
    seed = 42
    final_model = OneVsRestClassifier(
        SVC(
            kernel='rbf',
            gamma=best_params['gamma_svm'],
            C=best_params['c_svm'],
            random_state=seed,
            max_iter=1000
        ),
        n_jobs=-1
    )

    svd = TruncatedSVD(n_components=2, random_state=seed).fit(X)
    X_svd = svd.transform(X)
    final_model.fit(X_svd, y)
    
    return final_model, svd



def test_final_model(final_model, X_test, y_test, svd):
    X_test_svd = svd.transform(X_test)
    predictions = final_model.predict(X_test_svd)

    cm = confusion_matrix(y_test, predictions)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    # Create the directory if it doesn't exist
    directory = 'classification_results'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the plot with unique file names
    file_name = 'confusion_matrix'
    save_path_png = os.path.join(directory, file_name + '.png')
    save_path_pdf = os.path.join(directory, file_name + '.pdf')

    # Check if the file already exists
    counter = 1
    while os.path.exists(save_path_png) or os.path.exists(save_path_pdf):
        file_name += '_' + str(counter)
        save_path_png = os.path.join(directory, file_name + '.png')
        save_path_pdf = os.path.join(directory, file_name + '.pdf')
        counter += 1

    # Save the plot
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(save_path_pdf, dpi=300)
    plt.show()

    # Generate the classification report
    report = classification_report(y_test, predictions)

    print("Classification Report:")
    print(report)
