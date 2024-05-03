import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout
from sklearn.metrics import precision_recall_curve, average_precision_score
import shap
from sklearn.inspection import permutation_importance



def preprocess_data(df):
    print("Starting preprocessing...")
    
    # Mapping binary values
    binary_map = {'yes': 1, 'no': 0}
    df['default'] = df['default'].map(binary_map)
    df['housing'] = df['housing'].map(binary_map)
    df['loan'] = df['loan'].map(binary_map)
    df['y'] = df['y'].map(binary_map)

    # Define categorical features to be one-hot encoded
    categorical_features = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
    
    # Features to be retained in the DataFrame (numerical + binary mapped)
    numeric_and_binary_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous', 'default', 'housing', 'loan']

    # Using ColumnTransformer to apply different preprocessing to different columns
    onehot_encoder = OneHotEncoder()
    transformer = ColumnTransformer(
        [("onehot", onehot_encoder, categorical_features)],
        remainder='passthrough'  # apply passthrough to the rest of the columns
    )
    
    # Apply transformation and create a DataFrame with the correct column names
    transformed_data = transformer.fit_transform(df)
    new_columns = list(transformer.named_transformers_['onehot'].get_feature_names_out()) + numeric_and_binary_features + ['y']
    df = pd.DataFrame(transformed_data, columns=new_columns)

    # Ensure only relevant numeric features are scaled
    scaler = StandardScaler()
    df[numeric_and_binary_features] = scaler.fit_transform(df[numeric_and_binary_features])

    print("Data preprocessing completed")
    return df


# Analyze the data
def analyze_data(df):
    print("\nAnalyzing data...")
    correlation_matrix = df.corr()
    print("Correlation matrix:")
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title("Correlation Matrix")
    plt.show()
    
def run_multiple_models(df, create_models_function):
    df = preprocess_data(df)
    analyze_data(df)
    X = df.drop('y', axis=1)
    y = df['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Create model configurations here, where X_train is defined
    models = create_models_function(X_train.shape[1])
    for model, name, is_neural_network in models:
        train_and_evaluate_model(model, X_train, y_train, X_test, y_test, X_train.columns, name, is_neural_network)


def create_neural_network(num_features):
    model = Sequential([
        Input(shape=(num_features,)),  # Use Input layer to specify input shape
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(2, activation='softmax')  # Assuming binary classification
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_models(num_features):
    # Define a simple neural network for binary classification
    def create_neural_network():
        model = Sequential([
            Input(shape=(num_features,)),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(2, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    return [
        #(RandomForestClassifier(n_estimators=100, random_state=42), "Random Forest", False),
        #(GradientBoostingClassifier(n_estimators=100, random_state=42), "Gradient Boosting", False),
        (create_neural_network(), "Simple Neural Network", True)
    ]


def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, feature_names, model_name, is_neural_network=False):
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))

    if is_neural_network:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        y_train_categorical = to_categorical(y_train)
        y_test_categorical = to_categorical(y_test)
        model.fit(X_train, y_train_categorical, class_weight=class_weight_dict, epochs=50, batch_size=32, verbose=1)
        predictions = model.predict(X_test)
        predictions_classes = np.argmax(predictions, axis=1)
        y_test_original = np.argmax(y_test_categorical, axis=1)
        
        feature_importance = model.layers[1].get_weights()[0]
        feature_importance = np.abs(feature_importance).sum(axis=0)
        top_feature_indices = np.argsort(feature_importance)[::-1][:30]

        # Ensure only valid indices are considered
        valid_indices = [i for i in top_feature_indices if i < len(feature_names)]
        top_10_features = [feature_names[i] for i in valid_indices]

        print(f"Top Features for {model_name}:")
        for feature in top_10_features:
            print(feature)
        

    else:
        model.fit(X_train, y_train, class_weight=class_weight_dict)
        predictions_classes = model.predict(X_test)

    y_scores = predictions[:, 1] if is_neural_network else model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test_original, y_scores)
    auc_pr = average_precision_score(y_test_original, y_scores)
    
    

    print(f"\nModel: {model_name}")
    print("Accuracy:", accuracy_score(y_test_original, predictions_classes))
    print("Classification Report:\n", classification_report(y_test_original, predictions_classes))
    print("AUC-PR:", auc_pr)

    plot_confusion_matrix(y_test_original, predictions_classes)
    plot_roc_curve(y_test_original, y_scores, model_name)


def plot_confusion_matrix(y_true, y_pred, classes=None,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        title = 'Normalized confusion matrix' if normalize else 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # If classes are not passed, determine them from the data
    if classes is None:
        classes = np.unique(np.concatenate((y_true, y_pred)))
    else:
        classes = np.asarray(classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def plot_roc_curve(y_true, y_scores, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {model_name}')
    plt.legend(loc="lower right")
    plt.show()
    

