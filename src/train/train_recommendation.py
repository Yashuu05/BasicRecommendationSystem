import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.pipelines.model_pipelines import prepare_model_pipeline
from src.utils.data_utils import load_data, save_data, prepare_data_for_split, split_dataset
from src.utils.model_utils import save_model, best_model, evaluate_classification_model

def train_model(model_lst, X_train, y_train, X_test, y_test):
    # dictionary to store peformance of each model
    classification_performance = {
    "model_name": [],
    "accuracy": [],
    "f1_score": [],
    "recall_score": [],
    "roc_auc_score":[]
    }

    plt.figure(figsize=(8, 6))
    
    # train each model from the "model_lst"
    for name, model in model_lst.items():
        print(f"training {name}...")
        # fit the model
        model.fit(X_train, y_train)
        # predict target
        
        y_pred = model.predict(X_test)
        # evaluate performance
        acc, f1, recall, roc = evaluate_classification_model(model=model, X_test=X_test, y_test=y_test)
        # store the metrics into performance list
        classification_performance["model_name"].append(name)
        classification_performance["accuracy"].append(acc)
        classification_performance["f1_score"].append(f1)
        classification_performance["recall_score"].append(recall)
        classification_performance["roc_auc_score"].append(roc)
        # update model_lst with trained models
        model_lst[name] = model
        
    # file saving path
    image_file_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'results', 'recommendation_performance.png')
    result_file_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'results', 'recommendation_model_performance.csv')
    
    # save the performance od trained model
    print("\nsaving performace data...")
    classification_performance_df = pd.DataFrame(classification_performance, columns=['model_name','accuracy','f1_score','recall_score','roc_auc_score'])
    classification_performance_df.to_csv(result_file_path, index=False)

    # plot the graph and save
    df_melted = classification_performance_df.melt(id_vars="model_name", 
                           var_name="Metric", 
                           value_name="Score")
    plt.figure(figsize=(12, 6))

    sns.barplot(data=df_melted, x="model_name", y="Score", hue="Metric")
    plt.title("Model Performance Comparison")
    plt.xticks(rotation=30)
    plt.legend(title="Metrics")
    plt.tight_layout()
    # plt.show()
    os.makedirs(os.path.dirname(image_file_path), exist_ok=True)
    plt.savefig(image_file_path, dpi=300)
    plt.close()
    # find best model and score
    print("\nfinding best model...")
    SCORE, NAME, MODEL = best_model(model_lst=model_lst, peformance_lst=classification_performance_df, metric="roc_auc_score", type="classification")
    print(f"\nbest model score:{SCORE}\nmodel name: {NAME}")
    # save only best model
    print("saving best model")
    save_model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'Recommendation', 'best_model.pkl')
    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
    # save model
    save_model(model=MODEL, path=save_model_path)
    # end of training
    print("\n============= Finished training =============")

if __name__ == "__main__":
    # load dataset
    file_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'clean_recommendation_dataset.csv')
    df = load_data(path=file_path)
    # create "X" and "y" 
    X, y = prepare_data_for_split(data=df, target="selected")
    # split dataset
    X_train, X_test, y_train, y_test = split_dataset(X=X, y=y, random_state=42, test_split_size=0.22)
    # ordinal cols
    ordinal_cols = ['severity','urgent']
    # non_ordinal
    non_ordinal_cols = ['issue','device','service_type','availability']
    # num cols
    num_cols = ['rating','num_reviews','success_rate','experience_years', 'distance_km','response_time_min', 'base_price']
    # build pipeline
    model_lst = prepare_model_pipeline(
        non_ordinal_cols=non_ordinal_cols,
        ordinal_cols=ordinal_cols,
        num_cols=num_cols,
        type="classification",
        scoring_param="roc_auc")
    # train model
    train_model(model_lst=model_lst, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
