from operator import sub
import pandas as pd
import argparse
import wandb
from time import time
from sklearn.pipeline import Pipeline
import config as cfg
from sklearn.model_selection import cross_validate
from pipeline import clean_pipe
from lightgbm import LGBMClassifier
from utils import gini_normalized_scorer, UpsampleStratifiedKFold, log_cv_plot

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset_path",
                    default="dataset/train.csv",
                    help="Path to training dataset")
parser.add_argument("-t", "--test_dataset_path",
                    default="dataset/test.csv",
                    help="Path to test dataset")
parser.add_argument("-s", "--submission",
                    default=False,
                    help="Create submission or not")
parser.add_argument("-f", "--submission_path",
                    default="submission",
                    help="Path to submission")
args = parser.parse_args()


hyperparameter_defaults = dict(
    # model
    num_leaves=31,
    max_depth=-1,
    learning_rate=0.1,
    n_estimators=100,
    min_split_gain=0,
    min_child_weight=0.001,
    min_child_samples=20,
    subsample=1,
    subsample_freq=0,
    colsample_bytree=1,
    reg_alpha=0,
    reg_lambda=0
)

wandb.init(project="porto-seguro", config=hyperparameter_defaults)
config = wandb.config

# Prepare dataset
try:
    train_df = pd.read_csv(args["dataset_path"])
    X = train_df.copy()
    y = X.pop(cfg.TARGET)
except:
    print("Cannot read training dataset.")

# Train model
clf = LGBMClassifier(
    random_state=42,
    # Improve
    num_leaves=config.num_leaves,
    max_depth=config.max_depth,
    learning_rate=config.learning_rate,
    n_estimators=config.n_estimators,
    # Deal with overfit
    min_split_gain=config.min_split_gain,
    min_child_samples=config.min_child_samples,
    min_child_weight=config.min_child_weight,
    subsample=config.subsample,
    subsample_freq=config.subsample_freq,
    colsample_bytree=config.colsample_bytree,
    # Regularization
    reg_alpha=config.reg_alpha,
    reg_lambda=config.reg_lambda
)

model_pipeline = Pipeline([
    ('data', clean_pipe),
    ('clf', clf)
], verbose=True)

scoring = {'gini': gini_normalized_scorer,
           'auc': 'roc_auc'}

upsample_cv = UpsampleStratifiedKFold(n_splits=5)
# Calculate training metrics using cross validation strategy
score = cross_validate(model_pipeline,
                       X=X,
                       y=y,
                       scoring=scoring,
                       cv=upsample_cv,
                       n_jobs=-1,
                       fit_params=None)

# Log cross validation metrics to wandb
cv_metrics = {
    "cv_gini": score['test_gini'].mean(),  # TODO: Improve this
    "cv_auc": score['test_auc'].mean()
}
wandb.log(cv_metrics)
# Log metrics for each fold
metric_names = ['gini', 'auc']
for metric_name in metric_names:
    log_cv_plot(metric_name, score)


if args["submission"]:
    # TODO: INFERENCE
    model_pipeline.fit(X, y)
    try:
        test_df = pd.read_csv(args["test_dataset_path"])
        # test = test_df.sample(n=1000).reset_index().copy()
    except:
        print("Cannot read training dataset.")
    # Make predictions
    predictions = model_pipeline.predict_proba(test_df)

    # Create submission file
    submission_path = args["submission_path"]
    time_stamp = str(int(time()))
    submission_file = f"{submission_path}/{time_stamp}.csv"
    submission = pd.DataFrame(
        {'id': test_df.index, 'target': predictions[:, 1]})
    submission.to_csv(submission_file, index=False)

    # Create a new artifact, which is a sample dataset
    inference = wandb.Artifact(time_stamp, type='inference')
    # Add files to the artifact, in this case a simple text file
    inference.add_file(submission_file)
    # Log the artifact to save it as an output of this run
    wandb.log_artifact(inference)
