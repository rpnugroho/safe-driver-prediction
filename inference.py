import pandas as pd
import argparse
import wandb
import joblib
from time import time
from sklearn.pipeline import Pipeline
import config as cfg
from sklearn.utils import resample
from pipeline import clean_pipe
from lightgbm import LGBMClassifier

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset_path",
                    default="dataset/train.csv",
                    help="Path to training dataset")
parser.add_argument("-t", "--test_dataset_path",
                    default="dataset/test.csv",
                    help="Path to test dataset")
parser.add_argument("-o", "--output_path",
                    default="output",
                    help="Path to output file")
parser.add_argument("-i", "--inference",
                    default=False,
                    action='store_true',
                    help="Create prediction file")
parser.add_argument("-m", "--model",
                    default=False,
                    action='store_true',
                    help="Save model file")
args, unknown = parser.parse_known_args()


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
    train_df = pd.read_csv(args.dataset_path)
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

# Separate minority and majority classes
not_target = train_df[train_df.target == 0]
yes_target = train_df[train_df.target == 1]

yes_target_up = resample(yes_target,
                         replace=True,  # sample without replacement
                         n_samples=len(not_target),  # match majority n
                         random_state=42)  # reproducible results

# Combine majority and upsampled minority
upsampled = pd.concat([yes_target_up, not_target],
                      ignore_index=True).sample(frac=1)

X = upsampled.copy()
y = X.pop(cfg.TARGET)

model_pipeline = Pipeline([
    ('data', clean_pipe),
    ('clf', clf)
], verbose=True)

model_pipeline.fit(X, y)

if args.inference:
    try:
        test_df = pd.read_csv(args.test_dataset_path)
    except:
        print("Cannot read testing dataset.")
    # Make predictions
    predictions = model_pipeline.predict_proba(test_df)

    # Create submission file
    output_path = args.output_path
    time_stamp = str(int(time()))
    submission_file = f"{output_path}/{time_stamp}.csv"
    submission = pd.DataFrame(
        {'id': test_df.index, 'target': predictions[:, 1]})
    submission.to_csv(submission_file, index=False)


if args.model:
    # Log model_pipeline.pkl
    model_file = f"{output_path}/{time_stamp}.pkl"
    joblib.dump(model_pipeline, model_file)
    # Create a new artifact, which is a sample dataset
    model_pkl = wandb.Artifact(time_stamp, type='model')
    # Add files to the artifact, in this case a simple text file
    model_pkl.add_file(model_file)
    # Log the artifact to save it as an output of this run
    wandb.log_artifact(model_pkl)
