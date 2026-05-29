import os
from transformers import Trainer,TrainingArguments

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

input_data = os.path.join(BASE_DIR,'Data', 'testing_data.csv')
Clean_data = os.path.join(BASE_DIR,'Data', 'Cleaned_reviews.csv')
model_name = "distilbert-base-multilingual-cased"
train_data = os.path.join(BASE_DIR,'Data', 'processed_data', 'train_data.pt')
test_data = os.path.join(BASE_DIR,'Data', 'processed_data', 'test_data.pt')
registered_model_name = "sentiment_analysis_model"

number_of_labels = 3
training_args = TrainingArguments(
    output_dir = "./results",
    num_train_epochs = 3,
    per_device_train_batch_size = 16,
    per_device_eval_batch_size = 64,
    eval_strategy = "epoch",
    save_strategy = "epoch",
    logging_dir = "./logs",
    logging_steps = 50,
    save_total_limit = 1,
    load_best_model_at_end =  True,
    metric_for_best_model = "accuracy"
)

