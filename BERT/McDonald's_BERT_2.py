import pandas as pd
import numpy as np
from transformers import MobileBertForSequenceClassification, MobileBertTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

model_path = 'mobilebert_model3.pt'
model = MobileBertForSequenceClassification.from_pretrained(model_path)
model.eval()

df = pd.read_csv("McDonald'sReviews_processed.csv", encoding="utf-8")
data_X = list(df['review'].values)
labels = df['label'].values

tokenizer = MobileBertTokenizer.from_pretrained('mobilebert-uncased', do_lower_case=True)
inputs = tokenizer(data_X, truncation=True, max_length=256, add_special_tokens=True, padding="max_length")
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

batch_size = 8
test_inputs = torch.tensor(input_ids)
test_labels = torch.tensor(labels)
test_masks = torch.tensor(attention_mask)
test_dataset = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)

test_loss, test_accuracy, test_steps, test_examples = 0, 0, 0, 0

for batch in test_dataloader:
    batch_ids, batch_masks, batch_labels = tuple(t for t in batch)
    with torch.no_grad():
        outputs = model(batch_ids, token_type_ids=None, attention_mask=batch_masks)

    logits = outputs[0]
    logits = logits.numpy()
    label_ids = batch_labels.numpy()

    pred_flat = np.argmax(logits, axis=1).flatten()
    label_flat = label_ids.flatten()

    test_accuracy_temp = np.sum(pred_flat == label_flat) / len(label_flat)
    test_accuracy += test_accuracy_temp
    test_steps += 1
    print(f"Test step : {test_steps}/{len(test_dataloader)}, Temp Accuracy : {test_accuracy_temp}")

avg_test_accuracy = test_accuracy / test_steps
print(f"Total Accuracy : {avg_test_accuracy}")
