import pandas as pd
from transformers import MobileBertForSequenceClassification, MobileBertTokenizer
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
from transformers import get_linear_schedule_with_warmup, logging
import time
import datetime

path = "McDonald'sReviews_labeled.csv"
df = pd.read_csv(path, encoding="utf-8")

data_X = list(df['review'].values)
labels = df['label'].values

print("*** 데이터 ***")
print("문장"); print(data_X[:5])
print("라벨"); print(labels[:5])

num_to_print = 3
tokenizer = MobileBertTokenizer.from_pretrained('mobilebert-uncased', do_lower_case=True)
inputs = tokenizer(data_X, truncation=True, max_length=256, add_special_tokens=True, padding="max_length")
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']
print("\n\n*** 토큰화 ***")
for j in range(num_to_print):
    print(f"\n{j+1}번째 데이터")
    print("** 토큰 **")
    print(input_ids[j])
    print("** 어텐션 마스크 **")
    print(attention_mask[j])

train, validation, train_y, validation_y = train_test_split(input_ids, labels, test_size=0.1, random_state=2024)
train_masks, validation_masks, _, _ = train_test_split(attention_mask, labels, test_size=0.1, random_state=2024)

batch_size = 8
train_inputs = torch.tensor(train)
train_labels = torch.tensor(train_y)
train_masks = torch.tensor(train_masks)
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_inputs = torch.tensor(validation)
validation_labels = torch.tensor(validation_y)
validation_masks = torch.tensor(validation_masks)
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

model = MobileBertForSequenceClassification.from_pretrained('mobilebert-uncased', num_labels=2)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, eps=1e-8)

epoch = 4
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=len(train_dataloader) * epoch)

for e in range(0, epoch):
    print(f'\n\nEpoch {e+1} / {epoch}')
    print('Training')
    t0 = time.time()
    total_loss = 0
    model.train()

    for step, batch in enumerate(train_dataloader):
        if step % 50 == 0 and not step == 0:
            elapsed_rounded = int(round(time.time() - t0))
            elapsed = str(datetime.timedelta(seconds=elapsed_rounded))
            print(f'- Batch {step} of {len(train_dataloader)}, Elapsed time: {elapsed}')

        batch_ids, batch_mask, batch_labels = tuple(t for t in batch)
        model.zero_grad()

        outputs = model(batch_ids, token_type_ids=None, attention_mask=batch_mask, labels=batch_labels)

        loss = outputs.loss
        total_loss += loss.item()

        if step % 10 == 0 and not step == 0:
            print(f'step : {step}, loss : {loss.item()}')

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_dataloader)
    print(f'Average training loss : {avg_train_loss}')
    train_time_per_epoch = str(datetime.timedelta(seconds=(int(round(time.time() - t0)))))
    print(f'Training time of epoch {e} : {train_time_per_epoch}')

    print('\n\Validation')
    t0 = time.time()
    model.eval()
    eval_loss, eval_accuracy, eval_step, eval_examples = 0, 0, 0, 0
    for batch in validation_dataloader:
        batch_ids, batch_mask, batch_labels = tuple(t for t in batch)

        with torch.no_grad():
            outputs = model(batch_ids, token_type_ids=None, attention_mask=batch_mask)

        logits = outputs[0]
        logits = logits.numpy()
        label_ids = batch_labels.numpy()

        pred_flat = np.argmax(logits, axis=1).flatten()
        labels_flat = label_ids.flatten()
        eval_accuracy_temp = np.sum(pred_flat == labels_flat) / len(labels_flat)
        eval_accuracy += eval_accuracy_temp
        eval_step += 1
    print(f'Validation accuracy : {eval_accuracy / eval_step}')
    val_time_per_epoch = str(datetime.timedelta(seconds=int(round(time.time() - t0))))
    print(f'Validation time of epoch {e} : {val_time_per_epoch}')

print('\nSave Model')
save_path = 'mobilebert_model3'
model.save_pretrained(save_path+'.pt')
print('\nFinish')
