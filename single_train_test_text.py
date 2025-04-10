from faknow.data.dataset.text import TextDataset
from faknow.data.process.text_process import TokenizerFromPreTrained
from faknow.evaluate.evaluator import Evaluator
from faknow.model.content_based.mdfend import MDFEND
from faknow.train.trainer import BaseTrainer
import torch
from torch.utils.data import DataLoader

# tokenizer for MDFEND
max_len, bert = 170, 'bert-base-uncased'
tokenizer = TokenizerFromPreTrained(max_len, bert)

# dataset for text dataset only
batch_size = 64
train_path, test_path, validate_path = "./data/train_text.json", "./data/test_text.json", "./data/val_text.json"

train_set = TextDataset(train_path, ['text'], tokenizer)
train_loader = DataLoader(train_set, batch_size, shuffle=True)

validate_set = TextDataset(validate_path, ['text'], tokenizer)
val_loader = DataLoader(validate_set, batch_size, shuffle=False)

test_set = TextDataset(test_path, ['text'], tokenizer)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

# prepare model
domain_num = 4
model = MDFEND(bert, domain_num)

# optimizer and lr scheduler
lr, weight_decay, step_size, gamma = 0.00005, 5e-5, 100, 0.98
optimizer = torch.optim.Adam(params=model.parameters(),
                             lr=lr,
                             weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)

# metrics to evaluate the model performance
evaluator = Evaluator()

# train and validate
num_epochs, device = 50, 'cpu'
trainer = BaseTrainer(model, evaluator, optimizer, scheduler, device=device)
trainer.fit(train_loader, num_epochs, validate_loader=val_loader)

# show test result
print(trainer.evaluate(test_loader))
