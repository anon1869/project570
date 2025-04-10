from faknow.data.process.text_process import TokenizerFromPreTrained
from faknow.evaluate.evaluator import Evaluator
from faknow.model.content_based.mdfend import MDFEND
from faknow.train.trainer import BaseTrainer
from faknow.data.dataset.multi_modal import MultiModalDataset
from torchvision import transforms

import torch
from torch.utils.data import DataLoader

# if you would like to run a single train/test, run this file

# options for training/testing
keyword_file_names = ["./data/train_keyword.json", "./data/test_keyword.json", "./data/val_keyword.json"]
random_file_names = ["./data/train_random.json", "./data/test_random.json", "./data/val_random.json"]
human_file_names = ["./data/train_human.json", "./data/test_human.json", "./dataval_human.json"]

# simply change these variables to match whichever test you would like run
train_path, test_path, validate_path = keyword_file_names

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# tokenizer for MDFEND
max_len, bert = 170, 'bert-base-uncased'
tokenizer = TokenizerFromPreTrained(max_len, bert)

# dataset
batch_size = 64

train_set = MultiModalDataset(train_path, ['text'], tokenizer, ["image"], image_transform)
train_loader = DataLoader(train_set, batch_size, shuffle=True, )

validate_set = MultiModalDataset(validate_path, ['text'], tokenizer, ["image"], image_transform)
val_loader = DataLoader(validate_set, batch_size, shuffle=False)

test_set = MultiModalDataset(test_path, ['text'], tokenizer, ["image"], image_transform)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

# prepare model
domain_num = 1
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

# Example output:
# {'accuracy': 0.5, 'precision': 0.6, 'recall': 0.5, 'f1': 0.5454545454545454}
