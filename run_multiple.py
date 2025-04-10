import numpy as np
import torch
from faknow.evaluate.evaluator import Evaluator
from faknow.model.content_based.mdfend import MDFEND
from faknow.train.trainer import BaseTrainer

from faknow.data.dataset.multi_modal import MultiModalDataset
from faknow.data.process.text_process import TokenizerFromPreTrained
from torch.utils.data import DataLoader
from torchvision import transforms


# generate and return data loaders
def get_dataloaders(file_names, tokenizer, transform, batch_size):
    train_path, test_path, val_path = file_names

    train_set = MultiModalDataset(train_path, ['text'], tokenizer, ["image"], transform)
    val_set = MultiModalDataset(val_path, ['text'], tokenizer, ["image"], transform)
    test_set = MultiModalDataset(test_path, ['text'], tokenizer, ["image"], transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# runs experiment (train/test) 5 times and records average and standard deviation of accuracy, precision, recall,
# and f1 score
def run_experiments(
        file_names,
        n_runs=5,
        domain_num=5,
        num_epochs=50,
        device="cpu",
        batch_size=64,
        max_len=170,
        bert_model="bert-base-uncased",
        lr=5e-5,
        weight_decay=5e-5,
        step_size=100,
        gamma=0.98,
):
    tokenizer = TokenizerFromPreTrained(max_len, bert_model)

    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_loader, val_loader, test_loader = get_dataloaders(
        file_names, tokenizer, image_transform, batch_size
    )

    all_results = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    for run in range(n_runs):
        print(f"\n--- Run {run + 1}/{n_runs} ---")

        model = MDFEND(bert_model, domain_num)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        evaluator = Evaluator()

        trainer = BaseTrainer(model, evaluator, optimizer, scheduler, device=device)
        trainer.fit(train_loader, num_epochs, validate_loader=val_loader)

        results = trainer.evaluate(test_loader)

        for metric in all_results:
            all_results[metric].append(results[metric])

    print("\n--- Final Results (Mean ± Std over {} runs) ---".format(n_runs))
    for metric, values in all_results.items():
        mean_val = np.mean(values) * 100
        std_val = np.std(values) * 100
        print(f"{metric.capitalize()}: {mean_val:.2f}% ± {std_val:.2f}%")

    return all_results
