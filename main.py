import argparse
from run_multiple import run_experiments

# file map
FILE_MAP = {
    "keyword": ["./data/train_keyword.json", "./data/test_keyword.json", "./data/val_keyword.json"],
    "random": ["./data/train_random.json", "./data/test_random.json", "./data/val_random.json"],
    "human": ["./data/train_human.json", "./data/test_human.json", "./data/val_human.json"]
}


# user has three train/test options: keyword, random, or human
def main():
    parser = argparse.ArgumentParser(description="Run MDFEND experiment on selected image type.")
    parser.add_argument("dataset", choices=FILE_MAP.keys(), help="Type of image dataset to use")

    args = parser.parse_args()
    selected_files = FILE_MAP[args.dataset]

    print(f"Running experiments using the {args.dataset} dataset...")
    run_experiments(selected_files)


if __name__ == "__main__":
    main()


# Example output:
# Accuracy: 60.00% ± 6.32%
# Precision: 62.26% ± 3.03%
# Recall: 83.33% ± 10.54%
# F1: 71.16% ± 5.85%