# project570

## Running the Tests

Follow these steps to set up and run the experiments:

Video demo link: https://youtu.be/IPL56dUahoY

### 1. Clone the Repository

```bash
git clone https://github.com/anon1869/project570.git
cd project570
```

### 2. Set Up a Virtual Environment

```bash
# On Windows
python -m venv venv
source venv/Scripts/activate
```

```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install torch
pip install torch-scatter
pip install -r requirements.txt
```

### 4. Modify the Dataset Loader

Open `faknow/data/dataset/multi_modal.py` and replace the `__getitem__` method with the code below. The file can be found at `~\venv\Lib\site-packages\faknow\data\dataset\multi_modal.py`.

```python
def __getitem__(self, index) -> Dict[str, Any]:
    """
    Args:
        index (int): index of item to get

    Returns:
        item (dict): a dict of features of the item
    """
    item = {}
    for feature_name, feature_values in self.data.items():
        if feature_name in self.image_features:
            image_path = os.path.join(self.root, feature_values[index])
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                raise RuntimeError(f"Failed to load image: {image_path}\n{e}")
            value = self.transform(image)

            if not isinstance(value, (torch.Tensor, dict)):
                raise RuntimeError('return type of transform function must be tensor or dict')

        elif feature_name in self.text_features and isinstance(feature_values, dict):
            value = {k: v[index] for k, v in feature_values.items()}
        else:
            value = feature_values[index]

        item[feature_name] = value

    return item
```

Make sure to import the PIL library at the top of the `multi_modal.py` file as well.

```python
from PIL import Image
```

### 5. Run an experiment

```bash
# Options: keyword, random, or human
python main.py keyword
```

To use the `images.py` file, create a .env in the root folder with your pexels API key.
```python
PEXELS_API_KEY=YOUR_PEXELS_API_KEY
```
