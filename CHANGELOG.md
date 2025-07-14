# CHANGELOG

## run.py
- Added command line argument support for flexible model selection and mode control:
  - `--model` argument selects between "CNN" and "ViT" architectures (default: "ViT").
  - `--mode` argument toggles between "train" and "test" modes (default: "test").
- Implemented the `write_test_predictions` function:
  - Loads pretrained model checkpoints.
  - Generates predictions on the test dataset.
  - Saves predicted labels to `predictions.txt` within the `outputs` directory.
- Implemented model checkpointing during training:
  - Training loop saves a model checkpoint when the current validation accuracy exceeds the previous best.
  - Ensures the best model (by validation performance) is always saved.

## dataset.py
- Implemented sample shuffling:
  - Added a `shuffle` parameter (default: True) to enable random shuffling of samples at dataset initialization.
  - When enabled, the dataset shuffles the indices of all samples, improving class mixing and reducing ordering bias during training.
- Added data normalization:
  - Integrated feature-wise normalization for all samples when loading the dataset into memory.

## ViT_classifier.py
- Configurable parameters:  
  `img_size`, `patch_size`, `in_chans`, `embed_dim`, `depth`, `num_heads`, `num_classes`
- Added `PatchEmbed` class:
  - Converts input images into a sequence of flattened, low-dimensional patch embeddings suitable for transformer input.
  - Projects each patch to a vector embedding using a Conv2d layer.
  - Uses a 2D convolution with kernel size and stride equal to the patch size for non-overlapping patches.
  - Flattens patches and returns `[B, num_patches, embed_dim]` for transformer processing.
- Added `ViTClassifier` class:
  - Classifies images using a stack of transformer encoder layers.
  - Patch Embedding: Images converted into patch embeddings.
  - `[CLS]` Token & Positional Encoding: Learnable classification token is prepended; positional embeddings are added.
  - Transformer Encoder: Configurable stack of transformer layers processes the sequence.
  - Classification Head: `[CLS]` token output is normalized and fed to a linear layer for final class prediction.

## Visualizations
- Created a `plots` directory to organize and save key data distribution visualizations, including:
  - Comparison of feature-wise mean and standard deviation for train, validation, and test datasets.
  - PCA and t-SNE visualizations of train, validation, and test sets.
  - Example images: one representative sample each from the train and validation sets.

---

### Usage

To train the ViT model:
python run.py --model vit --mode train

To test the model:
python run.py --model vit --mode test

---

