# FANformer: Lightweight Transformer with Linear Attention for Sentiment Analysis

This project implements **FANformer**, a lightweight Transformer-based architecture with **linear attention** for efficient sequence modeling. The model is tested on the IMDb sentiment classification task using Hugging Face datasets and tokenizers.

## 🚀 Features

- Custom linear attention mechanism for faster computation.
- Multi-layer Transformer encoder with residual connections.
- End-to-end training on IMDb with `transformers`, `torch`, and `datasets`.
- Model checkpoint saving and inference demo included.

## 📒 Notebook Highlights

- **LinearAttention**: A custom attention mechanism using ELU kernel for linear complexity.
- **FANformer**: Stacked linear attention layers with classifier head.
- **Training Loop**: Train/validation/test split with evaluation metrics.
- **Inference Function**: Predicts sentiment from raw text input.

## 🛠 Setup

Install the required libraries:
```bash
pip install torch transformers datasets
```

## 📦 Dataset

Uses the IMDb dataset from Hugging Face:
```python
from datasets import load_dataset
dataset = load_dataset("imdb")
```

## 📈 Training

Train the model for 3 epochs on a subset (demo):
```python
python fanformer_train.py  # if converted to script
```

The training loop includes validation and testing, with output like:
```
Epoch 0 | Train Loss: 0.5432 | Val Loss: 0.4351 | Val Acc: 81.23%
...
Test Accuracy: 82.50% | Test Loss: 0.4223
```

## 🧠 Inference

Run predictions with:
```python
predict("This movie was fantastic! The acting blew me away.", model, tokenizer)
```

Returns:
```
"Positive"
```

## 💾 Model Saving

```python
torch.save(model.state_dict(), "best_fanformer.pth")
```

---

## 🔭 Future Improvements

- [ ] **Multi-class Classification Support** for other NLP tasks.
- [ ] **Multi-head Output** for multi-label sentiment analysis.
- [ ] **Visualization** of attention weights and learned embeddings.
- [ ] **Model Quantization** or **ONNX export** for deployment.
- [ ] **Performance Benchmarking** vs. standard Transformers.
- [ ] Add **dropout** and **gradient clipping** for stability.
- [ ] Upgrade to **FAN-2/FAN-3** variants with learnable kernels (paper extensions).
- [ ] Convert into a **Hugging Face-compatible model** for easier integration.

---

## 📚 References

- [FAN: Fast Attention via Positive Orthogonal Random Features](https://arxiv.org/abs/2104.09938)
- [Transformers Library](https://github.com/huggingface/transformers)
- [IMDb Dataset](https://huggingface.co/datasets/imdb)