# MNIST CNN Regression Test

Trained **784→CNN→10** classifier on MNIST using a stack common in Keras/TensorFlow tutorials. Runs as part of `make test` alongside the [MNIST MLP suite](MNIST.md).

## Architecture

| Stage | Config | Activation |
|-------|--------|------------|
| Input | 28×28×1 (NHWC) | — |
| Conv2D | 3×3, 32 filters, stride 1 | ReLU |
| MaxPool2D | 2×2, stride 2 | — |
| Conv2D | 3×3, 64 filters, stride 1 | ReLU |
| MaxPool2D | 2×2, stride 2 | — |
| Flatten | 5×5×64 → 1600 | — |
| Dense | 128 units | ReLU |
| Dense | 10 units | Softmax |

This is the standard “two conv blocks + pooling + dense head” recipe used in introductory MNIST CNN examples (TensorFlow/Keras docs, Colab tutorials, etc.).

## Accuracy vs published results

| Metric | netkit (this export) | Typical tutorial (full 60k train) |
|--------|----------------------|----------------------------------|
| Test accuracy | **97.88%** | ~98.5–99.3% |
| Training set | 15,000 images (subset) | 60,000 images |
| Optimizer | Adam, lr=0.001, 10 epochs | Similar |

netkit’s result is **within ~1–1.5 points** of published tutorial numbers. Training on the full 60k set (`TRAIN_LIMIT = 0` in the export script) typically closes most of that gap and reaches ~99% test accuracy — the engine matches the reference architecture; the gap is mostly training budget.

Recorded in `models/mnist_cnn/training_meta.json`.

## Files

| Path | Purpose |
|------|---------|
| `models/mnist_cnn.json` | Architecture JSON |
| `models/mnist_cnn.bin` | float32 weights (~900 KiB) |
| `models/mnist_cnn/manifest.json` | 10 regression cases |
| `models/mnist_cnn/case_*.input.bin` | Flattened 28×28 NHWC input |
| `models/mnist_cnn/case_*.expected.bin` | 10 softmax outputs |
| `tools/export_mnist_cnn.py` | Train + export script |

## Running

MNIST CNN tests run automatically in `make test` / `./netkit test` — see [TESTING.md](TESTING.md).

```cpp
merge(run_mnist_cnn_tests());  // in src/test.cpp, after MNIST MLP
```

## Regenerating

```bash
make export-mnist-cnn
```

Requires **numpy**. Uses the same MNIST sources as the MLP export (`../python/mnist/*.csv` or downloaded IDX files).

## Engine support added for this test

CNN JSON models may now include:

- `max_pool2d` — 2×2 max pooling (valid, no padding)
- `flatten` — NHWC → 1×features before dense layers
- `dense` — fully connected head (ReLU / softmax activations)

See [MODEL_FORMAT.md](MODEL_FORMAT.md).
