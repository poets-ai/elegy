# Losses and Metrics Guide

This guide goes into depth on how to losses and metrics work in Elegy and how to create your own. One of our goals with Elegy was to solve Keras restrictions around the type of losses and metrics you can define.

When creating a complex model with multiple outputs with Keras, say `output_a` and `output_b`, you are forced to define losses and metrics per-output only:

```python
model.compile(
    loss={
        "output_a": keras.losses.BinaryCrossentropy(from_logits=True),
        "output_b": keras.losses.CategoricalCrossentropy(from_logits=True),
    },
    metrics={
        "output_a": keras.losses.BinaryAccuracy(from_logits=True),
        "output_b": keras.losses.CategoricalAccuracy(from_logits=True),
    },
    ...
)
```
This very restrictive, in particular it doesn't allow the following:
1. Losses and metrics that combine multiple outputs with multiple labels.
2. A especial case of the previous is a single loss based on multiple labels.
3. Losses and metrics cannot depend on the inputs of the model.

Most of these are usually

## 