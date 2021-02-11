
# States
`elegy.States` is an immutable `Mapping` that contains all the states needed in `Model`, the low-level API provides a simple state management system by passing the `states` parameter (of type `elegy.States`) to all methods. 

### Basic usage
The most common way to use `States` is via its `update` method you can use to set or update field:
```python
states = states.update(some_field=some_value)
```

You can access a field via index or field access notation:
```python
some_value = states["some_field"]
some_value = states.some_field
```

### Default Implementation
The default implementation uses the following fields:

| name               | description                                                              |
| :----------------- | :----------------------------------------------------------------------- |
| `rng`              | contains an `elegy.RNGSeq` instance you can you to request random state. |
| `net_params`       | the trainable parameters of the model.                                   |
| `net_states`       | the non-trainable parameters of the model.                               |
| `metrics_states`   | the states used to calculate cumulative metrics.                         |
| `optimizer_states` | the states for the optimizer.                                            |