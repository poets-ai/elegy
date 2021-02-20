# pred_step
This method is tasked with taking the input data and calculatin the predictions.

### Inputs
The following inputs are available for `pred_step`:

| name           | types          | description                              |
| :------------- | :------------- | :--------------------------------------- |
| `x`            | `tp.Any`       | Input data                               |
| `states`       | `types.States` | Current state of the model               |
| `initializing` | `bool`         | Whether the model is initializing or not |
| `training`     | `bool`         | Whether the model is training or not     |

### Output
`pred_step` must output a tuple with the following values:

| name     | types          | description                  |
| :------- | :------------- | :--------------------------- |
| `y_pred` | `tp.Any`       | The predictions of the model |
| `states` | `types.States` | The new state of the model   |


### Callers
| method             | when                        |
| :----------------- | :-------------------------- |
| `predict`          | always                      |
| `predict_on_batch` | always                      |
| `test_step`        | default implementation only |
| `summary_step`     | default implementation only |

### Example


### Default Implementation