# Default Implementation

### Methods
The default implementation favors composition by implementing a method in term of another, especifically if follows this call graph:

```
     summary            predict         evalutate                            fit              init
        ⬇️                  ⬇️                ⬇️                                 ⬇️                 ⬇️
call_summary_step    call_pred_step   call_test_step                   call_train_step   call_init_step
        ⬇️                  ⬇️                ⬇️                                 ⬇️                 ⬇️
 summary_step    ➡️     pred_step    ⬅   test_step    ⬅   grad_step   ⬅   train_step    ⬅    init_step
```
This structure allows you to for example override `test_step` and still be able to use use `fit` since `train_step` (called by `fit`) will call your `test_step` via `grad_step`. It also means that if you implement `test_step` but not `pred_step` there is a high chance both `predict` and `summary` will not work.

##### call_* methods
The `call_<method>` method family are _entrypoints_ that usually just redirect to their inputs to `<method>`, you choose to override these if you need to perform some some computation only when method in question is the entry point. For example if you want to change the behavior of `evaluate` without affecting the behavior of `fit` while preserving most of the default implementation you can override `call_step_step` to do the corresponding adjustments and then call `test_step`. Since `train_step` does not depend on `call_step_step` then the change will  manifest during `evaluate` but not during `fit`.