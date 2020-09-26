# Changelog

## [0.2.2] - 2020-08-31
* Fixes `metrics.BinaryAccuracy` and `losses.BinaryCrossentropy`

## [0.2.1] - 2020-08-25
* Uses `optax` instead of `optix`.
* Implements `BinaryAccuracy`.

* Big refactor. Elegy has its own Module system independent of Haiku and its now incompatible with it. #85
## [0.2.0] - 2020-08-17
* Big refactor. Elegy has its own Module system independent of Haiku and its now incompatible with it. #85

## [0.1.5] - 2020-07-28
* Mean Absolute Percentage Error Implementation @Ciroye
* Adds `elegy.nn.Linear`, `elegy.nn.Conv2D`, `elegy.nn.Flatten`, `elegy.nn.Sequential` @cgarciae
* Add Elegy hooks @cgarciae
* Improves Tensorboard support @Davidnet
* Added coverage metrics to CI @charlielito
  
## [0.1.4] - 2020-07-24
* Adds `elegy.metrics.BinaryCrossentropy` @sebasarango1180
* Adds `elegy.nn.Dropout` and `elegy.nn.BatchNormalization` @cgarciae
* Improves documentation
* Fixes bug that cause error when using `training` via dependency injection on `Model.predict`.

## [0.1.3] - 2020-07-22
* Initial release