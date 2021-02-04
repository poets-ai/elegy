# Changelog

## [0.4.1](https://github.com/poets-ai/elegy/tree/0.4.1) (2021-02-03)

[Full Changelog](https://github.com/poets-ai/elegy/compare/0.4.0...0.4.1)

**Merged pull requests:**

- fix-maybe-initialize [\#155](https://github.com/poets-ai/elegy/pull/155) ([cgarciae](https://github.com/cgarciae))
- Add simple Flax low-level API Model example to README.md [\#153](https://github.com/poets-ai/elegy/pull/153) ([sooheon](https://github.com/sooheon))

## [0.4.0](https://github.com/poets-ai/elegy/tree/0.4.0) (2021-02-01)

[Full Changelog](https://github.com/poets-ai/elegy/compare/0.3.0...0.4.0)

**Implemented enhancements:**

- \[Feature Request\] Monitoring learning rates [\#124](https://github.com/poets-ai/elegy/issues/124)

**Merged pull requests:**

- Update Getting Started + README [\#152](https://github.com/poets-ai/elegy/pull/152) ([cgarciae](https://github.com/cgarciae))
- Pretrained ResNet fix after \#139 [\#151](https://github.com/poets-ai/elegy/pull/151) ([alexander-g](https://github.com/alexander-g))
- Dataset: better default batch\_fn and custom batch\_fn [\#148](https://github.com/poets-ai/elegy/pull/148) ([alexander-g](https://github.com/alexander-g))
- Label Smoothing for Binary Crossentropy [\#146](https://github.com/poets-ai/elegy/pull/146) ([alexander-g](https://github.com/alexander-g))
- Add adapter for handling torch dataloaders [\#145](https://github.com/poets-ai/elegy/pull/145) ([charlielito](https://github.com/charlielito))
- Feature/tf dataset adapter [\#144](https://github.com/poets-ai/elegy/pull/144) ([charlielito](https://github.com/charlielito))
- \[\*.md,\*.py,\*.sh\] Fix typos [\#142](https://github.com/poets-ai/elegy/pull/142) ([SamuelMarks](https://github.com/SamuelMarks))
- verbose=4 [\#140](https://github.com/poets-ai/elegy/pull/140) ([alexander-g](https://github.com/alexander-g))
- Framework Agnostic API: Introduces a new low-level API, removes the dependency between Model and Module, adds support for Flax and Haiku, simplifies hooks. [\#139](https://github.com/poets-ai/elegy/pull/139) ([cgarciae](https://github.com/cgarciae))
- DataLoader Optimizations [\#137](https://github.com/poets-ai/elegy/pull/137) ([alexander-g](https://github.com/alexander-g))
- Autodownload pretrained ResNet [\#136](https://github.com/poets-ai/elegy/pull/136) ([alexander-g](https://github.com/alexander-g))
- Add learning rate logging [\#135](https://github.com/poets-ai/elegy/pull/135) ([cgarciae](https://github.com/cgarciae))
- Adds gitpod support to be able to develop elegy on the cloud [\#134](https://github.com/poets-ai/elegy/pull/134) ([cgarciae](https://github.com/cgarciae))
- Make Models Pickleable Again [\#133](https://github.com/poets-ai/elegy/pull/133) ([alexander-g](https://github.com/alexander-g))
- SCCE fix for bug in Jax\<0.2.7 [\#130](https://github.com/poets-ai/elegy/pull/130) ([alexander-g](https://github.com/alexander-g))
- table progress [\#127](https://github.com/poets-ai/elegy/pull/127) ([alexander-g](https://github.com/alexander-g))

## [0.3.0](https://github.com/poets-ai/elegy/tree/0.3.0) (2020-12-17)

[Full Changelog](https://github.com/poets-ai/elegy/compare/0.2.2...0.3.0)

**Implemented enhancements:**

- elegy.nn.Sequential docs not clear [\#107](https://github.com/poets-ai/elegy/issues/107)
- \[Feature Request\] Community example repo. [\#98](https://github.com/poets-ai/elegy/issues/98)

**Fixed bugs:**

- \[Bug\] Accuracy from Model.evaluate\(\) is inconsistent with manually computed accuracy [\#109](https://github.com/poets-ai/elegy/issues/109)
- Exceptions in "Getting Started" colab notebook [\#104](https://github.com/poets-ai/elegy/issues/104)

**Closed issues:**

- l2\_normalize [\#102](https://github.com/poets-ai/elegy/issues/102)
- Need some help for contributing new losses. [\#93](https://github.com/poets-ai/elegy/issues/93)
- Document Sum [\#62](https://github.com/poets-ai/elegy/issues/62)
- Binary Accuracy Metric [\#58](https://github.com/poets-ai/elegy/issues/58)
- Automate generation of API Reference folder structure [\#19](https://github.com/poets-ai/elegy/issues/19)
- Implement Model.summary [\#3](https://github.com/poets-ai/elegy/issues/3)

**Merged pull requests:**

- `sparse\_categorical\_crossentropy` should check bounds [\#123](https://github.com/poets-ai/elegy/pull/123) ([alexander-g](https://github.com/alexander-g))
- float sample\_weight for precision/recall metrics [\#122](https://github.com/poets-ai/elegy/pull/122) ([alexander-g](https://github.com/alexander-g))
- Added Huber loss [\#121](https://github.com/poets-ai/elegy/pull/121) ([abhinavsp0730](https://github.com/abhinavsp0730))
- ResNet Docs + CIFAR10 Example [\#119](https://github.com/poets-ai/elegy/pull/119) ([alexander-g](https://github.com/alexander-g))
- Dataset & DataLoader [\#118](https://github.com/poets-ai/elegy/pull/118) ([alexander-g](https://github.com/alexander-g))
- fix/docs [\#116](https://github.com/poets-ai/elegy/pull/116) ([cgarciae](https://github.com/cgarciae))
- Better save + load [\#114](https://github.com/poets-ai/elegy/pull/114) ([cgarciae](https://github.com/cgarciae))
- Examples Cleanup [\#113](https://github.com/poets-ai/elegy/pull/113) ([alexander-g](https://github.com/alexander-g))
- merge resnet into master [\#111](https://github.com/poets-ai/elegy/pull/111) ([cgarciae](https://github.com/cgarciae))
- Fix metrics error [\#110](https://github.com/poets-ai/elegy/pull/110) ([cgarciae](https://github.com/cgarciae))
- Fix colab notebook getting started [\#105](https://github.com/poets-ai/elegy/pull/105) ([charlielito](https://github.com/charlielito))
- Added Cosine Similarity loss. [\#103](https://github.com/poets-ai/elegy/pull/103) ([abhinavsp0730](https://github.com/abhinavsp0730))
- small change to trigger build [\#101](https://github.com/poets-ai/elegy/pull/101) ([charlielito](https://github.com/charlielito))
- New metrics [\#100](https://github.com/poets-ai/elegy/pull/100) ([anvelezec](https://github.com/anvelezec))
- Update CONTRIBUTING.md [\#97](https://github.com/poets-ai/elegy/pull/97) ([haruiz](https://github.com/haruiz))
- Enhance docs [\#96](https://github.com/poets-ai/elegy/pull/96) ([charlielito](https://github.com/charlielito))
- Loss Mean Squared Logarithmic error. [\#95](https://github.com/poets-ai/elegy/pull/95) ([abhinavsp0730](https://github.com/abhinavsp0730))
- Documentation improvements [\#94](https://github.com/poets-ai/elegy/pull/94) ([chjort](https://github.com/chjort))
- Module v3 [\#92](https://github.com/poets-ai/elegy/pull/92) ([cgarciae](https://github.com/cgarciae))
- Documentation fixes of module-system.md [\#91](https://github.com/poets-ai/elegy/pull/91) ([chjort](https://github.com/chjort))
- binary precision and recall metrics [\#86](https://github.com/poets-ai/elegy/pull/86) ([anvelezec](https://github.com/anvelezec))

## [0.2.2](https://github.com/poets-ai/elegy/tree/0.2.2) (2020-08-31)

[Full Changelog](https://github.com/poets-ai/elegy/compare/0.2.1...0.2.2)

**Merged pull requests:**

- nn.Embeddings + fix BinaryCrossentropy & BinaryAccuracy [\#90](https://github.com/poets-ai/elegy/pull/90) ([cgarciae](https://github.com/cgarciae))
- test/ci [\#89](https://github.com/poets-ai/elegy/pull/89) ([cgarciae](https://github.com/cgarciae))

## [0.2.1](https://github.com/poets-ai/elegy/tree/0.2.1) (2020-08-25)

[Full Changelog](https://github.com/poets-ai/elegy/compare/0.2.0...0.2.1)

**Merged pull requests:**

- feature/optax [\#88](https://github.com/poets-ai/elegy/pull/88) ([cgarciae](https://github.com/cgarciae))
- feature/binary-accuracy [\#87](https://github.com/poets-ai/elegy/pull/87) ([cgarciae](https://github.com/cgarciae))

## [0.2.0](https://github.com/poets-ai/elegy/tree/0.2.0) (2020-08-17)

[Full Changelog](https://github.com/poets-ai/elegy/compare/0.1.5...0.2.0)

**Merged pull requests:**

- Reference preserving hooks [\#85](https://github.com/poets-ai/elegy/pull/85) ([cgarciae](https://github.com/cgarciae))

## [0.1.5](https://github.com/poets-ai/elegy/tree/0.1.5) (2020-07-28)

[Full Changelog](https://github.com/poets-ai/elegy/compare/0.1.4...0.1.5)

**Implemented enhancements:**

- Change Tensorboard callback behavior to tf2 [\#47](https://github.com/poets-ai/elegy/issues/47)

**Merged pull requests:**

- feature/improve-hooks [\#84](https://github.com/poets-ai/elegy/pull/84) ([cgarciae](https://github.com/cgarciae))
- feature/conv2d [\#83](https://github.com/poets-ai/elegy/pull/83) ([cgarciae](https://github.com/cgarciae))
- feature/rename-call [\#82](https://github.com/poets-ai/elegy/pull/82) ([cgarciae](https://github.com/cgarciae))
- feature/update-defered [\#81](https://github.com/poets-ai/elegy/pull/81) ([cgarciae](https://github.com/cgarciae))
- add coverage [\#80](https://github.com/poets-ai/elegy/pull/80) ([charlielito](https://github.com/charlielito))
- Elegy hooks [\#79](https://github.com/poets-ai/elegy/pull/79) ([cgarciae](https://github.com/cgarciae))
- feature/clean-CI [\#78](https://github.com/poets-ai/elegy/pull/78) ([cgarciae](https://github.com/cgarciae))
- Update issue templates [\#77](https://github.com/poets-ai/elegy/pull/77) ([Davidnet](https://github.com/Davidnet))
- Create CONTRIBUTING.md [\#76](https://github.com/poets-ai/elegy/pull/76) ([Davidnet](https://github.com/Davidnet))
- feature/ci-2 [\#75](https://github.com/poets-ai/elegy/pull/75) ([cgarciae](https://github.com/cgarciae))
- Feature/tensorboard tf2 [\#73](https://github.com/poets-ai/elegy/pull/73) ([Davidnet](https://github.com/Davidnet))
- Mean Absolute Percentage Error Implementation [\#53](https://github.com/poets-ai/elegy/pull/53) ([Ciroye](https://github.com/Ciroye))

## [0.1.4](https://github.com/poets-ai/elegy/tree/0.1.4) (2020-07-24)

[Full Changelog](https://github.com/poets-ai/elegy/compare/0.1.3...0.1.4)

**Implemented enhancements:**

- Tensorboard Callback [\#20](https://github.com/poets-ai/elegy/issues/20)

**Closed issues:**

- Document Loss [\#60](https://github.com/poets-ai/elegy/issues/60)
- Document Metric [\#59](https://github.com/poets-ai/elegy/issues/59)
- Specific Requirements for losses and metrics [\#54](https://github.com/poets-ai/elegy/issues/54)
- Document Metric and Loss "on" parameter [\#50](https://github.com/poets-ai/elegy/issues/50)
- Add how to build the docs instructions [\#49](https://github.com/poets-ai/elegy/issues/49)
- Binary Crossentropy + Accuracy [\#22](https://github.com/poets-ai/elegy/issues/22)

**Merged pull requests:**

- feature/dropout-and-batchnorm [\#70](https://github.com/poets-ai/elegy/pull/70) ([cgarciae](https://github.com/cgarciae))
- fix/generator [\#69](https://github.com/poets-ai/elegy/pull/69) ([cgarciae](https://github.com/cgarciae))
- fix docstrings removing kerasspecific stuff [\#68](https://github.com/poets-ai/elegy/pull/68) ([charlielito](https://github.com/charlielito))
- feature/fix-bce-metric-docs [\#67](https://github.com/poets-ai/elegy/pull/67) ([cgarciae](https://github.com/cgarciae))
- feature/document-metric [\#66](https://github.com/poets-ai/elegy/pull/66) ([cgarciae](https://github.com/cgarciae))
- feature/document-loss [\#64](https://github.com/poets-ai/elegy/pull/64) ([cgarciae](https://github.com/cgarciae))
- feature/document-on [\#63](https://github.com/poets-ai/elegy/pull/63) ([cgarciae](https://github.com/cgarciae))
- Implemented BinaryCrossentropy metric [\#57](https://github.com/poets-ai/elegy/pull/57) ([sebasarango1180](https://github.com/sebasarango1180))

## [0.1.3](https://github.com/poets-ai/elegy/tree/0.1.3) (2020-07-23)

[Full Changelog](https://github.com/poets-ai/elegy/compare/6d5338f35ee5ab24a1edba1f19632b6ec97d54b4...0.1.3)

**Implemented enhancements:**

- Updated docs to allow eventual BibTeX citations for the project [\#55](https://github.com/poets-ai/elegy/pull/55) ([sebasarango1180](https://github.com/sebasarango1180))

**Closed issues:**

- Change favicon in Mkdocs [\#41](https://github.com/poets-ai/elegy/issues/41)
- Callbacks Documentation [\#31](https://github.com/poets-ai/elegy/issues/31)
- Fix Docs [\#28](https://github.com/poets-ai/elegy/issues/28)
- Checkpoint Callback [\#26](https://github.com/poets-ai/elegy/issues/26)
- Make state/params objects public in Model [\#24](https://github.com/poets-ai/elegy/issues/24)
- Add atleast 3 examples [\#23](https://github.com/poets-ai/elegy/issues/23)
- Document Model [\#21](https://github.com/poets-ai/elegy/issues/21)
- Support label smoothing in CategoricalCrossentropy [\#18](https://github.com/poets-ai/elegy/issues/18)
- \[RFC\] How to properly define the model function? [\#17](https://github.com/poets-ai/elegy/issues/17)
- Fix predict\_on\_batch when y is None and unintialized model [\#15](https://github.com/poets-ai/elegy/issues/15)
- Make Model\(..., loss\) Optional [\#13](https://github.com/poets-ai/elegy/issues/13)
- Document Loss.weight [\#12](https://github.com/poets-ai/elegy/issues/12)
- Fix license [\#11](https://github.com/poets-ai/elegy/issues/11)
- Port some Metrics and Losses [\#5](https://github.com/poets-ai/elegy/issues/5)
- Implement Callback API [\#4](https://github.com/poets-ai/elegy/issues/4)
- Finish training loop [\#2](https://github.com/poets-ai/elegy/issues/2)

**Merged pull requests:**

- Readme windows support [\#56](https://github.com/poets-ai/elegy/pull/56) ([anvelezec](https://github.com/anvelezec))
- \[Feat\]: Apply to get the new files that push containers to dockerhub. [\#52](https://github.com/poets-ai/elegy/pull/52) ([Davidnet](https://github.com/Davidnet))
- Feat 22: Binary Crossentropy Loss Implementation [\#51](https://github.com/poets-ai/elegy/pull/51) ([haruiz](https://github.com/haruiz))
- Update modules-losses-metrics.md [\#48](https://github.com/poets-ai/elegy/pull/48) ([srcolinas](https://github.com/srcolinas))
- fix docs and move exmaples [\#45](https://github.com/poets-ai/elegy/pull/45) ([charlielito](https://github.com/charlielito))
- add vae example [\#44](https://github.com/poets-ai/elegy/pull/44) ([charlielito](https://github.com/charlielito))
- feature/guides [\#40](https://github.com/poets-ai/elegy/pull/40) ([cgarciae](https://github.com/cgarciae))
- Create LICENSE [\#39](https://github.com/poets-ai/elegy/pull/39) ([Davidnet](https://github.com/Davidnet))
- Feature/tensorboard callbacks [\#38](https://github.com/poets-ai/elegy/pull/38) ([Davidnet](https://github.com/Davidnet))
- Feature/loss\_mae [\#36](https://github.com/poets-ai/elegy/pull/36) ([sebasarango1180](https://github.com/sebasarango1180))
- Feature/more callbacks [\#35](https://github.com/poets-ai/elegy/pull/35) ([Davidnet](https://github.com/Davidnet))
- Feature/checkpoints callback [\#32](https://github.com/poets-ai/elegy/pull/32) ([charlielito](https://github.com/charlielito))
- remove tf specific doc strings [\#30](https://github.com/poets-ai/elegy/pull/30) ([charlielito](https://github.com/charlielito))
- Feature/model docsv1 [\#27](https://github.com/poets-ai/elegy/pull/27) ([charlielito](https://github.com/charlielito))
- feature/no-keras-losses-metrics-mode [\#25](https://github.com/poets-ai/elegy/pull/25) ([cgarciae](https://github.com/cgarciae))
- Feature/model predict [\#16](https://github.com/poets-ai/elegy/pull/16) ([charlielito](https://github.com/charlielito))
- Feature/callbacks [\#14](https://github.com/poets-ai/elegy/pull/14) ([charlielito](https://github.com/charlielito))
- feature/docs [\#10](https://github.com/poets-ai/elegy/pull/10) ([cgarciae](https://github.com/cgarciae))
- add list adapter and test [\#9](https://github.com/poets-ai/elegy/pull/9) ([charlielito](https://github.com/charlielito))
- feature/handle-array-structures [\#8](https://github.com/poets-ai/elegy/pull/8) ([cgarciae](https://github.com/cgarciae))
- Feature/model fit [\#7](https://github.com/poets-ai/elegy/pull/7) ([charlielito](https://github.com/charlielito))
- feature/metrics [\#6](https://github.com/poets-ai/elegy/pull/6) ([cgarciae](https://github.com/cgarciae))
- \[Feat\] Creating Container pipeline to do automated test. [\#1](https://github.com/poets-ai/elegy/pull/1) ([Davidnet](https://github.com/Davidnet))



\* *This Changelog was automatically generated by [github_changelog_generator](https://github.com/github-changelog-generator/github-changelog-generator)*
