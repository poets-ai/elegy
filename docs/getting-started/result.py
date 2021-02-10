#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/poets-ai/elegy/blob/master/docs/getting-started-low-level-api.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Getting Started: Low Level API

# Elegy's low-level API works similar to the [Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) API or [Keras Custom Training](https://keras.io/guides/customizing_what_happens_in_fit/) and has the following features:
# 
# * Its functional: it uses a simple functional state management pattern so it should be compatible with all jax features.
# * Its framework agnostic: since its just Jax you can use any Jax framework like Flax, Haiku, Objax, etc, or even just use pure Jax.
# * Its very flexible: you get to define everything from how predictions, losses, metrics, gradients, parameters update, etc, are calculated, this is more ideal for research or less standar training procedures like Adversarial Training.
# 
# Before getting started lets install some dependencies and define some styles for the notebook:

# In[ ]:


#get_ipython().system(' pip install --upgrade pip')
#get_ipython().system(' pip install elegy dataget matplotlib')
# For GPU install proper version of your CUDA, following will work in COLAB:
# ! pip install --upgrade jax jaxlib==0.1.59+cuda101 -f https://storage.googleapis.com/jax-releases/jax_releases.html


# In[7]:


#get_ipython().run_cell_magic('html', '', '<style>\n  table {margin-left: 0 !important;}\n</style>')


# **Note:** that Elegy depends on the jax CPU version hosted on Pypi, if you want to run jax on GPU you will need to [install it](https://github.com/google/jax#installation) separately. If you are running this example on Colab, jax is already preinstalled but you can uncomment the last line of the previous cell if you want to update it.
# 
# ## Loading the Data
# In this tutorial we will train a Neural Network on the MNIST dataset, for this we will first need to download and load the data into memory. Here we will use `dataget` for simplicity but you can use you favorite datasets library.

# In[1]:


import dataget

X_train, y_train, X_test, y_test = dataget.image.mnist(global_cache=True).get()

print("X_train:", X_train.shape, X_train.dtype)
print("y_train:", y_train.shape, y_train.dtype)
print("X_test:", X_test.shape, X_test.dtype)
print("y_test:", y_test.shape, y_test.dtype)


# In this case `dataget` loads the data from Yann LeCun's website.
# 
# ## Defining a simple Model
# 
# The low-level API lets you redefine what happens during the various stages of training, evaluation and inference by implementing some methods in a custom class. Here is the list of methods you can define along with the high-level method that uses it:
# 
# | Low-level Method | High-level Method | 
# | :- | :- |
# | `pred_step` | `predict` |
# | `test_step` | `evaluate` |
# | `grad_step` | NA |
# | `train_step` | `fit` |
# 
# Check out the guides on the low-level API for more information.
# 
# In this tutorial we are going to implement Linear Classifier using pure Jax by overriding `Model.test_step` which defines loss and metrics of our model. `test_step` returns a tuple with:
# * `loss`: the scalar loss use to calculate the gradient
# * `logs`: a dictionary with the logs to be reported during training
# * `states`: a `elegy.States` namedtuple that contains the states for thing like network trainable parameter, network states, metrics states, optimizer states, rng state.
# 
# Since Jax is functional you will find that low-level API is very explicit with state management, that is, you always get the currrent state as input and you return the new state as output. Lets define `test_step` to make things clearer:

# In[2]:


import elegy, jax
import numpy as np
import jax.numpy as jnp


class LinearClassifier(elegy.Model):
    # request parameters by name via depending injection.
    # possible: net_params, x, y_true, net_states, metrics_states, sample_weight, class_weight, rng, states, initializing
    def test_step(
        self,
        x, # inputs
        y_true, # labels
        states: elegy.States, # model state
        initializing: bool, # if True we should initialize our parameters
        rng: elegy.RNGSeq, # rng.next() ~= jax.random.split(...)
    ):  
        # flatten + scale
        x = jnp.reshape(x, (x.shape[0], -1)) / 255

        # maybe init
        if initializing:
            w = jax.random.uniform(
                rng.next(), shape=[np.prod(x.shape[1:]), 10], minval=-1, maxval=1
            )
            b = jax.random.uniform(rng.next(), shape=[1], minval=-1, maxval=1)
        else:
            w, b = states.net_params
        
        # model
        logits = jnp.dot(x, w) + b

        # crossentropy loss
        labels = jax.nn.one_hot(y_true, 10)
        loss = jnp.mean(-jnp.sum(labels * jax.nn.log_softmax(logits), axis=-1))
        accuracy=jnp.mean(jnp.argmax(logits, axis=-1) == y_true)

        # metrics
        logs = dict(
            accuracy=accuracy,
            loss=loss,
        )

        return loss, logs, states.update(rng=rng, net_params=(w, b))


# Notice the following:
# * We define a bunch of arguments with specific names, Elegy uses Dependency Injection so you can just request what you need.
# * `initializing` tells use if we should initialize our parameters, here we are directly creating them ourselves but if you use a Module system you can conditionally call its `init` method here.
# * Our model is defined by a simple linear function.
# * Defined a simple crossentropy loss and an accuracy metric, we added both the the logs.
# * We set the updated `States.net_params` with the `w` and `b` parameters so we get them as an input on the next run after they are initialized.
# * `States.update` ofers a clean way inmutably update the states without having to copy all fields to a new States structure.
# 
# Remember `test_step` only defines what happens what happens during `evaluate`, however, Elegy's `Model` default implementation has a structure where on method is defined in terms of another:
# 
# ```
# pred_step ⬅ test_step ⬅ grad_step ⬅ train_step
# ```
# 
# Because of this, we get the `train_step` / `fit` for free if we just pass an optimizer to the the constructor as we are going to do next:

# In[4]:


import optax

model = LinearClassifier(
    optimizer=optax.adam(1e-3)
)


# If we try to get the summaries we will get an error as we haven't defined `pred_step` which is a depencency:

# In[5]:


try:
    model.summary(X_train[:64])
except BaseException as e:
    print(e)


# ## Training the Model
# Now that we have our model we can just call `fit`. The following code will train our model for `100` epochs while limiting each epoch to `200` steps and using a batch size of `64`:

# In[ ]:


history = model.fit(
    x=X_train,
    y=y_train,
    epochs=100,
    steps_per_epoch=200,
    batch_size=64,
    validation_data=(X_test, y_test),
    shuffle=True,
    callbacks=[elegy.callbacks.ModelCheckpoint("models/low-level", save_best_only=True)],
)


# ```
# ...
# 
# Epoch 99/100
# 200/200 [==============================] - 0s 680us/step - accuracy: 0.9375 - loss: 0.2508 - val_accuracy: 1.0000 - val_loss: 0.0878
# Epoch 100/100
# 200/200 [==============================] - 0s 618us/step - accuracy: 0.9219 - loss: 0.2440 - val_accuracy: 0.8750 - val_loss: 0.1224                      
# ```
# 
# The `elegy.callbacks.ModelCheckpoint` callback will periodicall saves the model a folder called `"models/low-level"` during training which is very useful if we want to load it after the process is finished, we will use it later.`fit` returns a `History` object which which contains information the time series for the values of the losses and metrics throughout training, we can use it to generate some nice plots of the evolution of our training:

# In[7]:


import matplotlib.pyplot as plt

def plot_history(history):
    n_plots = len(history.history.keys()) // 2
    plt.figure(figsize=(14, 24))
    
    for i, key in enumerate(list(history.history.keys())[:n_plots]):
        metric = history.history[key]
        val_metric = history.history[f"val_{key}"]

        plt.subplot(n_plots, 1, i + 1)
        plt.plot(metric, label=f"Training {key}")
        plt.plot(val_metric, label=f"Validation {key}")
        plt.legend(loc="lower right")
        plt.ylabel(key)
        plt.title(f"Training and Validation {key}")
    plt.show()
    
plot_history(history)


# Notice that the logs are very noisy, this is because for this example we didn't use cummulative metrics so the reported value is just the value for the last batch of that epoch, not the value for the entire epoch. To fix this we could use some of the modules in `elegy.metrics`.

# ## Generating Predictions
# 
# Having trained our model we can now get some samples from the test set and generate some predictions. First we will just pick some random samples using `numpy`:

# In[8]:


import numpy as np

idxs = np.random.randint(0, 10000, size=(9,))
x_sample = X_test[idxs]


# Here we selected `9` random images. Now if we had implemented `pred_step` we could've used `predict`, instead we are just going to the the calculation by hand.

# In[9]:


x = x_sample.reshape(x_sample.shape[0], -1)
w, b = model.states.net_params
y_pred = jnp.dot(x, w) + b
y_pred.shape


# Easy right? Finally lets plot the results to see if they are accurate.

# In[10]:


plt.figure(figsize=(12, 12))
for i in range(3):
    for j in range(3):
        k = 3 * i + j
        plt.subplot(3, 3, k + 1)
    
        plt.title(f"{np.argmax(y_pred[k])}")
        plt.imshow(x_sample[k], cmap="gray")


# Good enough!
# 
# ## Serialization
# To serialize the `Model` you can use the `model.save(...)`, this will create a folder with some files that contain the model's code plus all parameters and states, however since we had previously used the `ModelCheckpoint` callback we can load it using `elegy.load`. Lets get a new model reference containing the same weights and call its `evaluate` method to verify it loaded correctly:

# In[11]:


# You can use can use `save` but `ModelCheckpoint already serialized the model
# model.save("model")

# current model reference
print("current model id:", id(model))

# load model from disk
model = elegy.load("models/low-level")

# new model reference
print("new model id:    ", id(model))

# check that it works!
model.evaluate(x=X_test, y=y_test)


# Excellent! We hope you've enjoyed this tutorial.
