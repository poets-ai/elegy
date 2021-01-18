import matplotlib.pyplot as plt


def plot_history(history):
    keys = [key for key in history.history.keys() if not key.startswith("val_")]
    n_plots = len(keys)

    figure = plt.figure(figsize=(14, 24))

    # for i, key in enumerate(list(history.history.keys())[:n_plots]):
    for i, key in enumerate(keys):
        if key == "size":
            continue

        metric = history.history[key]

        plt.subplot(n_plots, 1, i + 1)
        plt.plot(metric, label=f"Training {key}")

        try:
            val_metric = history.history[f"val_{key}"]
            plt.plot(val_metric, label=f"Validation {key}")
            title = f"Training and Validation {key}"
        except KeyError:
            title = f"Training {key}"

        plt.legend(loc="lower right")
        plt.ylabel(key)
        plt.title(title)

    plt.show()
