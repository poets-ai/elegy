import matplotlib.pyplot as plt


def plot_history(history):
    n_plots = len(history.history.keys()) // 2
    figure = plt.figure(figsize=(14, 24))

    for i, key in enumerate(list(history.history.keys())[:n_plots]):
        if key == "size":
            continue

        metric = history.history[key]
        val_metric = history.history[f"val_{key}"]

        plt.subplot(n_plots, 1, i + 1)
        plt.plot(metric, label=f"Training {key}")
        plt.plot(val_metric, label=f"Validation {key}")
        plt.legend(loc="lower right")
        plt.ylabel(key)
        plt.title(f"Training and Validation {key}")
    plt.show()
