import os
import haiku as hk
import jax.numpy as jnp
import dataget


def main(debug: bool = False):
    if debug:
        import debugpy

        print("Waiting for debugger...")
        debugpy.listen(5678)
        debugpy.wait_for_client()

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    logdir = os.path.join(logdir, current_time)

    X_train, y_train, X_test, y_test = dataget.image.mnist(global_cache=True).get()

    print("X_train:", X_train.shape, X_train.dtype)
    print("y_train:", y_train.shape, y_train.dtype)
    print("X_test:", X_test.shape, X_test.dtype)
    print("y_test:", y_test.shape, y_test.dtype)

    def softmax_cross_entropy(logits, labels):
        one_hot = jax.nn.one_hot(labels, logits.shape[-1])
        # return -jnp.


if __name__ == "__main__":
    main()
