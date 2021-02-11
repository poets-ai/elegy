import jax, jax.numpy as jnp
import elegy
import optax

# the generator architecture adapted from DCGAN
class Generator(elegy.Module):
    def call(self, z):
        assert len(z.shape) == 2
        x = elegy.nn.Reshape([1, 1, z.shape[-1]])(z)
        for i, c in enumerate([1024, 512, 256, 128]):
            padding = "VALID" if i == 0 else "SAME"
            x = elegy.nn.conv.Conv2DTranspose(
                c, (4, 4), stride=(2, 2), padding=padding
            )(x)
            x = elegy.nn.BatchNormalization(decay_rate=0.9)(x)
            x = jax.nn.leaky_relu(x, negative_slope=0.2)
        x = elegy.nn.conv.Conv2DTranspose(3, (4, 4), stride=(2, 2))(x)
        x = jax.nn.sigmoid(x)
        return x


# the discriminator architecture adapted from DCGAN
# also called 'critic' in the WGAN paper
class Discriminator(elegy.Module):
    def call(self, x):
        for c in [128, 256, 512, 1024]:
            x = elegy.nn.conv.Conv2D(c, (4, 4), stride=(2, 2))(x)
            x = jax.nn.leaky_relu(x, negative_slope=0.2)
        x = elegy.nn.Flatten()(x)
        x = elegy.nn.Linear(1)(x)
        return x


# multiplier for gradient normalization
LAMBDA_GP = 10

# gradient regularization term
def gradient_penalty(x_real, x_fake, applied_discriminator_fn, rngkey):
    assert len(x_real) == len(x_fake)
    alpha = jax.random.uniform(rngkey, shape=[len(x_real), 1, 1, 1])
    x_hat = x_real * alpha + x_fake * (1 - alpha)
    grads = jax.grad(lambda x: applied_discriminator_fn(x)[0].mean())(x_hat)
    norm = jnp.sqrt((grads ** 2).sum(axis=[1, 2, 3]))
    penalty = (norm - 1) ** 2
    return penalty.mean() * LAMBDA_GP


class WGAN_GP(elegy.Model):
    def __init__(self):
        super().__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.g_optimizer = optax.adam(2e-4, b1=0.5)
        self.d_optimizer = optax.adam(2e-4, b1=0.5)

    def init(self, x):
        rng = elegy.RNGSeq(0)
        gx, g_params, g_states = self.generator.init(rng=rng)(x)
        dx, d_params, d_states = self.discriminator.init(rng=rng)(gx)

        g_optimizer_states = self.g_optimizer.init(g_params)
        d_optimizer_states = self.d_optimizer.init(d_params)

        self.states = elegy.States(
            g_states=g_states,
            d_states=d_states,
            g_params=g_params,
            d_params=d_params,
            g_opt_states=g_optimizer_states,
            d_opt_states=d_optimizer_states,
            rng=rng,
            step=0,
        )
        self.initial_states = self.states.copy()

    def pred_step(self, x, states):
        z = x
        x_fake = self.generator.apply(states.g_params, states.g_states)(z)[0]
        return (x_fake, states)

    def train_step(self, x, states):
        # training the discriminator on every iteration
        d_loss, gp, states = self.discriminator_step_jit(x, states)

        step = states.step + 1
        no_update = lambda args: (0.0, args[1])
        do_update = lambda args: self.generator_step_jit(len(args[0]), args[1])
        g_loss, states = jax.lax.cond(step % 5 == 0, do_update, no_update, (x, states))

        return {"d_loss": d_loss, "g_loss": g_loss, "gp": gp}, states.update(step=step)

    def discriminator_step(self, x_real: jnp.ndarray, S: elegy.States):
        z = jax.random.normal(S.rng.next(), (len(x_real), 128))
        x_fake = self.generator.apply(S.g_params, S.g_states)(z)[0]

        def d_loss_fn(d_params, S, x_real, x_fake):
            y_real, d_params, d_states = self.discriminator.apply(d_params, S.d_states)(
                x_real
            )
            y_fake, d_params, d_states = self.discriminator.apply(d_params, d_states)(
                x_fake
            )
            y_pred = jnp.concatenate([y_real, y_fake], axis=0)
            y_true = jnp.concatenate(
                [jnp.ones(len(x_real)), jnp.zeros(len(x_fake))], axis=0
            )
            loss = -y_real.mean() + y_fake.mean()
            gp = gradient_penalty(
                x_real,
                x_fake,
                self.discriminator.apply(d_params, d_states),
                S.rng.next(),
            )
            loss = loss + gp
            return loss, (gp, S.safe_update(**locals()))

        (d_loss, (gp, S)), d_grads = jax.value_and_grad(d_loss_fn, has_aux=True)(
            S.d_params, S, x_real, x_fake
        )
        d_grads, d_opt_states = self.d_optimizer.update(
            d_grads, S.d_opt_states, S.d_params
        )
        d_params = optax.apply_updates(S.d_params, d_grads)

        return d_loss, gp, S.safe_update(**locals())

    def generator_step(self, batch_size: int, S: elegy.States):
        z = jax.random.normal(S.rng.next(), (batch_size, 128))

        def g_loss_fn(g_params, S, z):
            x_fake, g_params, g_states = self.generator.apply(g_params, S.g_states)(z)
            y_fake_scores = self.discriminator.apply(S.d_params, S.d_states)(x_fake)[0]
            y_fake_true = jnp.ones(len(z))
            loss = -y_fake_scores.mean()
            return loss, S.safe_update(**locals())

        (g_loss, S), g_grads = jax.value_and_grad(g_loss_fn, has_aux=True)(
            S.g_params, S, z
        )
        g_grads, g_opt_states = self.g_optimizer.update(
            g_grads, S.g_opt_states, S.g_params
        )
        g_params = optax.apply_updates(S.g_params, g_grads)

        return g_loss, S.safe_update(**locals())
