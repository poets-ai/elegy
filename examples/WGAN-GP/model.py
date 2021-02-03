import jax, jax.numpy as jnp
import elegy
import optax

#the generator architecture adapted from DCGAN
class Generator(elegy.Module):
    def call(self, z):
        assert len(z.shape)==2
        x = elegy.nn.Reshape([1,1, z.shape[-1]])(z)
        for i,c in enumerate([1024, 512, 256, 128]):
            padding = 'VALID' if i==0 else 'SAME'
            x = elegy.nn.conv.Conv2DTranspose(c, (4,4), stride=(2,2), padding=padding)(x)
            x = elegy.nn.BatchNormalization(decay_rate=0.9)(x)
            x = jax.nn.leaky_relu(x, negative_slope=0.2)
        x = elegy.nn.conv.Conv2DTranspose(3, (4,4), stride=(2,2))(x)
        x = jax.nn.sigmoid(x)
        return x

#the discriminator architecture adapted from DCGAN
#also called 'critic' in the WGAN paper
class Discriminator(elegy.Module):
    def call(self, x):
        for c in [128, 256, 512, 1024]:
            x = elegy.nn.conv.Conv2D(c, (4,4), stride=(2,2))(x)
            x = jax.nn.leaky_relu(x, negative_slope=0.2)
        x = elegy.nn.Flatten()(x)
        x = elegy.nn.Linear(1)(x)
        return x


#multiplier for gradient normalization
LAMBDA_GP = 10

#gradient regularization term
def gradient_penalty(x_real, x_fake, applied_discriminator_fn, rngkey):
        assert len(x_real)==len(x_fake)
        alpha = jax.random.uniform(rngkey, shape=[len(x_real),1,1,1])
        x_hat = x_real * alpha + x_fake * (1-alpha)
        grads = jax.grad( lambda x: applied_discriminator_fn(x)[0].mean() )(x_hat)
        norm  = jnp.sqrt((grads**2).sum(axis=[1,2,3]))
        penalty = (norm - 1)**2
        return penalty.mean() * LAMBDA_GP



class WGAN_GP(elegy.Model):
    def __init__(self):
        #run_eagerly=True is needed to train the generator only every 5 iterations
        #as recommended in the WGAN paper
        super().__init__(run_eagerly=True)
        self.generator     = Generator()
        self.discriminator = Discriminator()
        self.g_optimizer   = optax.adam(2e-4, b1=0.5)
        self.d_optimizer   = optax.adam(2e-4, b1=0.5)
        
        #iteration counter
        self.i = 0
        self.discriminator_step_j = jax.jit(self.discriminator_step)
        self.generator_step_j     = jax.jit(self.generator_step, static_argnums=[0])
    
    def init(self, x):
        rng = elegy.RNGSeq(0)
        gx, g_variables = self.generator.init(rng=rng)(x)
        dx, d_variables = self.discriminator.init(rng=rng)(gx)
        
        g_optimizer_states = self.g_optimizer.init(g_variables['parameters'])
        d_optimizer_states = self.d_optimizer.init(d_variables['parameters'])

        return elegy.States(
            net_states=(g_variables.get('states',{}), d_variables.get('states',{})),
            net_params=(g_variables['parameters'],    d_variables['parameters']),
            optimizer_states=(g_optimizer_states, d_optimizer_states),
            rng=rng,
        )
    
    def pred_step(self, x, net_states, net_params, states, rng):
        g_states, d_states = net_states
        g_params, d_params = net_params
        z = x
        x_fake = self.generator.apply({'parameters':g_params, 'states':g_states})(z)[0]
        return elegy.PredStep.simple(x_fake, states)
        
    
    def train_step(self, x, net_states, net_params, optimizer_states, rng):
        g_states, d_states = net_states
        g_params, d_params = net_params
        g_variables = {'parameters':g_params, 'states':g_states}
        g_opt_states, d_opt_states = optimizer_states
        
        #training the discriminator on every iteration
        d_loss, d_variables, d_opt_states, rng, gp = self.discriminator_step_j(x, d_params, d_states, g_variables, d_opt_states, rng)
        
        self.i+=1
        #training the generator only every 5 iterations as recommended in the original WGAN paper
        if self.i%5 == 0:
            g_loss, g_variables, g_opt_states, rng  = self.generator_step_j(len(x), g_params, g_states, d_variables, g_opt_states, rng)
        else:
            g_loss = 0
        
        return {'d_loss':d_loss, 'g_loss':g_loss, 'gp':gp}, elegy.States(
            net_params=(g_variables.get('parameters',{}), d_variables.get('parameters',{})),
            net_states=(g_variables.get('states',{}), d_variables.get('states',{})),
            optimizer_states=(g_opt_states, d_opt_states),
            rng=rng,
        )
    
    def discriminator_step(self, x_real, d_params, d_states, g_variables, d_opt_states, rng):
        z        = jax.random.normal(rng.next(), (len(x_real), 128))
        x_fake   = self.generator.apply(g_variables)(z)[0]
        
        def d_loss_fn(d_params, d_states, x_real, x_fake, rng):
            d_variables            = {'parameters':d_params, 'states':d_states}
            y_real, d_variables    = self.discriminator.apply(d_variables)(x_real)
            y_fake, d_variables    = self.discriminator.apply(d_variables)(x_fake)
            y_pred                 = jnp.concatenate([y_real, y_fake], axis=0)
            y_true                 = jnp.concatenate([jnp.ones(len(x_real)), jnp.zeros(len(x_fake))], axis=0)
            loss                   = -y_real.mean() + y_fake.mean()
            gp                     = gradient_penalty(x_real, x_fake, self.discriminator.apply(d_variables), rng.next())
            loss                   = loss + gp
            return loss, (d_variables, rng, gp)
        
        (d_loss, (d_variables, rng, gp)), d_grads = jax.value_and_grad(d_loss_fn, has_aux=True)(d_params, d_states, x_real, x_fake, rng)
        d_grads, d_opt_states                 = self.d_optimizer.update(d_grads, d_opt_states, d_params)
        d_params                              = optax.apply_updates(d_params, d_grads)
        
        d_variables = {'parameters':d_params, 'states':d_states}
        return d_loss, d_variables, d_opt_states, rng, gp
    
    def generator_step(self, batch_size, g_params, g_states, d_variables, g_opt_states, rng):
        z = jax.random.normal(rng.next(), (batch_size, 128))
        
        def g_loss_fn(g_params, g_states, d_variables, z):
            g_variables            = {'parameters':g_params, 'states':g_states}
            x_fake, g_variables    = self.generator.apply(g_variables)(z)
            y_fake_scores          = self.discriminator.apply(d_variables)(x_fake)[0]
            y_fake_true            = jnp.ones(len(z))
            loss                   = -y_fake_scores.mean()
            return loss, g_variables
        
        (g_loss, g_variables), g_grads = jax.value_and_grad(g_loss_fn, has_aux=True)(g_params, g_states, d_variables, z)
        g_grads, g_opt_states          = self.g_optimizer.update(g_grads, g_opt_states, g_params)
        g_params                       = optax.apply_updates(g_params, g_grads)
        
        g_variables                    = {'parameters':g_params, 'states':g_variables.get('states',{})}
        return g_loss, g_variables, g_opt_states, rng



