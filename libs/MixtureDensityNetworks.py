import tensorflow as tf
from tensorflow import keras



def calc_pdf(y, mu, var):
    """Calculate component density"""
    value = tf.subtract(y, mu)**2
    value = (1/tf.math.sqrt(2 * np.pi * var)) * tf.math.exp((-1/(2*var)) * value)
    return value


def mdn_loss(y_true, pi, mu, var):
    """MDN Loss Function
    The eager mode in tensorflow 2.0 makes is extremely easy to write 
    functions like these. It feels a lot more pythonic to me.
    """
    out = calc_pdf(y_true, mu, var)
    # multiply with each pi and sum it
    out = tf.multiply(out, pi)
    out = tf.reduce_sum(out, 1, keepdims=True)
    out = -tf.math.log(out + 1e-10)
    return tf.reduce_mean(out)


class GaussianMDN(keras.Model):

    def __init__(self,
                 n_mixtures,
                 input_dim,
                 layer_units,
                 layer_activations=None
                 ):

        self.n_mixtures  = n_mixtures
        self.input_dim   = input_dim
        self.layer_units = layer_units

        if layer_activations is None:
            self.layer_activations = [ 'relu' for _ in range(len(layer_units)) ]
        
        elif isinstance(layer_activations, list):
            if len(layer_activations) != len(layer_units): raise ValueError
            self.layer_activations = layer_activations

        elif isinstance(layer_activations, str):
            self.layer_activations = [layer_activations for _ in range(len(layer_units))]

        else:
            raise ValueError

            

        inp = keras.Input(shape=(self.input_dim,), name='mdn_input')
        prev_tensor = inp
        for i, units, activation in zip(range(len(self.layer_units)), self.layer_units, self.layer_activations):
            layer       = keras.layers.Dense(units, activation=activation, name=f"mdn_dense{i}")(prev_tensor)
            prev_tensor = layer

        mu   = keras.layers.Dense(self.n_mixtures*self.input_dim, activation=None,                          name='mdn_mean')(prev_tensor)
        var  = keras.layers.Dense(self.n_mixtures,                activation=keras.activations.exponential, name='mdn_var')(prev_tensor)
        pi   = keras.layers.Dense(self.n_mixtures,                activation='softmax',                     name='mdn_pi')(prev_tensor)

    
        super(keras.Model, self).__init__(inp, [pi, mu, var])

        self.pi  = pi
        self.mu  = mu
        self.var = var
        self.optimizer = None



    def compile(self, optimizer):
        self.optimizer = optimizer


    def fit(self, X, y, epochs=5, batch_size=32, validation_data=None, verbose=1):

        N = X.shape[0]
        dataset = tf.data.Dataset \
            .from_tensor_slices((X, y)) \
            .shuffle(N).batch(batch_size)


        @tf.function
        def train_step(model, optimizer, train_x, train_y):
            # GradientTape: Trace operations to compute gradients
            with tf.GradientTape() as tape:
                pi_, mu_, var_ = self(train_x, training=True)
                # calculate loss
                loss = mdn_loss(train_y, pi_, mu_, var_)
            # compute and apply gradients
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return loss


        losses     = []
        val_losses = []

        for epoch_num in range(epochs):


            for train_x, train_y in dataset:
                loss = train_step(self.optimizer, train_x, train_y)
                losses.append(loss)


            if verbose > 0:
                print('Epoch {}/{}: loss {}'.format(epoch_num, epochs, loss), end='')

            if validation_data is not None:
                X_val, y_val   = validation_data
                pi_, mu_, var_ = self(X_val, training=False)
                val_loss       = mdn_loss(y_val, pi_, mu_, var_)
                val_losses.append(val_loss)

                if verbose > 0:
                    print(f" - val_loss {val_loss}")
            else:
                if verbose > 0:
                    print('')


        return losses, val_losses



    def predict(self, X):
        pi_, mu_, var_ = self(X, training=False)
        max_component = tf.argmax(pi_, axis=1)
        preds = np.empty(X.shape[0])
        for i in range(X.shape[0]):
            preds[i] = max_components[i]





if __name__ == '__main__':
    model = GaussianMDN(3, 1, [10,20,30], 'relu')