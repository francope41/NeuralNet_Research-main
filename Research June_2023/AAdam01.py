import tensorflow as tf

class AAdam(tf.keras.optimizers.Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, name="AAdam", **kwargs):
        super().__init__(name=name, **kwargs)
        self.learning_rate = self.add_weight("learning_rate", initializer=tf.constant_initializer(learning_rate))
        self.beta1 = self.add_weight("beta1", initializer=tf.constant_initializer(beta1))
        self.beta2 = self.add_weight("beta2", initializer=tf.constant_initializer(beta2))
        self.epsilon = epsilon

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'm')
            self.add_slot(var, 'v')
            self.add_slot(var, 'd')
            self.add_slot(var, 'g')

    @tf.function
    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype
        lr_t = tf.cast(self.learning_rate, var_dtype)
        beta1_t = tf.cast(self.beta1, var_dtype)
        beta2_t = tf.cast(self.beta2, var_dtype)
        epsilon_t = tf.cast(self.epsilon, var_dtype)

        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')
        d = self.get_slot(var, 'd')
        g = self.get_slot(var, 'g')

        m_t = m.assign(beta1_t * m + (1. - beta1_t) * grad)
        v_t = v.assign(beta2_t * v + (1. - beta2_t) * tf.square(grad))
        alpha_t =  tf.sqrt(1 - beta2_t) / (1 - beta1_t)

        g_t =  (m_t*alpha_t) / (tf.sqrt(v_t) + epsilon_t)
        m_t = tf.where((tf.sign(g) * tf.sign(grad)) < 0, tf.sign(g_t) * (beta1_t * tf.abs(g_t) - (1-beta1_t) * tf.abs(d)), tf.sign(g_t) * (beta1_t * tf.abs(g_t) + (1-beta1_t) * tf.abs(d)))
        d_t = d.assign(m_t)
        g_t = g.assign(grad)

        var_update = var.assign_sub(lr_t * d_t)
        updates = [var_update, m_t, v_t, d_t, g_t]
        return tf.group(*updates)

    def _resource_apply_sparse(self, grad, var, indices):
        raise NotImplementedError("Sparse gradient updates are not supported.")

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "learning_rate": self.learning_rate.numpy(),
            "beta1": self.beta1.numpy(),
            "beta2": self.beta2.numpy(),
            "epsilon": self.epsilon,
        }
