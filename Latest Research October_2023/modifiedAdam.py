import tensorflow as tf

class ModifiedAdam(tf.keras.optimizers.Adam):
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, use_locking=False, amsgrad=False, name="ModifiedAdam", **kwargs):
        super(ModifiedAdam, self).__init__(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, amsgrad=amsgrad, name=name, **kwargs)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        # get current states
        # m = self.get_slot(var, 'm')
        # v = self.get_slot(var, 'v')

        # # here are the original update equations in Adam
        # m.assign(m * self._get_hyper('beta_1') + grad * (1. - self._get_hyper('beta_1')))
        # v.assign(v * self._get_hyper('beta_2') + (grad ** 2) * (1. - self._get_hyper('beta_2')))

        # # make your modifications here
        # # for example, let's square the m term before using it in the update
        # m_mod = m ** 2

        # var.assign_sub(self._get_hyper('learning_rate') * m_mod / (tf.sqrt(v) + self._get_hyper('epsilon')))

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

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'm')
            self.add_slot(var, 'v')
            self.add_slot(var, 'd')
            self.add_slot(var, 'g')


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