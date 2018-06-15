import keras.backend as K 
import numpy as np

class BEGANLoss():
    '''
    intended to implement BEGAN loss as described in
    https://arxiv.org/abs/1703.10717

    crafted from keras github issue#4813
    and
    mokemokechicken's repository

    # Arguments
        k_init: Float; initial k factor
        lambda_k: Float; k learning rate
        gamma: Float; equilibrium factor

    # To User:
        #   you should instanciate this object
            before compiling a model with
        #   updates will only work if the model
            collects them (LossUpdaterModel from
            this library .models will achieve
            this)
    
    '''
    
    __name__ = 'began_loss'
    
    def __init__(self, initial_k=0.001, lambda_k=0.001, gamma=0.5):
        self.lambda_k = lambda_k
        self.gamma = gamma
        self.k_var = K.variable(initial_k, dtype=K.floatx(), name="shadow_k")
        self.m_global_var = K.variable(np.array([0]), dtype=K.floatx(), name="m_global")
        self.updates=[]

    def __call__(self, y_true, y_pred):  # y_true, y_pred shape: (batch_size, nb_class)
        # LET'S MAKE A STRONG HYPOTHESIS: BATCH IS HALF TRUE & HALF GENERATED
        # ORDERED AS  EVEN NUMBERS = TRUE & ODD NUMBERS = GENERATED
        true_true, generator_true = y_true[:, ::2], y_true[:, 1::2] #even, odd
        true_pred, generator_pred = y_pred[:, ::2], y_pred[:, 1::2] #even, odd
        loss_true = K.mean(K.abs(true_true - true_pred))
        loss_generator = K.mean(K.abs(generator_true - generator_pred))
        began_loss = loss_true - self.k_var * loss_generator
        mean_loss_true = K.mean(loss_true)
        mean_loss_gen = K.mean(loss_generator)

        # updates will be collected by a model such as LossUpdaterModel
        # update K
        new_k = self.k_var + self.lambda_k * (self.gamma * mean_loss_true - mean_loss_gen)
        new_k = K.clip(new_k, 0, 1)
        self.updates.append(K.update(self.k_var, new_k))

        # calculate M-Global
        m_global = mean_loss_true + K.abs(self.gamma * mean_loss_true - mean_loss_gen)
        m_global = K.reshape(m_global, (1,))
        self.updates.append(K.update(self.m_global_var, m_global))

        return began_loss

    # Allow user to consult values of k and M_global
    @property
    def k(self):
        return K.get_value(self.k_var)

    @property
    def m_global(self):
        return K.get_value(self.m_global_var)