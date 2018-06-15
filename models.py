from keras.engine import Model

class LossUpdaterModel(Model):
    """Model which collects updates from loss_func.updates"""

    # This should be expanded into a model object that collects
    # updates from all its subcomponents. Another item on
    # my to-do list.

    @property
    def updates(self):
        updates = super().updates
        if hasattr(self, 'loss_functions'):
            for loss_func in self.loss_functions:
                if hasattr(loss_func, 'updates'):
                    updates += loss_func.updates
        return updates