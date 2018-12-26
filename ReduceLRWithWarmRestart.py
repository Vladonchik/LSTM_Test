#
#
#
#
#
#
#
#
#
#

import keras
from keras.callbacks import *
import numpy

class ReduceLRWithWarmRestart(Callback): # implementation of https://arxiv.org/abs/1608.03983
    def __init__(self,
                 arg_filepath='ReduceLRWithWarmRestart.dhf5',
                 arg_min_lr=0.0,
                 arg_t_multiplier=2.0,
                 arg_initial_epoch_count=1,
                 arg_max_epoch_count=0,
                 monitor='val_loss',
                 mode='auto',
                 min_delta=1e-6):

        super(ReduceLRWithWarmRestart, self).__init__()

        self._att_filepath = arg_filepath
        self._att_min_lr = arg_min_lr
        self._att_max_lr = None
        self._att_t_multiplier = arg_t_multiplier

        self._att_current_max_epoch_count = int(arg_initial_epoch_count)
        self._att_max_epoch_count = int(arg_max_epoch_count)
        self.monitor = monitor
        self.mode = mode
        self.min_delta = min_delta

        self.monitor_op = None
        self._att_current_epoch_count = 0
        self._att_best_weights = None
        self._att_best_score = 0
        self._att_cycle_count = 1
        self._att_initial_epoch_count = int(arg_initial_epoch_count)

        self._reset()

    def _reset(self):
        if self.mode not in ['auto', 'min', 'max']:
            warnings.warn('Learning Rate Plateau Reducing mode %s is unknown, '
                          'fallback to auto mode.' % self.mode,
                          RuntimeWarning)

            self.mode = 'auto'

        if (self.mode == 'min' or
           (self.mode == 'auto' and 'acc' not in self.monitor)):
            self.monitor_op = lambda a, b: numpy.less(a, b - self.min_delta)
            self._att_best_score = numpy.Inf
        else:
            self.monitor_op = lambda a, b: numpy.greater(a, b + self.min_delta)
            self._att_best_score = -numpy.Inf

    def on_train_begin(self, logs=None):
        self._reset()

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)


    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn(
                'metric `%s` is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )
        else:
            self._att_current_epoch_count += 1

            if self._att_max_lr is None:
                self._att_max_lr = float(K.get_value(self.model.optimizer.lr))

            if self.monitor_op(current, self._att_best_score):
                self._att_best_score = current
                self.model.save_weights(self._att_filepath, overwrite=True)

            if self._att_current_epoch_count >= self._att_current_max_epoch_count:
                self.model.load_weights(filepath=self._att_filepath)
                self._att_cycle_count += 1

                self._att_current_epoch_count = 0
                self._att_current_max_epoch_count = int(self._att_initial_epoch_count * (self._att_t_multiplier ** self._att_cycle_count))
                if self._att_max_epoch_count > 0:
                    if self._att_current_max_epoch_count > self._att_max_epoch_count:
                        self._att_current_max_epoch_count = self._att_max_epoch_count

                K.set_value(self.model.optimizer.lr, self._att_max_lr)

                print('ReduceLRWithWarmRestart: WarmRestarted, new max_epoch_count =', self._att_current_max_epoch_count)
            else:
                the_new_lr = self._att_min_lr + 0.5 * (self._att_max_lr - self._att_min_lr) * (1 + numpy.cos(self._att_current_epoch_count / self._att_current_max_epoch_count * numpy.pi))
                K.set_value(self.model.optimizer.lr, the_new_lr)