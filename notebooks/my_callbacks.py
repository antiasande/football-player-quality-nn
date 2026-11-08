from tensorflow.keras.callbacks import Callback
from tqdm import tqdm


################
class TqdmProgressCallback(Callback):
    def __init__(self, epochs, verbose=0):
        super(TqdmProgressCallback, self).__init__()
        self.epochs = epochs
        self.verbose = verbose
        
    def on_train_begin(self, logs=None):
        self.epoch_bar = tqdm(total=self.epochs, desc='Epochs', position=0)
       
    def on_epoch_end(self, epoch, logs=None):
        self.epoch_bar.update(1)
        
    def on_train_end(self, logs=None):
        self.epoch_bar.close()

#########################
class StopWhenValLossBelow(Callback):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss')
        if val_loss is not None and val_loss < self.threshold:
            print(f"\nStopping training: val_loss has reached {val_loss:.4f}, which is below the threshold of {self.threshold}")
            self.model.stop_training = True

#######################
class StopTrainingAtEpoch(Callback):
    def __init__(self, stop_epoch=5):
        super().__init__()
        self.stop_epoch = stop_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch == self.stop_epoch:
            print(f"\nStopping training at epoch {epoch}")
            self.model.stop_training = True