import os
import math
import numpy as np
import datetime as dt
from numpy import newaxis
from Utils.General import Timer
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

####Data Shape:
#"shape": {
 #   "TimeSteps" : 24,
#  "FeaturesPerStep" : 35
#}

class Model() :
    """A class for an building and inferencing an lstm model"""

    def __init__ ( self , model_name) :
        self.model = Sequential(name = model_name)

    def load_model ( self, filepath ) :
        print('[Model] Loading model from file %s' % filepath)
        self.model = load_model(filepath)

    def BuildModel (self, configs, sequence, rows, columns):

        model = Sequential()
        model.add(LSTM(sequence, return_sequences=True, input_shape=(rows, columns)))

    def build_model ( self, configs ) :
        timer = Timer()
        timer.start()

        FeaturesPerStep = configs['data']['sequence_length']
        TimeSteps = configs['training']['batch_size']

        for layer in configs['model']['layers'] :
            if layer["type"] =="dense":
                neurons = TimeSteps

            else:
                neurons = layer['neurons'] if 'neurons' in layer else None

            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None

            if 'input_timesteps' in layer and layer['input_timesteps'] == -1:
                input_timesteps = TimeSteps
                input_dim = FeaturesPerStep

            else:
                input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
                input_dim = layer['input_dim'] if 'input_dim' in layer else None

            print("INPUT TIME STEPS ARE: ", input_timesteps, "time steps and FEATURES ARE: ", input_dim)

            if layer['type'] == 'dense' :
                self.model.add(Dense(neurons, activation=activation))
            if layer['type'] == 'lstm' :
                self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
            if layer['type'] == 'dropout' :
                self.model.add(Dropout(dropout_rate))

        self.model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'])

        print('[Model] Model Compiled')
        print(self.model.summary())
        timer.stop()

    def train ( self, x, y, epochs, batch_size, save_dir ) :
        timer = Timer()
        timer.start()

        save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=2),
            ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True)
        ]
        self.model.fit(
            x,
            y,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        self.model.save(save_fname)

        print('[Model] Training Completed. Model saved as %s' % save_fname)
        timer.stop()

    def train_generator ( self, data_gen, epochs, batch_size, steps_per_epoch, save_dir ) :
        timer = Timer()
        timer.start()
        #print('[Model] Training Started')
        #print('[Model] %s epochs, %s batch size, %s batches per epoch' % (epochs, batch_size, steps_per_epoch))

        save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
        callbacks = [
            ModelCheckpoint(filepath=save_fname, monitor='loss', save_best_only=True)
        ]
        self.model.fit_generator(
            data_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            callbacks=callbacks,
            workers=1
        )

        print('[Model] Training Completed. Model saved as %s' % save_fname)
        timer.stop()

    def predict (self, data):
        predicted = self.model.predict(data)
        return predicted

    def predict_point_by_point ( self, data ) :
        # Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
        print('[Model] Predicting Point-by-Point...')
        predicted = self.model.predict(data)
        predicted = np.reshape(predicted, (predicted.size,))
        return predicted

    def predict_sequences_multiple ( self, data, window_size, prediction_len ) :
        # Predict sequence of 50 steps before shifting prediction run forward by 50 steps
        print('[Model] Predicting Sequences Multiple...')
        prediction_seqs = []
        print(" IN PREDICT SEQUENCE MULTIPLE")
        for i in range(int(len(data) / prediction_len)) :
            #print(" I ", i)
            #print("prediction len", prediction_len)
            #print(" DATA SHAPE: ", data.shape)
            #print(" DATA LEN: ", len(data))
            #print(" DATA AT 0 ", data.iloc[0])
            curr_frame = data.iloc[i * prediction_len]
            #print(" GOT CURRENT FRAME!")
            predicted = []
            for j in range(prediction_len) :
                predicted.append(self.model.predict(curr_frame[newaxis, :, :])[0, 0])
                curr_frame = curr_frame[1 :]
                curr_frame = np.insert(curr_frame, [window_size - 2], predicted[-1], axis=0)
            prediction_seqs.append(predicted)
        return prediction_seqs

    def predict_sequence_full ( self, data, window_size ) :
        # Shift the window by 1 new prediction each time, re-run predictions on new window
        print('[Model] Predicting Sequences Full...')
        curr_frame = data[0]
        predicted = []
        for i in range(len(data)) :
            predicted.append(self.model.predict(curr_frame[newaxis, :, :])[0, 0])
            curr_frame = curr_frame[1 :]
            curr_frame = np.insert(curr_frame, [window_size - 2], predicted[-1], axis=0)
        return predicted