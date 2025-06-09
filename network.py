import numpy as np
from tensorflow import keras
from collections import deque
import os

class SnakeAI:
    def __init__(self, buffer_size=15000, batch_size=64):
        self.gamma = 0.99
        self.input_size = 12
        self.output_size = 4
        self.hidden_size = 100    # 更大隐藏层
        self.discount_factor = 0.99

        self.model = self.build_model()
        self.target_model = self.build_model()
        self.model.compile(optimizer='adam', loss='mse')
        self.target_model.compile(optimizer='adam', loss='mse')

        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.last_loss = 0.0

    def build_model(self):
        model = keras.Sequential()
        model.add(keras.layers.Dense(self.hidden_size, activation='relu'))
        model.add(keras.layers.Dense(self.hidden_size, activation='relu'))
        model.add(keras.layers.Dense(self.hidden_size, activation='relu'))
        model.add(keras.layers.Dense(self.output_size, activation='linear'))
        return model

    def get_action(self, state):
        state = np.reshape(state, [1, self.input_size])
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def train_model(self):
        if len(self.buffer) < self.batch_size:
            self.last_loss = 0.0
            return

        batch_indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        batch = [self.buffer[idx] for idx in batch_indices]

        states = np.array([sample[0] for sample in batch])
        actions = np.array([sample[1] for sample in batch])
        rewards = np.array([sample[2] for sample in batch])
        next_states = np.array([sample[3] for sample in batch])
        dones = np.array([sample[4] for sample in batch])

        targets = rewards + self.gamma * np.amax(self.model.predict_on_batch(next_states), axis=1) * (1 - dones)
        target_vec = self.model.predict_on_batch(states)
        indexes = np.array([i for i in range(self.batch_size)])
        target_vec[[indexes], [actions]] = targets

        history = self.model.fit(states, target_vec, epochs=1, verbose=0)
        self.last_loss = history.history['loss'][0]

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def add_experience(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def load_best_weights(self, filepath='best_weights.weights.h5'):
        if os.path.exists(filepath):
            self.model.build(input_shape=(None, self.input_size))
            self.model.load_weights(filepath)
            print(f"Loaded best model weights from {filepath}.")

    def save_best_weights(self, filepath='best_weights.weights.h5'):
        self.model.save_weights(filepath)
        print(f"Saved best model weights to {filepath}.")
