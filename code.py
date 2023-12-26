!pip install mitdeeplearning
import tensorflow as tf
import mitdeeplearning as mdl
import numpy as np
import os
import time
from IPython import display as ipythondisplay
from tqdm import tqdm
songs = mdl.lab1.load_training_data()
!apt-get -y install abcmidi timidity
example_song = songs[0]
print("\nExample song: ")
print(example_song)
waveform = mdl.lab1.play_song(example_song)
if waveform:
   ipythondisplay.display(waveform)
songs_joined = "\n\n".join(songs)

# Find all unique characters in the joined string
vocab = sorted(set(songs_joined))
print("There are", len(vocab), "unique characters in the dataset")
vocab
char_to_index = dict((j,i) for i,j in enumerate(vocab))
print(char_to_index)
index_to_char = dict((i,j) for i,j in enumerate(vocab))
print(index_to_char)
print('{')
for char,_ in zip(char_to_index, range(20)):
    print('  {:4s}: {:3d},'.format(repr(char), char_to_index[char]))
print('  ...\n}')
def vectorize_string(string):
    vector = []
    for char in string:
#         print(type(char))
        vector .append(char_to_index[char])
    return np.array(vector)

vectorized_songs = vectorize_string(songs_joined)
# print(vectorized_songs.shape)
print ('{} ---- characters mapped to int ----> {}'.format(repr(songs_joined[:10]), vectorized_songs[:10]))
# check that vectorized_songs is a numpy array
assert isinstance(vectorized_songs, np.ndarray), "returned result should be a numpy array"
def get_batch(vectorized_songs, seq_length, batch_size):
    # the length of the vectorized songs string
    n = vectorized_songs.shape[0] - 1
    # randomly choose the starting indices for the examples in the training batch
    idx = np.random.choice(n-seq_length, batch_size)
    input_batch = [vectorized_songs[i : i+seq_length] for i in idx]
    '''TODO: construct a list of output sequences for the training batch'''
    output_batch = [vectorized_songs[i+1 : i+seq_length+1] for i in idx]

    # x_batch, y_batch provide the true inputs and targets for network training
    x_batch = np.reshape(input_batch, [batch_size, seq_length])
    y_batch = np.reshape(output_batch, [batch_size, seq_length])
    return x_batch, y_batch
test_args = (vectorized_songs, 10, 2)
if not mdl.lab1.test_batch_func_types(get_batch, test_args) or \
   not mdl.lab1.test_batch_func_shapes(get_batch, test_args) or \
   not mdl.lab1.test_batch_func_next_step(get_batch, test_args):
    print("======\n[FAIL] could not pass tests")
else:
    print("======\n[PASS] passed all tests!")
x_batch, y_batch = get_batch(vectorized_songs, seq_length=5, batch_size=1)

for i, (input_idx, target_idx) in enumerate(zip(np.squeeze(x_batch), np.squeeze(y_batch))):
    print("Step {:3d}".format(i))
    print("  input: {} ({:s})".format(input_idx, repr(index_to_char[input_idx])))
    print("  expected output: {} ({:s})".format(target_idx, repr(index_to_char[target_idx])))
def LSTM(rnn_units):
    return tf.keras.layers.LSTM(rnn_units, return_sequences=True, recurrent_initializer='glorot_uniform',
                                recurrent_activation='sigmoid',stateful=True,)
def build_model_1(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
    # Layer 1: Embedding layer to transform indices into dense vectors
    #   of a fixed embedding size
    tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),

    # Layer 2: LSTM with `rnn_units` number of units.
    # TODO: Call the LSTM function defined above to add this layer.
    LSTM(rnn_units),
    # LSTM('''TODO'''),

    # Layer 3: Dense (fully-connected) layer that transforms the LSTM output
    #   into the vocabulary size.
    # TODO: Add the Dense layer.
    tf.keras.layers.Dense(vocab_size)
    # '''TODO: DENSE LAYER HERE'''
    ])
    return model

# Build a simple model with default hyperparameters. You will get the
#   chance to change these later.
model = build_model_1(len(vocab), embedding_dim=256, rnn_units=1024, batch_size=32)
model.summary()
x, y = get_batch(vectorized_songs, seq_length=100, batch_size=32)
pred = model(x)
print("Input shape:      ", x.shape, " # (batch_size, sequence_length)")
print("Prediction shape: ", pred.shape, "# (batch_size, sequence_length, vocab_size)")
def compute_loss(labels, logits):
    loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True) # TODO
    return loss

'''TODO: compute the loss using the true next characters from the example batch
    and the predictions from the untrained model several cells above'''
example_batch_loss = compute_loss(y, pred) # TODO

print("Prediction shape: ", pred.shape, " # (batch_size, sequence_length, vocab_size)")
print("scalar_loss:      ", example_batch_loss.numpy().mean())
#Optimization parameters
num_training_iterations = 10000
batch_size = 4
seq_length = 100
learning_rate = 5e-3

# Model parameters:
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024  # Experiment between 1 and 2048

# Checkpoint location:
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "my_ckpt")
model = build_model_1(vocab_size, embedding_dim, rnn_units, batch_size)
optimizer = tf.keras.optimizers.Adam()
def train_step(x,y):
    with tf.GradientTape() as tape:
        y_pred = model(x)
        loss = compute_loss(y, y_pred)
    grad = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grad, model.trainable_variables))
    return loss.numpy().mean()
losses = []
for i in tqdm(range(0,num_training_iterations)):
    x, y = get_batch(vectorized_songs, seq_length, batch_size)
    loss = train_step(x,y)
    losses.append(loss)

    if i % 100 == 0:
        model.save_weights(checkpoint_prefix)

model.save_weights(checkpoint_prefix)
import matplotlib.pyplot as plt
plt.plot(losses)
model = build_model_1(vocab_size, embedding_dim, rnn_units, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

model.summary()
def generate_text(model, start_string, generation_length=1000):
    input_eval = [char_to_index[i] for i in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []
    model.reset_states()
    tqdm._instances.clear()

    for i in tqdm(range(generation_length)):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
#         print(predicted_id)
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(index_to_char[predicted_id])
    return (start_string + ''.join(text_generated))
generated_text = generate_text(model, start_string="a", generation_length=1000000)
print(generated_text[:1000])
generated_songs = mdl.lab1.extract_song_snippet(generated_text)
m = 0;
for i, song in enumerate(generated_songs):
  # Synthesize the waveform from a song
    waveform = mdl.lab1.play_song(song)
    m+=1
  # If its a valid song (correct syntax), lets play it!
    if waveform:
        print("Generated song", i)
        ipythondisplay.display(waveform)
        if m>3:
            break
