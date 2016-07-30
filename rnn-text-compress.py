#!/usr/bin/python2.7
# Based off of lstm_text_generation.py from https://github.com/fchollet/keras/tree/master/examples 
from __future__ import print_function
from keras.layers import Dense, Activation, Dropout, GRU, Merge
from keras.utils.data_utils import get_file
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from subprocess import Popen, PIPE
import matplotlib.pyplot as plt
from os.path import basename
from os import devnull
from sys import argv
import numpy as np
import random
import shlex
import glob
import time
import sys
import re

# Cython imports:
from vectorize import vectorize_input

'''
TODO: 
    interface with arithmetic coding
    decompression
'''

# Configuration:
step = 3 # sliding window step
maxlen = 40 # length of sliding window
prev_weights_file_path = '' # path to pre-trained weights (optional)

def argcheck():
    if len(argv) < 2:
        usage(); exit()
    return

def usage():
    print('usage: rnn-text-compress.py < flag > < input file path > < weights directory (-e,-p) >')
    print('flags:')
    print('  -t    train model')
    print('  -e    evaluate a single model')
    print('  -p    evaluate and plot results for a directory of models')
    return

def conll_parse(conll_data):
    text, text_tags = [], []
    conll_data = [l.split() for l in conll_data.split('\n') if len(l) > 0]
    words, pos_tags = [l[1] for l in conll_data], [l[4] for l in conll_data]
    for word_tag_pair in zip(words, pos_tags):
        for char in word_tag_pair[0]:
            text.append(char)
            text_tags.append(word_tag_pair[1])
        text.append(' ')
        text_tags.append('S')
    return text, text_tags

# Use Google's SyntaxNet to perform POS tagging on text
def get_pos_tags(path):
    print('Running SyntaxNet')
    cmd1       = 'cat {}'.format(path)
    cmd2       = 'docker run -i davidcox143/conll-format-syntaxnet --rm'
    cmd1, cmd2 = [shlex.split(cmd) for cmd in [cmd1, cmd2]]
    p1         = Popen(cmd1, stdout=PIPE)
    p2         = Popen(cmd2, stdin=p1.stdout, stdout=PIPE, stderr=open(devnull, 'wb'))
    conll_data, status = p2.communicate()
    if status != None: 
        print('Error getting POS data from SyntaxNet. Exiting...'); exit()
    return conll_parse(conll_data)

# cut text into semi-redundant sequences of maxlen characters
def cut_text(text, text_tags):
    sentences, sentence_tags, next_chars = [], [], []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        sentence_tags.append(text_tags[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    print('total sequences:', len(sentences))
    return sentences, sentence_tags, next_chars

# build the model
def build_model(chars, tags):
    print('Building model...')
    char_encoder = Sequential(name='char_encoder')
    pos_encoder  = Sequential(name='pos_encoder')
    char_encoder.add(
            GRU(
                output_dim=256, 
                return_sequences=True, 
                input_shape=(maxlen, len(chars)), 
                consume_less='gpu',
                )
            )
    char_encoder.add(Dropout(0.1))
    pos_encoder.add(
            GRU(
                output_dim=49, 
                return_sequences=True, 
                input_shape=(maxlen, len(tags)), 
                consume_less='gpu',
                )
            )
    pos_encoder.add(Dropout(0.1))
    decoder = Sequential(name='decoder')
    decoder.add(Merge([char_encoder, pos_encoder], mode='concat'))
    decoder.add(
            GRU(
                output_dim=305, 
                return_sequences=False, 
                consume_less='gpu',
                ) 
            )
    decoder.add(Dense(len(chars), activation='relu'))
    decoder.add(Dense(len(chars), activation='softmax'))
    decoder.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return decoder

# train the model (for debugging only. use weights file instead.)
def train(model, X_chars, X_pos, y, path, num_iterations=150):
    start_t = str(int(time.time()))
    for i in xrange(num_iterations / 10):
        print('Iteration:', i)
        checkpoint_name = '_'.join([basename(path), start_t, str((i+1)*10), 'weights.hdf5'])
        checkpointer = ModelCheckpoint(filepath=checkpoint_name)
        model.fit([X_chars, X_pos], y,
                  batch_size=512,
                  nb_epoch=10, 
                  callbacks=[checkpointer])  
    return model

# evaluate the model by calculating the percentage of correctly predicted chars 
def evaluate_model(                                                             
    model, text_tags, chars, char_indices,                                      
    indices_char, tag_indices, sentences, next_chars):                          
        print('Evaluating model...')
        num_correct, num_missed = 0., 0.                                        
        accuracy = lambda x, y: x / (x + y)                                     
        for sentence, next_char in zip(sentences, next_chars):                  
            x_chars = np.zeros((1, maxlen, len(chars)))                         
            x_pos = np.zeros((1, maxlen, len(tag_indices)))                     
            for t, char in enumerate(sentence):                                 
                x_chars[0, t, char_indices[char]] = 1.                          
                x_pos[0, t, tag_indices[text_tags[t]]] = 1.                     
            preds = model.predict([x_chars, x_pos], verbose=0)[0]               
            predicted_next_char = indices_char[preds.argmax()]                  
            if predicted_next_char == next_char: num_correct += 1               
            else: num_missed += 1                                               
        print('Accuracy:', accuracy(num_correct, num_missed))              
        return accuracy(num_correct, num_missed)      

def plot_results(path, results): 
    model_name = re.split('\w+)\.',path)[1]
    epochs = [10*x for x in range(1,len(results)+1)]
    plt.plot(epochs, results)
    plt.xlabel('epochs')
    plt.ylabel('accuracy (%)')
    plt.title('Model accuracy over time. Input: {}'.format(model_name))
    plt.grid(True)
    plt.savefig('{}.png'.format(model_name))
    return

def main():
    argcheck()
    flag = argv[1] 
    path = argv[2] 
    if flag == '-e' or flag == '-p': 
        global step
        step *= 100 # increase step size to keep evaluation time reasonable 

    # Initialization: 
    text, text_tags = get_pos_tags(path)
    tags            = set([tag for tag in text_tags])
    tag_indices     = {t:i for i, t in enumerate(tags)}
    chars           = [chr(x) for x in range(0, 256)]
    char_indices    = {c:i for i, c in enumerate(chars)}
    indices_char    = {i:c for i, c in enumerate(chars)}

    print('corpus length:', len(text))
    print('total chars:', len(chars))
    print('total pos tags:', len(tags))

    # read in our input file, split into characters and sentences
    sentences, sentence_tags, next_chars = cut_text(text, text_tags)

    # vectorize our input data
    X_chars, X_pos, y = vectorize_input(sentences, char_indices, chars, 
                                next_chars, sentence_tags, tag_indices, maxlen)

    # build the GRU network
    decoder = build_model(chars, tags) 

    # load weights from a previous run 
    if prev_weights_file_path != '':
        decoder.load_weights(prev_weights_file_path)

    # Training:

    # train a model
    if flag == '-t':
        train(decoder, X_chars, X_pos, y, path, 150)

    # Evaluation:

    # switch path to weights
    if len(argv) != 4: 
        usage(); exit()
    path = argv[3]

    # evaluate a single model
    if flag == '-e':
        decoder.load_weights(path)
        evaluate_model(decoder, text_tags, chars, char_indices, indices_char, 
                       tag_indices, sentences, next_chars)

    # batch evaluate and plot
    if flag == '-p':
	results = []                                                                   
	natural = lambda x: [int(c) if c.isdigit() else c for c in re.split('(\d+)', x)] 
        weight_paths = sorted(glob.glob(path+'*.hdf5'), key=natural)
        print('total weight files:', len(weight_paths))
	for weight_path in weight_paths:
            decoder.load_weights(weight_path)
            results.append(evaluate_model(decoder, text_tags, chars, char_indices, 
                indices_char, tag_indices, sentences, next_chars))
        plot_results(path, results)

    if flag not in ['-t', '-e', '-p']: usage()

if __name__ == '__main__':
    main()
