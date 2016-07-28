import cython
import numpy as np
cimport numpy as np

# vectorize our training data
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def vectorize_input(sentences, char_indices, chars, next_chars, sentence_tags, tag_indices, max_len):
    print('Vectorization...')                                                   
    cdef:
        int maxlen = max_len
        int i, t, offset
    '''
    Notation:
        X_chars: [ [sentence index], [intra-sentence index], [character index] ]
        X_pos:   [ [sentence index], [intra-sentence index], [POS tag index] ]
        y:       [ [sentence index], [character index] ]                                
    '''
    cdef np.ndarray X_chars = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.int16)     
    cdef np.ndarray X_pos = np.zeros((len(sentences), maxlen, len(tag_indices)), dtype=np.int16) 
    cdef np.ndarray y = np.zeros((len(sentences), len(chars)), dtype=np.int16)                   
    cdef short [:, :, :] X_chars_view = X_chars
    cdef short [:, :, :] X_pos_view = X_pos
    cdef short [:, :]    y_view = y
    for i, sentence in enumerate(sentences):                                    
        for t, char in enumerate(sentence):                                     
            tag = sentence_tags[i][t]
            X_chars_view[i, t, <int>char_indices[char]] = 1                           
            X_pos_view[i, t, <int>tag_indices[tag]] = 1 
        y_view[i, <int>char_indices[next_chars[i]]] = 1                               
    return X_chars, X_pos, y
