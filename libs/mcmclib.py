import numpy as np
import random
import json

from libs.datalib import get_counts

class Permutation:
    map_dict = {}
    inverse = {}
    def __init__(self, map_dict = None):
        if map_dict is None:
            base = list("abcdefghijklmnopqrstuvwxyz ")
            shuffled_base = list("abcdefghijklmnopqrstuvwxyz ")
            random.shuffle(shuffled_base)
            randomized = {}
            for i in range(len(shuffled_base)):
                randomized[base[i]] = shuffled_base[i]
            
            self.map_dict = randomized
            self.inverse = self.invert(randomized)
        else:
            self.map_dict = map_dict
            self.inverse = self.invert(map_dict)

    def invert(self, sigma):
        inverted = {}
        for key in sigma:
            inverted[sigma[key]] = key
        return inverted

    def encode(self, char):
        return self.map_dict[char]
    def decode(self, char):
        return self.inverse[char]

    # randomly swap two keys
    def iterate(self):
        sampled = random.sample(self.map_dict.keys(), 2)
        new_dict = self.map_dict.copy()
        temp = new_dict[sampled[0]]
        new_dict[sampled[0]] = new_dict[sampled[1]]
        new_dict[sampled[1]] = temp
        return Permutation(new_dict)
    
    def to_string(self):
        return json.dumps(self.inverse)
        

def decode_ciphertext(ciphertext, sigma, length = -1):
    ans = ""
    if length == -1:
        length = len(ciphertext)

    for i in range(length):
        ans += sigma.decode(ciphertext[i])

    return ans

def naive_likelihood(ciphertext, sigma, char_freqs, transition_matrix, letter_dict):
    plaintext = decode_ciphertext(ciphertext, sigma)
    likelihood = np.log(char_freqs[letter_dict[plaintext[0]]])
    for i in range(1, len(plaintext)):
        cur_char_index = letter_dict[plaintext[i]]
        prev_char_index = letter_dict[plaintext[i - 1]]
        likelihood += np.log(transition_matrix[prev_char_index][cur_char_index])
    
    return -1 * likelihood

def precomp_ciphertext(filepath, letter_dict):
    _, digram_counts = get_counts(filepath, letter_dict)
    return digram_counts

def fast_likelihood(ciphertext_matrix, sigma, transition_matrix, letter_dict, letter_array=list("abcdefghijklmnopqrstuvwxyz ")):
    likelihood = 0
    for i in range(len(ciphertext_matrix)):
        first_letter = letter_array[i]
        for j in range(len(ciphertext_matrix[i])):
            second_letter = letter_array[j]
            cur_count = ciphertext_matrix[i][j]

            index_decoded_first = letter_dict[sigma.decode(first_letter)]
            index_decoded_second = letter_dict[sigma.decode(second_letter)]
            likelihood += cur_count * np.log(transition_matrix[index_decoded_first][index_decoded_second])
    
    return - likelihood
 




