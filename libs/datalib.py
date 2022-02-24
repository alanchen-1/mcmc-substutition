import re
import numpy as np
def read_ciphertext(filepath):
    open_file = open(filepath, 'r', encoding='utf8')
    text = open_file.readline()
    open_file.close()
    return text


def cleaner(filepath, towritepath):
    unclean = open(filepath, 'r', encoding="utf8")
    clean = open(towritepath, 'w+', encoding="utf8")

    full = unclean.readlines()

    for line in full:
        lower = line.lower()
        cleaned_line = re.sub('[^a-z ]', '', lower, flags=re.UNICODE).strip()
        clean.write(cleaned_line)
        if len(cleaned_line) > 0 and cleaned_line[-1] != ' ':
            clean.write(' ')
    
    unclean.close()
    clean.close()

def get_counts(filepath, letter_dict):
    text = open(filepath, 'r', encoding="utf8")

    sz = len(letter_dict)
    char_counts = np.zeros(sz)
    digram_counts = np.zeros((sz, sz))
    full = text.readlines()

    for line in full:
        for i, char in enumerate(line):
            cur_index = letter_dict.get(char)
            char_counts[cur_index] += 1

            if i < (len(line) - 1):
                next_index = letter_dict.get(line[i + 1])
                digram_counts[cur_index][next_index] += 1

    return char_counts, digram_counts

def get_frequencies(char_counts):
    total = np.sum(char_counts)
    freq = np.array(list(map(lambda cur_count: (cur_count / total if cur_count > 0 else np.exp(-20)), char_counts)))

    return freq

def get_transition_matrix(digram_counts):
    sz = len(digram_counts)
    transition_matrix = np.empty((sz, sz))
    for i, row in enumerate(digram_counts):
        transition_matrix[i] = get_frequencies(row)
    
    return transition_matrix

def get_softmax_frequencies(char_counts):
    mx = np.max(char_counts)
    e_x = np.exp(char_counts - mx)
    e_x = np.array(list(map(lambda cur_count: (np.exp(1 - mx) if cur_count == 0 else cur_count), e_x)))
    return e_x / np.sum(e_x)

def get_softmax_transition(digram_counts):
    sz = len(digram_counts)
    transition_matrix = np.empty((sz, sz))
    for i, row in enumerate(digram_counts):
        transition_matrix[i] = get_softmax_frequencies(row)
    
    return transition_matrix