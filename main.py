from libs.datalib import *
from libs.mcmclib import *
import string
import os

#INFILE = "data/corpus/war-and-peace.txt"
#OUTFILE = "data/corpus/war-and-peace-clean.txt"
#cleaner(INFILE, OUTFILE)

IN_CIPHERTEXT = "data/test/j_5.txt"
OUT_DECODED = "./results/" + IN_CIPHERTEXT.split("/")[-1]
BETA = 0.5
ciphertext = read_ciphertext(IN_CIPHERTEXT)
STATE_SPACE = list(string.ascii_lowercase) + [' ']
CONVERGENCE = 5000

def make_letter_dict(state_space):
    dict = {}
    for i, letter in enumerate(state_space):
        dict[letter] = i
    
    return dict

letter_dict = make_letter_dict(STATE_SPACE)


char_counts = np.zeros(27)
digram_counts = np.zeros((27, 27))
CLEAN_DIR = './data/corpus/cleaned/'
for clean in os.listdir(CLEAN_DIR):
    cur_char_counts, cur_digram_counts = get_counts(CLEAN_DIR + clean, letter_dict)
    char_counts += cur_char_counts
    digram_counts += cur_digram_counts

char_freqs = get_frequencies(char_counts)
transition_matrix = get_transition_matrix(digram_counts)

#char_freqs = get_softmax_frequencies(char_counts)
#transition_matrix = get_softmax_transition(digram_counts)

def naive_forward_mcmc(ciphertext, iterations):
    permutation = Permutation()
    convergence_count = 0
    
    for iteration in range(iterations):
        if iteration % 1000 == 0:
            print("Iteration", iteration)
            print(decode_ciphertext(ciphertext, permutation, length = 100))

        new_permutation = permutation.iterate()
        likelihood_cur = naive_likelihood(ciphertext, permutation, char_freqs, transition_matrix, letter_dict)
        likelihood_new = naive_likelihood(ciphertext, new_permutation, char_freqs, transition_matrix, letter_dict)

        prob_accept = np.exp(-BETA * max(likelihood_new - likelihood_cur, 0))

        if random.random() < prob_accept:
            permutation = new_permutation
            convergence_count = 0
        else:
            convergence_count += 1
        
        if convergence_count >= CONVERGENCE:
            break;
    
    return permutation, naive_likelihood(ciphertext, permutation, char_freqs, transition_matrix, letter_dict)

def fast_forward_mcmc(iterations):
    ciphertext_counts = precomp_ciphertext(IN_CIPHERTEXT, letter_dict)
    permutation = Permutation()
    convergence_count = 0
    
    likelihood_cur = fast_likelihood(ciphertext_counts, permutation, transition_matrix, letter_dict)
    
    for iteration in range(iterations):
        if iteration % 1000 == 0:
            print("Iteration", iteration)
            print(decode_ciphertext(ciphertext, permutation, length = 100))

        new_permutation = permutation.iterate()
        likelihood_new = fast_likelihood(ciphertext_counts, new_permutation, transition_matrix, letter_dict)

        prob_accept = np.exp(-BETA * max(likelihood_new - likelihood_cur, 0))

        if random.random() < prob_accept:
            permutation = new_permutation
            convergence_count = 0
            likelihood_cur = likelihood_new
        else:
            convergence_count += 1
        
        if convergence_count >= CONVERGENCE:
            break;
    
    return permutation, likelihood_cur

def full_mcmc(iterations, each_iterations=50000):
    best_energy = float('inf')
    best_permutation = Permutation()
    for i in range(iterations):
        print("Try", i + 1)
        permute, energy = fast_forward_mcmc(each_iterations)
        if(energy < best_energy):
            best_energy = energy
            best_permutation = permute
    
    print("Best permutation found:", best_permutation.map_dict)
    print("Sample decoded:", decode_ciphertext(ciphertext, best_permutation, length = 100))
    return best_permutation


#final_permute = naive_forward_mcmc(ciphertext, 25000)
#final_permute, energy = fast_forward_mcmc(25000)
final_permute = full_mcmc(10, each_iterations=25000)

print("Writing to", OUT_DECODED)
with open(OUT_DECODED, 'w+', encoding='utf8') as out:
    out.write("Final permutation: \n" + final_permute.to_string() + "\n")
    out.write("Scrambled text: \n" + ciphertext + "\n")
    out.write("Unscrambled text: \n" + decode_ciphertext(ciphertext, final_permute))
    






