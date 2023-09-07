#!/usr/bin/python
#
# Perform optical character recognition, usage:
#     python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png
#
# Authors: Sripal reddy Nomula - srnomula, Harshini Mysore Narasimha Ranga - hmn, sanjana Agrawal - sanagra
# (based on skeleton code by D. Crandall, Oct 2020)
#

from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
import sys, math, numpy as np

CHARACTER_WIDTH = 14
CHARACTER_HEIGHT = 25
TRAIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "


class HMM:
    def __init__(self, train_img_fname, train_txt_fname, test_img_fname):

        self.noise = 0.4258
        self.train_file = train_img_fname
        self.test_file = test_img_fname
        self.data_train = []
        self.data_char = []
        self.len_train_letters = len(TRAIN_LETTERS)
        self.test_letters = self.load_letters(self.test_file)
        self.train_letters = self.load_training_letters(self.train_file)
        self.load_data(train_txt_fname)
        self.prob_start = defaultdict(int)
        self.prob_trans = np.zeros(shape=(self.len_train_letters, self.len_train_letters))
        self.length_train_img_letters = len(self.train_letters)
        self.lenth_test_img_letters = len(self.test_letters)
        self.prob_emissions = np.zeros(shape=(self.length_train_img_letters, self.lenth_test_img_letters))

    def load_data(self, train_fpath):
        with open(train_fpath, 'r') as f:
            for line in f:
                self.data_train += [line.split()]  # words
                self.data_char += [char for char in line]  # all the chars with spaces

    @staticmethod
    def load_letters(f_path):
        im = Image.open(f_path)
        px = im.load()
        (x_size, y_size) = im.size
        print(im.size)
        print(int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH)
        result = []
        for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
            result += [["".join(['*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg + CHARACTER_WIDTH)]) for y in
                        range(0, CHARACTER_HEIGHT)], ]
        return result

    def load_training_letters(self, fname):
        # TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
        letter_images = self.load_letters(fname)
        return {TRAIN_LETTERS[i]: letter_images[i] for i in range(0, self.len_train_letters)}

    def calculate_probabilities(self):

        # count probabilities
        words_counter = 0
        for word_arr in self.data_train:
            for each_word in word_arr:
                words_counter += 1
                start_char_temp = each_word[0]
                if start_char_temp in TRAIN_LETTERS:
                    # increase start prob
                    self.prob_start[TRAIN_LETTERS.index(start_char_temp)] += 1

        for key in self.prob_start:
            self.prob_start[key] /= words_counter

        # transition prob
        for ind, char in enumerate(self.data_char):
            try:
                next_char = self.data_char[ind + 1]
                if char in TRAIN_LETTERS and next_char in TRAIN_LETTERS:
                    self.prob_trans[TRAIN_LETTERS.index(char)][TRAIN_LETTERS.index(next_char)] += 1
            except:
                continue  # or break

        for ltr in range(self.len_train_letters):
            sum_total_char = np.sum(self.prob_trans[ltr])  # sum of counts for each letter
            for n in range(self.len_train_letters):

                if self.prob_trans[ltr, n]:
                    self.prob_trans[ltr, n] = math.log(self.prob_trans[ltr, n] / sum_total_char)
                else:
                    self.prob_trans[ltr, n] = math.log(1 / (sum_total_char + 2))

        for j in range(self.lenth_test_img_letters):
            for letter in self.train_letters:
                match = 0
                for m in range(25):
                    for n in range(14):
                        if self.train_letters[letter][m][n] == self.test_letters[j][m][n]:
                            match += 1  # the number of matched pixels
                self.prob_emissions[TRAIN_LETTERS.index(letter), j] = ((1 - self.noise) ** match) * (
                        self.noise ** (self.lenth_test_img_letters - match))

    def simple_model(self):
        result = [TRAIN_LETTERS[np.argmax(self.prob_emissions[:, i])] for i in range(self.lenth_test_img_letters)]
        return "".join(result)

    def viterbi_model(self):
        # inititalize prob comments
        ini_char = np.zeros(self.lenth_test_img_letters)
        vit_prob = np.zeros(shape=(self.len_train_letters, self.lenth_test_img_letters))
        temp = np.empty(shape=(self.len_train_letters, self.lenth_test_img_letters))

        for j in range(1, self.lenth_test_img_letters):
            for i in range(self.len_train_letters):
                check_list = []
                vit_prob[i, 0] = self.prob_start[i] + math.log(self.prob_emissions[i, 0])

                for k in range(self.len_train_letters):
                    check_list += [vit_prob[k, j - 1] + self.prob_trans[k, i] + math.log(self.prob_emissions[i, j])]
                temp[i, j] = np.argmax(check_list)
                vit_prob[i, j] = max(check_list)

        for i in range(self.lenth_test_img_letters)[::-1]:
            if i != 0:
                ini_char[i - 1] = temp[int(ini_char[i]), i]
            else:
                ini_char[-1] = np.argmax(np.array(vit_prob)[:, -1])

        return ini_char


# #####
# # main program
if len(sys.argv) != 4:
    raise Exception("Usage: python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png")

(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]

hmm_obj = HMM(train_img_fname, train_txt_fname, test_img_fname)
train_letters_img = hmm_obj.load_training_letters(train_img_fname)
test_letters = hmm_obj.load_letters(test_img_fname)

hmm_obj.calculate_probabilities()
simple_result = hmm_obj.simple_model()
hmm_result = hmm_obj.viterbi_model()
print("Simple: " + simple_result)
print("HMM: " + "".join([TRAIN_LETTERS[int(i)] for i in hmm_result]))

# The final two lines of your output should look something like this:
# print("Simple: " + "Sample s1mple resu1t")
# print("   HMM: " + "Sample simple result")
