# Image_to_Text_Extraction

### **Problem statement**: 
*we are provided with images and need to recognise the text from an image. the images may be noisy and unable to extract but we need make some assumptions and use statistical properities of the language to figure it out.

### **classification**:

*simple : In this case each column in emission matrix/observation is dependent on its states,here we comapared the test letters using pixels and the output are the one's with most praobable matching case.

*HMM_viterbi: The viterbi approach is used to calculate the most probable hidden state in every iteration.

### **Approach**:

*To start with the problem we calculated initial probabilities from the part 1 training text data file.i.e., to calculate probability of starting letter of a word w.r.t total number of words in the given data.
(P(ini) = count of letter starting in a word/total no. of words)

*Then we calculated transmission to find the probability for a given initial letter followed by previous letter. 
(P(i/i-1) = count of letter i-1 came after i/total no. of i-1)

*above two probabilities are extracted from using given Train_letters and training text data file.
Now we to find the emission probabilities from the test image repective letter but the test image is not text and in the form of pixels where each letter is of fixed size i.e.,14 x 25(character_width x character_height).
here the pixels are either black or white i.e., * or ''.

*after pixels here comes the next parameter noise, as we assume the after many iterations finally approached to 42.5 by binary search method. for noise above 0.5 the letter are being observed as number only and after iterations we found it to be optimum.
if we assume m% of pixels to be noisy then as per bayes classifier corresponding pixel is (100-m)%.
 
*so we calculated the emission probability as P(observed|char) = (non-noisy pixels)**(count of matched pixels) * (noisy pixels)**(total - matched pixels)

### ** HMM with viterbi:
* as we already calculated transmission probabilities which indicating the transition of letters from one state to another.
*create an vit_table/matrix of zeros of dimensions len(train letters) and len(test letters) and put the inital prob values in the first column. 
*here we wrote a standard viterbi function which includes created a viterbi table and also using backtracking to find the most probable path.
