import nltk
from nltk.util import ngrams
from nltk.probability import FreqDist

# Download required data (only first time)
nltk.download('punkt')

# Get tweet from user
tweet = input("Enter a tweet: ")

# Convert to lowercase and tokenize
tokens = nltk.word_tokenize(tweet.lower())

print("\nTokens:")
print(tokens)

# Generate n-grams
unigrams = list(ngrams(tokens, 1))
bigrams = list(ngrams(tokens, 2))
trigrams = list(ngrams(tokens, 3))

print("\nUnigrams:")
print(unigrams)

print("\nBigrams:")
print(bigrams)

print("\nTrigrams:")
print(trigrams)

# Frequency distribution
fd = FreqDist(tokens)

print("\nWord Frequencies:")
for word, freq in fd.items():
    print(word, ":", freq)

# Sample HMM Output
print("\nHMM Prediction (Sample)")
print("AI -> NOUN")
print("improves -> VERB")
print("technology -> NOUN")

print("\nComparison")
print("N-Gram Model: Shows word sequences and frequencies.")
print("HMM Model: Predicts the grammatical tag of each word.")