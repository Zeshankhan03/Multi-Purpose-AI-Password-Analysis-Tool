from collections import defaultdict
import random

def load_dataset(filename, encoding="utf-8"):
  """Loads a list of words from a UTF-8 encoded text file.

  Args:
    filename: The path to the text file containing the dataset.
    encoding: The encoding of the text file (default: "utf-8").

  Returns:
    A list of words from the text file.
  """

  with open(filename, 'r', encoding=encoding) as file:
    words = file.read().splitlines()  # Read lines and split into words
  return words

def train_markov_chain(words, order=2):
  """Trains a Markov chain on a list of words.

  Args:
    words: A list of words to train the model on.
    order: The order of the Markov chain (default: 2). This determines
           how many preceding words are considered when predicting the next word.

  Returns:
    A dictionary representing the Markov chain. Keys are tuples of length `order`
    representing the current state (preceding words), and values are dictionaries
    mapping following words to their probabilities.
  """

  markov_chain = defaultdict(lambda: defaultdict(int))
  for i in range(len(words) - order):
    current_state = tuple(words[i:i + order])  # Get the current state (order words)
    next_word = words[i + order]  # Get the next word
    markov_chain[current_state][next_word] += 1  # Count occurrences

  # Normalize probabilities for each state
  for state, following_words in markov_chain.items():
    total_count = sum(following_words.values())
    for word, count in following_words.items():
      following_words[word] = count / total_count  # Probability of following word

  return markov_chain

def generate_text(markov_chain, start_words, length=100):
  """Generates new text using the trained Markov chain.

  Args:
    markov_chain: The trained Markov chain model.
    start_words: A list of words to start the generation (default: first order words).
    length: The desired length of the generated text (default: 100 words).

  Returns:
    A list of generated words.
  """

  generated_text = start_words[:]
  for _ in range(length):
    current_state = tuple(generated_text[-len(start_words):])
    if current_state in markov_chain:  # Check if state exists in the model
      available_words = list(markov_chain[current_state].keys())
      next_word = random.choices(available_words, weights=markov_chain[current_state].values())[0]
    else:
      # Choose a random word from all words if the state is unseen
      all_words = list(set(word for sentence in markov_chain.keys() for word in sentence))  # Get all unique words
      next_word = random.choice(all_words)
    generated_text.append(next_word)

  return generated_text

def generate_similar_word(markov_chain, seed_word, order=2):
  """Generates a word similar to the seed word based on the Markov chain.

  Args:
    markov_chain: The trained Markov chain model.
    seed_word: The word to use as a seed for generating a similar word.
    order: The order of the Markov chain used for similarity (default: 2).

  Returns:
    A word similar to the seed word, or None if no similar word is found.
  """

  # Get all states (preceding word sequences) that lead to the seed word
  possible_states = [state for state in markov_chain.keys() if state[-1] == seed_word]

  # If no states lead to the seed word, return None
  if not possible_states:
    return None

  # Choose a random state that leads to the seed word
  random_state = random.choice(possible_states)

  # Generate a new word based on the random state (excluding the seed word)
  new_word = random.choices(list(markov_chain[random_state].keys()), weights=list(markov_chain[random_state].values()))[0]
  
