import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences #padding for same number of column/rows

sentence=['i love my dog','i love my cat']
#tokenize the sentence (seperate using words)
tokenizer=Tokenizer(num_words=100,oov_token='<OOV>')
tokenizer.fit_on_texts(sentence)
word_index=tokenizer.word_index
#print(word_index)
#word sequence in the sentences
sequences=tokenizer.texts_to_sequences(sentence)
print(word_index)
print(sequences)
#padded=pad_sequences(sequences,padding="post" truncating="post" maxlen=5)