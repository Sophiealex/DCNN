from collections import Counter
import itertools
import numpy as np
import re


from gensim import models
from collections import OrderedDict

def transform_labels(y):
    return np.array([int(s) for s in y.split('\t')])

punct = set(u''':!),:;?]}¢'"、。〉》」』】〕〗〞︰︱︳﹐､﹒
﹔﹕﹖﹗﹚﹜﹞！），．：；？｜｝︴︶︸︺︼︾﹀﹂﹄﹏､～￠
々‖•·ˇˉ―--′’”([{£¥'"‵〈《「『【〔〖（［｛￡￥〝︵︷︹︻
︽︿﹁﹃﹙﹛﹝（｛“‘-—_…''')

filterpunt = lambda s: ''.join(filter(lambda x: x not in punct, s))

def load_data_and_labels():
    """
    Loads data from files, splits the data into words
    Returns split sentences
    """
    # Load data from files
    folder_prefix = 'data/'
    x_train = list(open(folder_prefix+"train_only_x.txt",encoding='utf-8').readlines())
    x_test = list(open(folder_prefix+"test_only_x.txt",encoding='utf-8').readlines())
    test_size = len(x_test)
    x_text = x_train + x_test
    x_text = [sent.strip('\n').strip() for sent in x_text] # remove last '\n' and ' '
    x_text = [filterpunt(sent) for sent in x_text] # remove punct
    x_text = [re.sub(' +', ' ', sent) for sent in x_text] # replace ' +' by ' '
    x_text = [s.split(' ') for s in x_text] # split sentence by ' '
    return [x_text, test_size]

def pad_sentences(sentences, padding_word=" "):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences

def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    # vocabulary_inv=['<PAD/>', 'the', ....]
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    # vocabulary = {'<PAD/>': 0, 'the': 1, ',': 2, 'a': 3, 'and': 4, ..}
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]

def build_input_data(sentences, vocabulary):
    """
    Maps sentences to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    return x

def load_data():
    """
    Loads and preprocessed data
    Returns input vectors, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, test_size = load_data_and_labels()
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x = build_input_data(sentences_padded, vocabulary)
    return [x, vocabulary, vocabulary_inv, test_size]

def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(list(data))
    data_size = len(np.atleast_1d(data))
    num_batches_per_epoch = int(len(np.atleast_1d(data))/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = (batch_num + 1) * batch_size
            if end_index > data_size:
                end_index = data_size
                start_index = end_index - batch_size
            yield shuffled_data[start_index:end_index]

def get_wordembedding(vocabulary):
    '''
    Generates wordembedding matrix for words.
    return size is [len(vocabulary), embed_dim]
    '''
    # merge48.model.bin 是预先训练好的word2vec模型 由word2vec-canwork/train.py 得到
    model = models.Word2Vec.load('data/merge48.model.bin')
    ordered_v = OrderedDict(vocabulary)
    bar = OrderedDict(sorted(ordered_v.items(), key=lambda x: x[1]))
    wordembedding = []
    for words, index in bar.items():
        if words == '':
            wordembedding.append(np.zeros(48))
        elif words == ' ':
            wordembedding.append(np.zeros(48))
        else:
            vec = model[words]
            wordembedding.append(np.array(vec))
    return np.array(wordembedding).astype(np.float32)