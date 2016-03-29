import numpy as np
import os

def read_data(transcript_file, semantic_file):
    spliter = '=>'
    corpus = {}
    count = 0
    with open(transcript_file, 'rt') as f:
        for line in f:
            line = line.strip()
            id_num, text = line.split(spliter)
            id_num = id_num.strip()
            text = text.strip()
            corpus[id_num] = {'text': text}
            count +=1

    with open(semantic_file, 'rt') as f:
        for line in f:
            line = line.strip()
            id_num, text = line.split(spliter)
            id_num = id_num.strip()
            text = text.strip()
            corpus[id_num]['sem'] = text
    
    return corpus, count
        
def add_unigram_from_content(unigram, content, word_standardizer=None):
    words = content.strip().lower().split()
    
    for word in words:
        if word_standardizer is not None:
            word = word_standardizer(word)
        if word not in unigram:
            unigram[word] = 1
        else:
            unigram[word] += 1

    return unigram

def get_unigram(corpus):
    unigram = {}
    for id_num in corpus:
        add_unigram_from_content(unigram, corpus[id_num]['text']) 
   
    return unigram 
   
def inverse_unigram(unigram):
    inv_unigram = {}
    for w1 in unigram:
        count = unigram[w1]
        if count not in inv_unigram:
            inv_unigram[count] = []
        inv_unigram[count].append(w1)
    return inv_unigram

def get_top_unigram(unigram, occur_above=None, top_n=None):
    inv_unigram = inverse_unigram(unigram)
    count = 0
    new_unigram = {}
    for order in sorted(inv_unigram.keys(), reverse=True):
        if occur_above is not None and order<occur_above:
            return new_unigram, count

        print '----order=%d, len=%d'%(order, len(inv_unigram[order])), inv_unigram[order][0:10]

        for w1 in inv_unigram[order]:
            new_unigram[w1] = order
            count += 1
            if top_n is not None and count >= top_n:
                return new_unigram, count

    return new_unigram, count#couldn't get enough topn

def format_vector(unigram):
    to_index = {}
    count = 0
    for w1 in unigram:
        to_index[w1] = count
        count +=1

    return to_index, count

def count_unigram(to_index, vector_size, text, word_standardizer=None):
    #feature extractor
    features = [0]* (vector_size) #last one for oov
    #features = np.ndarray((1, vector_size)
    
    words = text.strip().lower().split()
    for word in words:
        if word_standardizer is not None:
            word = word_standardizer(word)

        index = vector_size-1
        if word in to_index:
            index = to_index[word]

        features[index] +=1

    return features

def inform_detector(semantic):
    if semantic.find('inform')>=0:
        return True
    return False
def inform_stop_detector(semantic):
    if semantic.find('inform')>=0 and semantic.find('from_top')>=0:
        return True
    return False
    

def build_data(unigram, corpus, num, label_detector):
    to_index, vector_size = format_vector(unigram)
    vector_size += 1#plus one for oov
    data = np.zeros(shape=(num, vector_size))
    labels = np.zeros(shape=(num,2))
    for idx, id_num in enumerate(corpus):
        data[idx] = count_unigram(to_index, vector_size, corpus[id_num]['text'])
        if label_detector(corpus[id_num]['sem']):
            labels[idx] = [1.0, 0.0]
        else:
            labels[idx] = [0.0, 1.0]
    
    return data, labels


class DataSet(object):
    def __init__(self, data):
        images = data['data']
        labels = data['labels']
        assert images.shape[0] == labels.shape[0], (
                "images.shape: %s labels.shape: %s" % (images.shape,
                                                       labels.shape))
        self._num_examples = images.shape[0]
        # Convert from [0, 255] -> [0.0, 1.0].
        #images = images.astype(numpy.float32)
        #images = numpy.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


def read_data_sets(data_path, use_unigram=40):
    train_transcript = os.path.join(data_path, 'train.trn')
    train_semantic = os.path.join(data_path, 'train.asr.hdc.sem')
    valid_transcript = os.path.join(data_path, 'dev.trn')
    valid_semantic = os.path.join(data_path, 'dev.asr.hdc.sem')
    test_transcript = os.path.join(data_path, 'test.trn')
    test_semantic = os.path.join(data_path, 'test.asr.hdc.sem')

    corpus = {}
    corpus['train'], train_count = read_data(train_transcript, train_semantic)
    corpus['valid'], valid_count = read_data(valid_transcript, valid_semantic)
    corpus['test'], test_count = read_data(test_transcript, test_semantic)

    unigram = get_unigram(corpus['train'])

    _, count  = get_top_unigram(unigram, occur_above=0)
    print 'total unigram', count
    top_n = float(use_unigram)*count/100
    unigram, count= get_top_unigram(unigram, top_n=top_n)
    print 'total unigram used', count

    vcorpus = {'train':{}, 'valid':{}, 'test':{}}
    data, labels = build_data(unigram, corpus['train'], train_count, inform_stop_detector)
    vcorpus['train']['data'] = data
    vcorpus['train']['labels'] = labels
    data, labels = build_data(unigram, corpus['valid'], valid_count, inform_stop_detector)
    vcorpus['valid']['data'] = data
    vcorpus['valid']['labels'] = labels
    data, labels = build_data(unigram, corpus['test'], test_count, inform_stop_detector)
    vcorpus['test']['data'] = data
    vcorpus['test']['labels'] = labels
    
    class DataSets(object):
        pass

    datasets = DataSets()
    datasets.train = DataSet(vcorpus['train'])
    datasets.valid = DataSet(vcorpus['valid'])
    datasets.test = DataSet(vcorpus['test'])

    return datasets
    
if __name__ == '__main__':
    datasets = read_data_sets('./data')
