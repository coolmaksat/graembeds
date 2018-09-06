import numpy as np
import math


class Dataset(object):
    """Dataset class for handling data operations.
    It requires specific format for graph data files.
    Graph should be stored in `data` folder under
    its the folder with its name and should contain
    three files `train.txt`, `valid.txt`, `test.txt`.
    All three files should have the same format where
    each line represents a triple (head, relation, tail)
    """

    def __init__(self, name):
        self.name = name
        self.train_triples = []
        self.valid_triples = []
        self.test_triples = []
        self.node_mappings = {}
        self.relation_mappings = {}

        for triple in self.load('train'):
            self.train_triples.append(triple)
        for triple in self.load('valid'):
            self.valid_triples.append(triple)
        for triple in self.load('test'):
            self.test_triples.append(triple)

        self.train_triple_set = set(self.train_triples)
        self.triple_set = set(
            self.train_triples + self.valid_triples + self.test_triples)

        self.train_triples = np.array(self.train_triples)
        self.test_triples = np.array(self.test_triples)
        self.valid_triples = np.array(self.valid_triples)
        
        self.train_size = len(self.train_triples)
        self.valid_size = len(self.valid_triples)
        self.test_size = len(self.test_triples)
        self.nb_nodes = len(self.node_mappings)
        self.nb_relations = len(self.relation_mappings)
        
        
    def load(self, part):
        """Reads dataset files (train, valid, test) and
        generates triples with integer ids.
        If entity or relation is not in the mapping variables
        a new id is created and entity or relation is put
        to the corresponding mapping variable.
        
        Arguments:
          part: The name of dataset part (train, valid, test)

        """

        with open('data/' + self.name + '/' + part + '.txt') as f:
            for line in f:
                it = line.strip().split()
                if it[0] not in self.node_mappings:
                    self.node_mappings[it[0]] = len(self.node_mappings)
                if it[2] not in self.node_mappings:
                    self.node_mappings[it[2]] = len(self.node_mappings)
                if it[1] not in self.relation_mappings:
                    self.relation_mappings[it[1]] = len(self.relation_mappings)
                yield (
                    self.node_mappings[it[0]],
                    self.relation_mappings[it[1]],
                    self.node_mappings[it[2]])

    def batch_generator(self, part, batch_size=256):
        triples = getattr(self, part + '_triples')
        n = len(triples)
        index = np.arange(n)
        np.random.shuffle(index)
        return Generator(
            triples[index], self.nb_nodes, self.train_triple_set,
            batch_size=batch_size)
        
