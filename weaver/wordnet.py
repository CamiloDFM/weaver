# encoding: utf-8

import io
import itertools
import re
from typing import List, Set

import nltk


def increase_value_in_dict(input_dict, index1, index2, increase):
    """auxiliar function that will always ensure that the operation
    dict[index1][index2] += increase succeeds
    it also checks to ensure that the index that comes lexicographically first
    will be associated to the outermost level of the dict"""

    if index1 < index2:
        first = index1
        second = index2
    else:
        first = index2
        second = index1
    try:
        input_dict[first][second] += increase
    except KeyError as k:
        if k.args[0] == first:  # this means the dict doesnt have anything on the outermost key
            input_dict[first] = {}  # create the inner dict
            input_dict[first][second] = increase  # initialise the value
        elif k.args[0] == second:  # this means that the missing key is the one in the inner dict
            input_dict[first][second] = increase  # fix by initialising
    return None


def set_value_in_dict(input_dict, index1, index2, value):
    if index1 < index2:
        first = index1
        second = index2
    else:
        first = index2
        second = index1
    try:
        input_dict[first][second] = value
    except KeyError:
        input_dict[first] = {}
        input_dict[first][second] = value
    return None


def filter_token_by_pos_tag(tagged_tokens, false_positives, tags_to_replace, replacement_tag):
    """auxiliar generator that allows to force a certain subset of
    pos-tagged tokens to take in a specific pos tag"""
    for token, tag in tagged_tokens:
        if tag in tags_to_replace and token.lower() in false_positives:
            yield (token, replacement_tag)
        else:
            yield (token, tag)


class NetBuilder:
    def __init__(
            self,
            criterion='distance1',
            sentence=False,
            stemming=False,
            stopwords=False,
            top_words=None,
            remove_common_words=None,
            weighted=False,
            whitelist_path=None,
            pos_whitelist=None,
            pos_blacklist=None,
    ):
        self.weighted = weighted
        self.top_words = top_words
        self.pos_whitelist = pos_whitelist if pos_whitelist else []
        self.pos_blacklist = pos_blacklist if pos_blacklist else []

        criteria = ['distance1', 'distance2', 'distance3', 'sentence']
        if criterion not in criteria:
            raise ValueError('Unknown specified criteria')
        else:
            self.criterion = criterion

        if criterion == 'sentence':
            self.sentence_partitioning = True
        else:
            self.sentence_partitioning = sentence

        if stopwords:
            self.blacklist = []
        else:
            self.blacklist = nltk.corpus.stopwords.words('english')

        if remove_common_words:
            amount = int(remove_common_words)
            if amount not in [100, 1000, 3000]:
                raise ValueError('Invalid argument for remove_common_words parameter')
            word_file_path = f'../Datos/most common words/{amount}_most_common_words.txt'
            with io.open(word_file_path, 'r', encoding='utf-8') as word_file:
                lines = word_file.readlines()
                lines = [word.lower().strip() for word in lines if word != '']
                self.blacklist.extend(lines)
                word_file.close()

        if stemming:
            self.stemmer = nltk.stem.snowball.SnowballStemmer('english')
        else:
            class IdentityStemmer:
                @staticmethod
                def stem(x):
                    return x
            self.stemmer = IdentityStemmer

        if whitelist_path:
            with io.open(whitelist_path, 'r', encoding='utf-8') as whitelist_file:
                self.whitelist = [word.strip() for word in whitelist_file]
                whitelist_file.close()
        else:
            self.whitelist = []

        # this is faster than isalpha()
        regex = '[^A-Za-z0-9]+'
        self.not_alphanum_regex = re.compile(regex)

    @staticmethod
    def search_for_pos_tagger_proper_noun_false_positives(text: List[List[str]]) -> Set[str]:
        """input: list of tokenised sentences
        output: set containing all the words that start sentences
        and appear somewhere else in the text without caps
        This is important because the default NLTK POS tagger (averaged_perceptron_tagger)
        mistakenly tags these as NNP"""

        # Words that start a sentence and for which we haven't found another non-sentence-starting
        # instance that doesn't start with caps
        candidates = set()
        # Words that don't start with caps somewhere else in the text
        false_positives = set()

        for sentence in text:
            if sentence[0][0].isupper() and sentence[0].lower() not in false_positives:
                candidates.add(sentence[0].lower())
        for sentence in text:
            to_remove = set()
            for word in sentence:
                if word in candidates:
                    false_positives.add(word)
                    to_remove.add(word)
            for word in to_remove:
                candidates.remove(word)

        return false_positives

    def partition_text(self, text: str) -> List[List[str]]:
        """input: a piece of text
        output: the same text, tokenized by sentences and words
        the list contains one string per sentence (if the user asked for sentence partitioning)
        or only one string, contaning the whole text"""

        text = text.strip().replace('\n', ' ')

        if self.sentence_partitioning:
            text = nltk.sent_tokenize(text)
        else:
            text = [text]

        token_text = [nltk.word_tokenize(sentence) for sentence in text]
        return token_text

    def remove_unwanted_words(self, text: List[List[str]]) -> List[List[str]]:
        """input: a list representing a text split in sentences
        output: a list of lists of clean tokens, filtered by what the user wants to keep
        """

        if self.pos_whitelist or self.pos_blacklist:
            if self.sentence_partitioning:
                # input is already a list of sentences
                tokenized_sentences = text
            else:
                # sent tokenise the list before searching for false positives
                joined = ' '.join(text[0])
                sentences = nltk.sent_tokenize(joined)
                tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
            false_positives = self.search_for_pos_tagger_proper_noun_false_positives(tokenized_sentences)  # noqa: E501

        clean_text = []
        for sentence in text:
            if self.pos_whitelist or self.pos_blacklist:
                tagged_tokens = nltk.pos_tag(sentence)
                tagged_tokens = filter_token_by_pos_tag(
                    tagged_tokens,
                    false_positives,
                    ('NNP', 'NNPS'),
                    'NN',
                )
                if self.pos_whitelist:
                    filtered = filter(lambda x: x[1] in self.pos_whitelist, tagged_tokens)
                else:
                    filtered = filter(lambda x: x[1] not in self.pos_blacklist, tagged_tokens)
                sentence = [x[0] for x in filtered]
            clean_tokens = []
            for token in sentence:
                token = token.lower()  # standardisation
                token = re.sub(self.not_alphanum_regex, '', token)  # remove numbers and puncts
                if ((self.whitelist and token in self.whitelist)
                        or (not self.whitelist and token != '' and token not in self.blacklist)):
                    token = self.stemmer.stem(token)
                    clean_tokens.append(token)
            clean_text.append(clean_tokens)
        return clean_text

    def filter_frequent_words(self, clean_text: List[List[str]], frequency: int) -> List[List[str]]:
        """input: a list representing a text split in sentences
        output: a list of sentences, keeping only the N most frequent words"""

        distribution = nltk.probability.FreqDist(
            [word for sentence in clean_text for word in sentence],
        )
        most_frequent = [token for token, _ in distribution.most_common(frequency)]

        new_clean_text = []
        for sentence in clean_text:
            sentence = [word for word in sentence if word in most_frequent]
            if sentence:
                new_clean_text.append(sentence)
        return new_clean_text

    def clean_text(self, text: str) -> List[List[str]]:
        """input: a piece of text
        output: a list of sentences (lists of strings) cleaned by class params"""

        tokenised_text = self.partition_text(text)
        clean_text = self.remove_unwanted_words(tokenised_text)
        if self.top_words:
            clean_text = self.filter_frequent_words(clean_text, int(self.top_words))
        return clean_text

    def build_network(self, text):

        clean_text = self.clean_text(text)
        edges = {}

        if self.criterion == 'distance1':
            # every time a concept appears next to other,
            # the corresponding edge is created / gets stronger
            for sentence in clean_text:
                if len(sentence) <= 1:
                    continue
                c, n = itertools.tee(sentence, 2)
                next(n)
                pairwise_word_list = zip(c, n)
                for cur, nex in pairwise_word_list:
                    if self.weighted:
                        increase_value_in_dict(edges, cur, nex, 1)
                    else:
                        set_value_in_dict(edges, cur, nex, 1)
        elif self.criterion == 'distance2':
            # every time two concepts appear within 2 words of each other
            # the corresponding edge gets stronger
            # and successive words get stronger edges
            for sentence in clean_text:
                c, n, nn = itertools.tee(sentence, 3)
                next(n)
                next(nn)
                next(nn)
                trio_word_list = zip(c, n, nn)
                for cur, nex, nexnex in trio_word_list:
                    if self.weighted:
                        increase_value_in_dict(edges, cur, nex, 2)
                        increase_value_in_dict(edges, cur, nexnex, 1)
                    else:
                        set_value_in_dict(edges, cur, nex, 1)
                        set_value_in_dict(edges, cur, nexnex, 1)
        elif self.criterion == 'sentence':
            # every time two concepts appear in the same sentence,
            # the corresponding edge gets stronger
            for sentence in clean_text:
                product = ((x, y) for x in sentence for y in sentence if x != y)
                for x, y in product:
                    if self.weighted:
                        increase_value_in_dict(edges, x, y, 1)
                    else:
                        set_value_in_dict(edges, x, y, 1)

        # get a vertex list and associate identifiers with labels
        ids = zip(
            itertools.count(1),
            sorted({word for sentence in clean_text for word in sentence}),
        )
        vertices_by_label = {}
        vertices_by_id = {}
        for identifier, label in ids:
            vertices_by_label[label] = identifier
            vertices_by_id[identifier] = label

        edges_by_id = []
        edge_weights = []
        for ver1 in edges:
            for ver2 in edges[ver1]:
                edges_by_id.append((vertices_by_label[ver1], vertices_by_label[ver2]))
                edge_weights.append(edges[ver1][ver2])

        return vertices_by_id, edges_by_id, edge_weights
