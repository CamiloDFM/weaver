#!/usr/bin/env python

import argparse
import io

from weaver.wordnet import NetBuilder

parser = argparse.ArgumentParser()
parser.add_argument(
    'input_path',
    help='Text to be analysed',
)
parser.add_argument(
    '-s', '--stemming',
    dest='stemming',
    default=False,
    help='Perform stemming?',
    action='store_true',
)
parser.add_argument(
    '-w', '--weighted',
    dest='weighted',
    default=False,
    help='Add weights to the edges?',
    action='store_true',
)
parser.add_argument(
    '-c', '--edge-criterion',
    dest='edge_criterion',
    default='distance1',
    help='Formation criterion for the edges between words',
)
parser.add_argument(
    '-x', '--stopwords',
    dest='stopwords',
    default=False,
    help='Parse stopwords?',
    action='store_true',
)
parser.add_argument(
    '-l', '--whitelist',
    dest='whitelist',
    default=None,
    help='Filter words using the provided path to a whitelist',
)
parser.add_argument(
    '--sentence',
    dest='sentence',
    default=False,
    help='Partition the text in sentences?',
    action='store_true',
)
parser.add_argument(
    '--top',
    dest='top_words',
    default=None,
    help='Take only this amount of the most frequent words during preprocessing',
)
parser.add_argument(
    '-r', '--rm-common-words',
    dest='remove_common_words',
    default=None,
    type=int,
    help='Remove the N most common English words from the corpus during preprocessing',
)
parser.add_argument(
    '-f', '--from-line',
    dest='from_line',
    default=0,
    type=int,
    help='Line from which to start reading the input file',
)
parser.add_argument(
    '-t', '--to-line',
    dest='to_line',
    default=-1,
    type=int,
    help='Line where reading will stop in the input file',
)
parser.add_argument(
    '-o', '--output-path',
    dest='output_path',
    default='network.net',
    help='Path to output file. Default is a "network.net" file on the current directory',
)
parser.add_argument(
    '--pos-whitelist',
    dest='pos_whitelist',
    default=[],
    help='Perform part-of-speech tagging and take only these tags',
    nargs='*',
)
parser.add_argument(
    '--pos-blacklist',
    dest='pos_blacklist',
    default=[],
    help='Perform part-of-speech tagging and take everything but these tags',
    nargs='*',
)

args = parser.parse_args()

with io.open(args.input_path, 'r', encoding='utf-8') as input_file:
    text_lines = input_file.readlines()
    input_file.close()

network_builder = NetBuilder(
    criterion=args.edge_criterion,
    sentence=args.sentence,
    stemming=args.stemming,
    stopwords=args.stopwords,
    top_words=args.top_words,
    remove_common_words=args.remove_common_words,
    weighted=args.weighted,
    whitelist_path=args.whitelist,
    pos_whitelist=args.pos_whitelist,
    pos_blacklist=args.pos_blacklist,
)

from_line = 0 if args.from_line == 0 else args.from_line - 1
to_line = args.to_line
vertices, edges, weights = network_builder.build_network('\n'.join(text_lines[from_line:to_line]))

with io.open(args.output_path, 'w', encoding='utf-8') as out_stream:
    out_stream.write(f'*Vertices {len(vertices)}\n')
    for i in range(1, len(vertices) + 1):
        out_stream.write(f'{i} "{vertices[i]}"\n')
    out_stream.write('*arcs\n')
    if args.weighted:
        for edge, weight in zip(edges, weights):
            vertex1, vertex2 = edge
            out_stream.write(f'{vertex1} {vertex2} {weight}\n')
    else:
        for vertex1, vertex2 in edges:
            out_stream.write(f'{vertex1} {vertex2}\n')
    out_stream.close()
