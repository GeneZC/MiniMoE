# -*- coding: utf-8 -*-

"""Build MLM/NSP TFRecord examples for BERT."""

import os
import glob
import argparse
import random

from transformers import BertTokenizer

from multiprocessing import Pool

from data import TFRecordWriter


BUFSIZE = 40960000


class Example:
    """An example (sentence pair)."""
    def __init__(self, tokens, segments, nsp):
        self.tokens = tokens
        self.segments = segments
        self.nsp = nsp

    def __str__(self):
        s = ""
        s += "tokens: %s\n" % (" ".join([str(x) for x in self.tokens]))
        s += "segments: %s\n" % (" ".join([str(x) for x in self.segments]))
        s += "nsp: %d\n" % self.nsp
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


def build_examples_from_document(documents, document_index, 
        max_seq_length, short_seq_prob, vocab_words, rng):

    document = documents[document_index]

    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_length = max_num_tokens
    if rng.random() < short_seq_prob:
        target_length = rng.randint(2, max_num_tokens)

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    examples = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i]
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= target_length:
            if current_chunk:
                # `a_end` is how many segments from `current_chunk` go into the `A`
                # (first) sentence.
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = rng.randint(1, len(current_chunk) - 1)

                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                tokens_b = []
                # Random next
                is_random_next = False
                if len(current_chunk) == 1 or rng.random() < 0.5:
                    is_random_next = True
                    target_b_length = target_length - len(tokens_a)

                    # This should rarely go for more than one iteration for large
                    # corpora. However, just to be careful, we try to make sure that
                    # the random document is not the same as the document
                    # we're processing.
                    for _ in range(10):
                        random_document_index = rng.randint(0, len(documents) - 1)
                        if random_document_index != document_index:
                            break
                    
                    random_document = documents[random_document_index]
                    random_start = rng.randint(0, len(random_document) - 1)
                    for j in range(random_start, len(random_document)):
                        tokens_b.extend(random_document[j])
                        if len(tokens_b) >= target_b_length:
                            break
                    # We didn't actually use these segments so we "put them back" so
                    # they don't go to waste.
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                # Actual next
                else:
                    is_random_next = False
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
                # The segment IDs are 0 for the [CLS] token, the A tokens and the first [SEP]
                # They are 1 for the B tokens and the final [SEP]
                segments = [0 for _ in range(len(tokens_a) + 2)] + [1 for _ in range(len(tokens_b) + 1)]

                example = Example(
                    tokens=tokens,
                    segments=segments,
                    nsp=is_random_next,
                )
                examples.append(example)
            current_chunk = []
            current_length = 0
        i += 1
    return examples


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if rng.random() < 0.5:
            trunc_tokens.pop(0)
        else:
            trunc_tokens.pop()


def worker(lines, tokenizer, max_seq_length, short_seq_prob, rng):
    documents = [[]]
    # Input file format:
    # (1) One sentence per line. These should ideally be actual sentences, not
    # entire paragraphs or arbitrary spans of text. (Because we use the
    # sentence boundaries for the "next sentence prediction" task).
    # (2) Blank lines between documents. Document boundaries are needed so
    # that the "next sentence prediction" task doesn't span between documents.
    for line in lines:
        line = line.strip()

        # Empty lines are used as document delimiters.
        if not line:
            documents.append([])
        tokens = tokenizer.tokenize(line)
        if tokens:
            documents[-1].append(tokens)

    # Remove empty documents.
    documents = [x for x in documents if x]
    # Shuffle documents.
    rng.shuffle(documents)

    vocab_words = list(tokenizer.vocab.keys())
    examples = []
    for document_index in range(len(documents)):
        examples.extend(
            build_examples_from_document(
                documents, 
                document_index, 
                max_seq_length, 
                short_seq_prob,
                vocab_words, 
                rng,
            )
        )

    # Shuffle examples.
    rng.shuffle(examples)
    return examples


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", 
        type=str, 
        required=True, 
        help="Directory for input files to format."
    )
    parser.add_argument(
        "--input_regex", 
        type=str, 
        required=True, 
        help="Regex for input files to format."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        help="Directory for output files to write."
    )
    parser.add_argument(
        "--tokenizer_name_or_path", 
        type=str, 
        required=True, 
        help="The tokenizer to use.",
    )
    parser.add_argument(
        "--do_lower_case", 
        action="store_true", 
        help="Whether to lower case the input text. Should be True for uncased "
        "models and False for cased models."
    )
    parser.add_argument("--max_seq_length", type=int, default=128, 
                        help="Maximum sequence length.")
    parser.add_argument("--short_seq_prob", type=float, default=0.1,
                        help="Probability of creating sequences which are shorter than the maximum length.")
    parser.add_argument("--seed", type=int, default=1234,
                        help="Random seed.")
    parser.add_argument("--num_processors", type=int, default=8,
                        help="Num of processors.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name_or_path, do_lower_case=args.do_lower_case)
    rng = random.Random(args.seed)

    pool = Pool(args.num_processors)
    for input_file in glob.glob(os.path.join(args.input_dir, args.input_regex)):
        print("*** Building examples ***")
        print(f"   from {input_file}")

        stream = open(input_file, "r")
        output_file = os.path.join(args.output_dir, os.path.basename(input_file) + ".tfrecord")
        print("*** Writing examples ***")
        print(f"   to {output_file}")
        num_examples = 0
        with TFRecordWriter(output_file) as writer:
            while True:
                lines = stream.readlines(BUFSIZE)
                if not lines:
                    break
                chunk_size = len(lines) // args.num_processors
                arguments = [(lines[i * chunk_size: (i + 1) * chunk_size], tokenizer, args.max_seq_length, args.short_seq_prob, rng) 
                            for i in range(args.num_processors)]
                gathered_examples = pool.starmap(worker, arguments)
                if not gathered_examples:
                    continue
                all_examples = []
                for examples in gathered_examples:
                    all_examples.extend(examples)
                for example in all_examples:
                    writer.write({
                        "indices": (tokenizer.convert_tokens_to_ids(example.tokens), "int"),
                        "segments": (example.segments, "int"),
                        # "nsp": (example.nsp, "int"),
                    })
                    # description = {"indices": "int", "segments": "int", "nsp": "int"}
                    num_examples += 1
                    if num_examples <= 5:
                        print(example)
                print(f"  Having written {num_examples} examples", )
        stream.close()		

if __name__ == "__main__":
    main()
