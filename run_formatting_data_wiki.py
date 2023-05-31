# -*- coding: utf-8 -*-

import os
import re
import math
import glob
import argparse

from multiprocessing import Pool


BUFSIZE = 40960000


def sent_tokenize(x):
    sents_temp = re.split(r'(……|；|;|。|\.|！|\!|？|\?)', x)
    sents = []
    for i in range(math.ceil(len(sents_temp) / 2)):
        sent = sents_temp[2 * i].strip()
        try:
            # We can get out-of-index here.
            sent += sents_temp[2 * i + 1].strip()
        except:
            pass
        sents.append(sent)
    return sents


def wiki_worker(line):
    line = line.strip()
    if line == "":
        return []
    if line.startswith("</doc>"):
        return [""]
    if line.startswith("<doc id="):
        return [""]
    sents = sent_tokenize(line)
    sents = [sent for sent in sents if len(sent.split()) > 7]
    return sents


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help="Directory for input files to format.")
    parser.add_argument('--input_regex', type=str, required=True, help="Regex for input files to format.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory for output files to write.")
    parser.add_argument('--num_processors', type=int, default=8, help="Num of processors.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    pool = Pool(args.num_processors)
    for input_file in glob.glob(os.path.join(args.input_dir, args.input_regex)):
        stream = open(input_file, "r")
        output_file = os.path.join(args.output_dir, os.path.basename(input_file) + ".format")
        with open(output_file, "w") as fo:
            while True:
                lines = stream.readlines(BUFSIZE)
                if not lines:
                    break
                gathered_sents = pool.map(wiki_worker, lines, len(lines) // args.num_processors)
                if not gathered_sents:
                    continue
                all_sents = []
                for sents in gathered_sents:
                    all_sents.extend(sents)
                marker_sent = True # Whether last sent is a marker sent, i.e., <doc> or </doc>.
                for sent in all_sents:
                    if sent:
                        fo.write(sent + "\n")
                        marker_sent = False
                    else: 
                        if not marker_sent:
                            fo.write("\n")
                        marker_sent = True
        stream.close()

if __name__ == "__main__":
    main()