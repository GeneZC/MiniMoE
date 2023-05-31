# -*- coding: utf-8 -*-

import random
import torch
import collections
import numpy as np


class DataPipeline:
    def __init__(self, tokenizer, max_length=None):
        self.tokenizer = tokenizer
        if max_length is None:
            self.max_length = tokenizer.model_max_length
        else:
            self.max_length = max_length

    @staticmethod
    def _pad(indices, max_length, pad_idx):
        """Pad a sequence to the maximum length."""
        pad_length = max_length - len(indices)
        return list(indices) + [pad_idx] * pad_length

    def collate(self, batch):
        raise NotImplementedError()


class BertMLMDataPipeline(DataPipeline):
    Example = collections.namedtuple(
        "Example", 
        (
            "indices",
            "segments",
        )
    )
    Batch = collections.namedtuple(
        "Batch", 
        (
            "text_indices", 
            "text_mask",
            "text_segments",
            "mlm_positions",
            "mlm_mask",
            "mlm_labels",
        )
    )

    def __init__(self, tokenizer, max_length=None):
        super().__init__(tokenizer, max_length)

    @staticmethod
    def mask_indices(indices, tokenizer, mlm_prob=0.15,
		max_num_masks=20, do_whole_word_mask=False):
        cand_positions = []
        for (pos, idx) in enumerate(indices):
            if idx == tokenizer.cls_token_id or idx == tokenizer.sep_token_id:
                continue
            # Whole Word Masking means that if we mask all of the wordpieces
            # corresponding to an original word. When a word has been split into
            # WordPieces, the first token does not have any marker and any subsequence
            # tokens are prefixed with ##. So whenever we see the ## token, we
            # append it to the previous set of word positions.
            #
            # Note that Whole Word Masking does *not* change the training code
            # at all -- we still predict each WordPiece independently, softmaxed
            # over the entire vocabulary.
            if (do_whole_word_mask and len(cand_positions) >= 1 and tokenizer.convert_ids_to_tokens(idx).startswith("##")):
                cand_positions[-1].append(pos)
            else:
                cand_positions.append([pos])
        random.shuffle(cand_positions)

        # Call list() here to get a copy.
        masked_indices = list(indices)
        num_to_mask = min(max_num_masks, max(1, int(round(len(indices) * mlm_prob))))

        mlms = []
        covered_positions = set()
        for pos_set in cand_positions:
            if len(mlms) >= num_to_mask:
                break
            # If adding a whole word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(mlms) + len(pos_set) > num_to_mask:
                continue
            is_any_pos_covered = False
            for pos in pos_set:
                if pos in covered_positions:
                    is_any_pos_covered = True
                    break
            if is_any_pos_covered:
                continue
            for pos in pos_set:
                covered_positions.add(pos)
                masked_idx = None
                # 80% of the time, replace with [MASK]
                if random.random() < 0.8:
                    masked_idx = tokenizer.mask_token_id
                else:
                    # 10% of the time, keep original
                    if random.random() < 0.5:
                        masked_idx = indices[pos]
                    # 10% of the time, replace with random word
                    else:
                        masked_idx = random.randint(0, len(tokenizer) - 1)
                masked_indices[pos] = masked_idx
                mlms.append((pos, indices[pos]))
        assert len(mlms) <= num_to_mask
        mlms = sorted(mlms, key=lambda x: x[0])

        mlm_positions = []
        mlm_labels = []
        for p in mlms:
            mlm_positions.append(p[0])
            mlm_labels.append(p[1])

        return masked_indices, mlm_positions, mlm_labels

    def collate(self, batch):
        batch_text_indices = []
        batch_text_mask = []
        batch_text_segments = []
        batch_mlm_positions = []
        batch_mlm_mask = []
        batch_mlm_labels = []
        for example in batch:
            example = BertMLMDataPipeline.Example(**example)
            text_indices, mlm_positions, mlm_labels = self.mask_indices(example.indices, self.tokenizer)
            text_mask = [1] * len(text_indices)
            mlm_mask = [1] * len(mlm_positions)
            text_segments = example.segments

            batch_text_indices.append(self._pad(text_indices, self.max_length, self.tokenizer.pad_token_id))
            batch_text_mask.append(self._pad(text_mask, self.max_length, 0))
            batch_text_segments.append(self._pad(text_segments, self.max_length, 0))
            batch_mlm_positions.append(self._pad(mlm_positions, 20, 0))
            batch_mlm_mask.append(self._pad(mlm_mask, 20, 0))
            batch_mlm_labels.append(self._pad(mlm_labels, 20, 0))
        return BertMLMDataPipeline.Batch(
            text_indices=torch.tensor(batch_text_indices, dtype=torch.long),
            text_mask=torch.tensor(batch_text_mask, dtype=torch.bool),
            text_segments=torch.tensor(batch_text_segments, dtype=torch.long),
            mlm_positions=torch.tensor(batch_mlm_positions, dtype=torch.long),
            mlm_mask=torch.tensor(batch_mlm_mask, dtype=torch.bool),
            mlm_labels=torch.tensor(batch_mlm_labels, dtype=torch.long),
        )


class BertNILDataPipeline(DataPipeline):
    Example = collections.namedtuple(
        "Example", 
        (
            "indices",
            "segments",
        )
    )
    Batch = collections.namedtuple(
        "Batch", 
        (
            "text_indices", 
            "text_mask",
            "text_segments",
        )
    )

    def __init__(self, tokenizer, max_length=None):
        super().__init__(tokenizer, max_length)

    def collate(self, batch):
        batch_text_indices = []
        batch_text_mask = []
        batch_text_segments = []
        for example in batch:
            example = BertNILDataPipeline.Example(**example)
            text_indices = example.indices
            text_mask = [1] * len(text_indices)
            text_segments = example.segments
            
            batch_text_indices.append(self._pad(text_indices, self.max_length, self.tokenizer.pad_token_id))
            batch_text_mask.append(self._pad(text_mask, self.max_length, 0))
            batch_text_segments.append(self._pad(text_segments, self.max_length, 0))
        return BertNILDataPipeline.Batch(
            text_indices=torch.tensor(batch_text_indices, dtype=torch.long),
            text_mask=torch.tensor(batch_text_mask, dtype=torch.bool),
            text_segments=torch.tensor(batch_text_segments, dtype=torch.long),
        )


RobertaMLMDataPipeline = BertMLMDataPipeline


RobertaNILDataPipeline = BertNILDataPipeline
