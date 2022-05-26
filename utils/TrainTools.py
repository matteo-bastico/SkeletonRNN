import torch

from typing import List


class SkeletonCollator(object):
    def __init__(self, mask_token):
        self.mask_token = mask_token

    def __call__(self, examples: List[torch.Tensor]):
        # get the length of each sentence
        # examples is a list of Tuple were the first elem is the input while the second is the label
        # the lengths are the same for inputs and labels
        inputs = [example[0] for example in examples]
        labels = [example[1] for example in examples]
        seq_lengths = [len(seq) for seq in inputs]
        # create an empty matrix with padding tokens
        longest_seq = max(seq_lengths)
        batch_size = len(examples)
        padded_batch = torch.ones((batch_size, longest_seq, inputs[0][0].shape[0], inputs[0][0].shape[1]),
                                  dtype=torch.float32) * self.mask_token  # copy over the actual sequences
        padded_labels = torch.ones((batch_size, longest_seq, labels[0][0].shape[0],
                                    labels[0][0].shape[1]),
                                   dtype=torch.float32) * self.mask_token  # copy over the actual sequences
        for i, seq_len in enumerate(seq_lengths):
            seq = inputs[i]
            label = labels[i]
            padded_batch[i, 0:seq_len, :, :] = seq
            padded_labels[i, 0:seq_len, :, :] = label
        return padded_batch, padded_labels


def maskedMSELoss(predict, ground_truth, mask_token):
    se_points = torch.zeros(((predict.shape[2]), predict.shape[3]), dtype=torch.float32,
                            device=predict.device)
    valid_frames = 0
    for idx_b, batch in enumerate(ground_truth):
        for idx_f, frame in enumerate(batch):
            predict_frame = predict[idx_b, idx_f, :, :]
            if not torch.all(frame == (torch.ones(frame.shape, device=predict.device) * mask_token)):
                for idx_r, row in enumerate(frame):
                    se_points[idx_r, :] += (predict_frame[idx_r, :] - row)**2
                valid_frames += 1
    return se_points, valid_frames
