import argparse
import torch
import random
import numpy as np

from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from Model.SkeletonRNN import SkeletonRNN
from Dataset.SkeletonDataset import SkeletonDatasetTest
from utils.TrainTools import maskedMSELoss, SkeletonCollator
from utils.IOutils import load_data

torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)

_PAD_POINT = -100


def test(args):
    # Load Data
    examples = load_data(args.data_path)
    labels = load_data(args.labels_path)
    test_dataset = SkeletonDatasetTest(examples, labels)
    skeleton_collator = SkeletonCollator(mask_token=_PAD_POINT)
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.test_batch_size, collate_fn=skeleton_collator
    )
    test_iterator = tqdm(test_dataloader, desc="Testing")
    # Create model
    skeletonRNN = SkeletonRNN(input_sz=args.input_size, hidden_sz=args.hidden_size,
                              loss_format=[-1, -1, -1])
    skeletonRNN.to(args.device)
    # Load State Dict
    skeletonRNN.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    skeletonRNN.eval()
    se_total = 0.0
    tot_valid_frames = 0.0
    for batch in test_iterator:
        input_batch = batch[0].to(args.device)
        labels = batch[1].to(args.device)
        with torch.no_grad():
            predicted_points_seq, _ = skeletonRNN(input_batch)
            se_points, valid_frames = maskedMSELoss(predicted_points_seq, labels, mask_token=_PAD_POINT)
            se_total += se_points
            tot_valid_frames += valid_frames
    mse_total = (se_total / valid_frames).mean()
    mse_x = (se_total / tot_valid_frames)[:, 0].mean()
    mse_y = (se_total / tot_valid_frames)[:, 1].mean()
    mse_z = (se_total / tot_valid_frames)[:, 2].mean()
    print("Total test MSE in cm: %.2f" % float(100 * torch.sqrt(mse_total)))
    print("Total test x MSE in cm: %.2f" % float(100 * torch.sqrt(mse_x)))
    print("Total test y MSE in cm: %.2f" % float(100 * torch.sqrt(mse_y)))
    print("Total test z MSE in cm: %.2f" % float(100 * torch.sqrt(mse_z)))
    # Compute Standard Deviation in each coordinate
    print("----------- Computing SD of the error ------------")
    test_iterator = tqdm(test_dataloader, desc="SD")
    sd_total = 0.0
    for batch in test_iterator:
        input_batch = batch[0].to(args.device)
        labels = batch[1].to(args.device)
        with torch.no_grad():
            predicted_points_seq, _ = skeletonRNN(input_batch)
            err_points = torch.zeros(((predicted_points_seq.shape[2]),
                                      predicted_points_seq.shape[3]), dtype=torch.float32, device=args.device)
            for idx_b, batch in enumerate(labels):
                for idx_f, frame in enumerate(batch):
                    predict_frame = predicted_points_seq[idx_b, idx_f, :, :]
                    if not torch.all(frame == (torch.ones(frame.shape, device=args.device) * _PAD_POINT)):
                        err = torch.abs(predict_frame - frame) * 100
                        sd_total += (err - torch.tensor([100 * torch.sqrt(mse_x),
                                                         100 * torch.sqrt(mse_y),
                                                         100 * torch.sqrt(mse_z)])) ** 2
    sd_coord = torch.sqrt(sd_total / len(test_iterator)).mean(axis=0)
    print("Total SD x MSE in cm: %.2f" % float(sd_coord[0]))
    print("Total SD y MSE in cm: %.2f" % float(sd_coord[1]))
    print("Total SD z MSE in cm: %.2f" % float(sd_coord[2]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="Data/test/examples.npy",
                        help="testing data")
    parser.add_argument("--labels_path", type=str, default="Data/test/labels.npy",
                        help="ground truth labels data")
    parser.add_argument("--input_size", default=(18, 3), type=tuple,
                        help="(Skeleton points, dimensions)")
    parser.add_argument("--hidden_size", default=1024, type=int,
                        help="LSTM hidden size: 256, 512 or 1024")
    parser.add_argument("--test_batch_size", default=1, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--checkpoint", default="runs/May26_11-15-32_MacBook-Pro-di-Matteo-2.local/checkpoint_207.pt", type=str,
                        help="Model checkpoint to be trained")
    args = parser.parse_args()
    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    test(args)