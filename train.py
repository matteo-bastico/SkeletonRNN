import argparse
import torch
import random
import os
import sklearn.model_selection
import numpy as np
import torch.optim as optim

from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm, trange
from Model.SkeletonRNN import SkeletonRNN
from Dataset.SkeletonDataset import SkeletonDataset
from utils.IOutils import load_data
from utils.OneCycleLR import OneCycleLR
from utils.TrainTools import maskedMSELoss, SkeletonCollator
from tensorboardX import SummaryWriter

torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)

_PAD_POINT = -100


def train(args):
    # Use tensorboard summary
    tb_writer = SummaryWriter()
    with open(tb_writer.logdir + "/args.txt", 'w+') as f:
        f.write(repr(args))
    # Load Data
    skeletons = load_data(args.data_path) # List of sequences each of them [n_frames, n_points, 3]
    train_size = int(0.8 * len(skeletons))
    test_size = len(skeletons) - train_size
    # Train and test splits
    train_skeletons, test_skeletons = sklearn.model_selection.train_test_split(skeletons,
                                                                               train_size=train_size,
                                                                               test_size=test_size,
                                                                               shuffle=True)
    # Generate Dataset
    train_dataset = SkeletonDataset(train_skeletons, augment=True)
    test_dataset = SkeletonDataset(test_skeletons, augment=True)
    train_sampler = RandomSampler(train_dataset)
    skeleton_collator = SkeletonCollator(mask_token=_PAD_POINT)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=skeleton_collator
    )
    test_sampler = RandomSampler(test_dataset)
    test_dataloader = DataLoader(
        test_dataset, sampler=test_sampler, batch_size=args.test_batch_size, collate_fn=skeleton_collator
    )
    # Create model
    skeletonRNN = SkeletonRNN(input_sz=args.input_size, hidden_sz=args.hidden_size,
                              loss_format=[-1, -1, -1])
    skeletonRNN.to(args.device)
    # Training parameters
    optimizer = optim.SGD(skeletonRNN.parameters(), lr=args.learning_rate[0], momentum=0.9)
    t_total = len(train_dataloader) * args.num_train_epochs
    scheduler = OneCycleLR(optimizer, t_total, lr_range=args.learning_rate)
    train_iterator = trange(
        0, int(args.num_train_epochs), desc="Epoch",
    )
    global_step = 0
    # LET'S TRAIN
    for _ in train_iterator:
        tr_se = 0.0
        tot_tr_frames = 0.0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        skeletonRNN.train()
        for step, batch in enumerate(epoch_iterator):
            input_batch = batch[0].to(args.device)
            labels = batch[1].to(args.device)
            # Zero grad
            optimizer.zero_grad()
            predicted_points_seq, hidden_seq = skeletonRNN(input_batch)
            se_points, valid_frames = maskedMSELoss(predicted_points_seq, labels, mask_token=_PAD_POINT)
            loss = se_points.mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1
            tr_se += se_points
            tot_tr_frames += valid_frames
        # Evaluate
        test_se = 0.0
        tot_test_frames = 0.0
        test_iterator = tqdm(test_dataloader, desc="Evaluating")
        skeletonRNN.eval()
        for batch in test_iterator:
            input_batch = batch[0].to(args.device)
            labels = batch[1].to(args.device)
            with torch.no_grad():
                predicted_points_seq, hidden_seq = skeletonRNN(input_batch)
                se_points, valid_frames = maskedMSELoss(predicted_points_seq, labels, mask_token=_PAD_POINT)
                test_se += se_points
                tot_test_frames += valid_frames
        tr_loss = (tr_se/tot_tr_frames).mean().item()
        test_loss = (test_se/tot_test_frames).mean().item()
        tb_writer.add_scalar("lr", scheduler.get_lr(), global_step)
        tb_writer.add_scalar("train_mse [cm]", tr_loss, global_step)
        tb_writer.add_scalar("test_mse [cm]", test_loss, global_step)
        # Save checkpoint
        if global_step % (len(epoch_iterator)*args.save_steps) == 0:
            save_dir = tb_writer.logdir
            save_prefix = os.path.join(save_dir)
            save_path = '{}/checkpoint_{}.pt'.format(save_prefix, global_step)
            output = open(save_path, mode="wb")
            torch.save(skeletonRNN.state_dict(), output)
            output.close()
    print("----------- Training ended ------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="Data/train/examples.npy",
                        help="training data")
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--input_size", default=(18, 3), type=tuple,
                        help="(Skeleton points, dimensions)")
    parser.add_argument("--hidden_size", default=1024, type=int,
                        help="LSTM hidden size: 256, 512 or 1024")
    parser.add_argument("--learning_rate", default=(1e-6, 1e-5), type=tuple,
                        help="The learning rate boundaries for OnceCyleLR.")
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=1, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--save_steps", default=1, type=int,
                        help="Every how many epochs checkpoint the model.")
    args = parser.parse_args()
    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    train(args)
