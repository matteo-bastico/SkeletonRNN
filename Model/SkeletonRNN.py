import math
import torch
import torch.nn as nn


class SkeletonRNN(nn.Module):
    def __init__(self, input_sz=(18, 3), hidden_sz=1024, loss_format=None):
        assert hidden_sz == 256 or hidden_sz == 512 or hidden_sz == 1024, 'Please, select hidden size of 256, 512 or 1024'
        super().__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz
        self.loss_format = loss_format
        self.W = nn.Parameter(torch.Tensor(self.input_size[1], hidden_sz * 4))
        self.U = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_sz * 4))
        self.init_weights()
        self.cnn_decoder = CNNDecoder(hidden_size=hidden_sz)

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x,
                init_states=None):
        # x is of shape (batch, n_frames, n_points, dimensions)
        bs, seq_sz, _, _ = x.size()
        hidden_seq = []
        predicted_points_seq = []
        if init_states is None:
            h_t, c_t = (torch.zeros(bs, self.input_size[0], self.hidden_size).to(x.device),
                        torch.zeros(bs, self.input_size[0], self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states
        # Forward on custom LSTM + CNN_decode
        HS = self.hidden_size
        for t in range(seq_sz):
            if len(predicted_points_seq) == 0:
                x_t = x[:, t, :]
            else:
                # Input refinement block
                prev_output = predicted_points_seq[-1][0, :, :, :].detach().clone()
                x_t = refine_input(x, t, prev_output, self.loss_format)
            # batch the computations into a single matrix multiplication
            gates = x_t @ self.W + h_t @ self.U + self.bias
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :, :HS]),  # input
                torch.sigmoid(gates[:, :, HS:HS * 2]),  # forget
                torch.tanh(gates[:, :, HS * 2:HS * 3]),
                torch.sigmoid(gates[:, :, HS * 3:]),  # output
            )
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            # Do point prediction with cnn
            # Reshape as image with 1 channel
            points_features = torch.reshape(h_t, (bs, 1, self.input_size[0], self.hidden_size))
            predicted_points = self.cnn_decoder(points_features)
            # Back to original shape
            predicted_points = torch.reshape(predicted_points, (bs, self.input_size[0], self.input_size[1]))
            predicted_points_seq.append(predicted_points.unsqueeze(0))
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        predicted_points_seq = torch.cat(predicted_points_seq, dim=0)
        predicted_points_seq = predicted_points_seq.transpose(0, 1).contiguous()
        return predicted_points_seq, hidden_seq


class CNNDecoder(nn.Module):
    def __init__(self, hidden_size=1024):
        """
        :param hidden_size: hidden_size of the LSTM Cell
        """
        super().__init__()
        # For hidden size 1024
        if hidden_size == 1024:
            self.decoder = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0, dilation=1, ceil_mode=False),
                nn.Conv2d(16, 32, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0, dilation=1, ceil_mode=False),
                nn.Conv2d(32, 64, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0, dilation=1, ceil_mode=False),
                nn.Conv2d(64, 32, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3), padding=0, dilation=1, ceil_mode=False),
                nn.Conv2d(32, 16, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0, dilation=1, ceil_mode=False),
                nn.Conv2d(16, 1, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            )
        # For hidden size 512
        elif hidden_size == 512:
            self.decoder = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0, dilation=1, ceil_mode=False),
                nn.Conv2d(16, 16, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0, dilation=1, ceil_mode=False),
                nn.Conv2d(16, 32, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3), padding=0, dilation=1, ceil_mode=False),
                nn.Conv2d(32, 16, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0, dilation=1, ceil_mode=False),
                nn.Conv2d(16, 1, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            )
        # For hidden size 256
        elif hidden_size == 256:
            self.decoder = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0, dilation=1, ceil_mode=False),
                nn.Conv2d(16, 32, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3), padding=0, dilation=1, ceil_mode=False),
                nn.Conv2d(32, 16, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0, dilation=1, ceil_mode=False),
                nn.Conv2d(16, 1, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            )

    def forward(self, x):
        out = self.decoder(x)
        return out


def refine_input(x, t, prev_output, loss_format=None):
    if loss_format is None:
        loss_format = [-1, -1, -1]
    x_t = x[:, t, :]
    for idx_b, frame in enumerate(x_t):
        for idx_r, point in enumerate(frame):
            # Is needed copy to device in case of cuda
            if torch.all(point == torch.tensor(loss_format).to(point.device)):
                # Take output of prev frame if the point is missing
                x_t[idx_b, idx_r, :] = prev_output[idx_b, idx_r, :]
    return x_t
