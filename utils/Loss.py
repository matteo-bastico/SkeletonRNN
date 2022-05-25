import torch


def maskedMSELoss(predict, ground_truth, mask_token):
    se_points = torch.zeros(((predict.shape[2]), predict.shape[3]), dtype=torch.float32)
    valid_frames = 0
    for idx_b, batch in enumerate(ground_truth):
        for idx_f, frame in enumerate(batch):
            predict_frame = predict[idx_b, idx_f, :, :]
            if not torch.all(frame == (torch.ones(frame.shape) * mask_token)):
                for idx_r, row in enumerate(frame):
                    se_points[idx_r, :] += (predict_frame[idx_r, :] - row)**2
                valid_frames += 1
    return se_points, valid_frames
