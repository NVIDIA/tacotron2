from torch import nn
import torch.nn.functional as F


class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    @staticmethod
    def masked_l2_loss(out, target, lengths):
        num_not_padded = lengths.sum() * out.size(1)
        loss = F.mse_loss(out, target, reduction="sum")
        loss = loss / num_not_padded
        return loss

    def forward(self, model_output, targets, output_lengths):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, _ = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = self.masked_l2_loss(mel_out, mel_target, output_lengths) + \
            self.masked_l2_loss(mel_out_postnet, mel_target, output_lengths)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        return mel_loss + gate_loss
