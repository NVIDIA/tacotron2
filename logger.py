import random
import torch
from tensorboardX import SummaryWriter
from plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy
from plotting_utils import plot_gate_outputs_to_numpy


class Tacotron2Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Tacotron2Logger, self).__init__(logdir)

    def log_training(self, reduced_loss, grad_norm, learning_rate, duration, diagonality, avg_prob, avg_MCD, avg_f0,
                     iteration):
        self.add_scalar("training.loss", reduced_loss, iteration)
        self.add_scalar("grad.norm", grad_norm, iteration)
        self.add_scalar("learning.rate", learning_rate, iteration)
        self.add_scalar("duration", duration, iteration)
        self.add_scalar("training.attention_alignment_diagonality", diagonality, iteration)
        self.add_scalar("training.average_max_attention_weight", avg_prob, iteration)
        self.add_scalar("training.log_MCD", avg_MCD, iteration)
        self.add_scalar("training.f0(100hz)", avg_f0, iteration)

    def log_validation(self, reduced_loss, model, y, y_pred, diagonality, avg_prob, avg_MCD, avg_f0, iteration):
        self.add_scalar("validation.loss", reduced_loss, iteration)
        _, mel_outputs, gate_outputs, alignments = y_pred
        mel_targets, gate_targets = y
        self.add_scalar("validation.attention_alignment_diagonality", diagonality, iteration)
        self.add_scalar("validation.average_max_attention_weight", avg_prob, iteration)
        self.add_scalar("validation.log_MCD", avg_MCD, iteration)
        self.add_scalar("validation.f0(100hz)", avg_f0, iteration)

        # plot distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)

        # plot alignment, mel target and predicted, gate target and predicted
        idx = random.randint(0, alignments.size(0) - 1)
        self.add_image(
            "alignment",
            plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T),
            iteration)
        self.add_image(
            "mel_target",
            plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy()),
            iteration)
        self.add_image(
            "mel_predicted",
            plot_spectrogram_to_numpy(mel_outputs[idx].data.cpu().numpy()),
            iteration)
        self.add_image(
            "gate",
            plot_gate_outputs_to_numpy(
                gate_targets[idx].data.cpu().numpy(),
                torch.sigmoid(gate_outputs[idx]).data.cpu().numpy()),
            iteration)
