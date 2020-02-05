import torch
from math import sqrt
from torch import nn


from tacotron2.models.modules import Encoder, Decoder, Postnet
from tacotron2.models.tacotron2 import Tacotron2
from tacotron2.utils import get_mask_from_lengths


class Tacotron2Embedded(Tacotron2):

    def __init__(self, hparams):
        super(Tacotron2Embedded, self).__init__(hparams)

    def forward(self, inputs):
        """
        inputs: dict{'x': (...), 'y':(...)}
        """
        text_inputs, text_lengths, mels, max_len, output_lengths, speaker_embeddings = inputs['x']
        text_lengths, output_lengths = text_lengths.data, output_lengths.data

        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)

        encoder_outputs = self.encoder(embedded_inputs, text_lengths)
        encoder_outputs_wide = encoder_outputs.size(1)

        embedded_speaker = torch.cat(
            encoder_outputs_wide * [speaker_embeddings],
            dim=1
        )
        encoder_outputs = torch.cat([encoder_outputs, embedded_speaker], dim=2)

        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, mels, memory_lengths=text_lengths)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments],
            output_lengths)

    def inference(self, inputs, speaker_embeddings):
        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)
        encoder_outputs_wide = encoder_outputs.size(1)

        embedded_speaker = torch.cat(
            encoder_outputs_wide * [speaker_embeddings],
            dim=1
        )
        encoder_outputs = torch.cat([encoder_outputs, embedded_speaker], dim=2)

        mel_outputs, gate_outputs, alignments = self.decoder.inference(
            encoder_outputs)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments])

        return outputs