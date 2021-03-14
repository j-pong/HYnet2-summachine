import random
from contextlib import contextmanager
from distutils.version import LooseVersion
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from typeguard import check_argument_types

from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel

from espnet.nets.pytorch_backend.nets_utils import pad_list

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield

def find_sub_list(sl,l):
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            return ind,ind+sll-1

class ESPnetASRModel(AbsESPnetModel):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        encoder: AbsEncoder,
        decoder: AbsDecoder,
        ctc: CTC,
        rnnt_decoder: None,
        ctc_weight: float = 0.5,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
    ):
        assert check_argument_types()
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight
        assert rnnt_decoder is None, "Not implemented"

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.token_list = token_list.copy()

        self.encoder = encoder
        self.decoder = decoder
        if ctc_weight == 0.0:
            self.ctc = None
        else:
            self.ctc = ctc
        self.rnnt_decoder = rnnt_decoder
        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        if report_cer or report_wer:
            self.error_calculator = ErrorCalculator(
                token_list, sym_space, sym_blank, report_cer, report_wer
            )
        else:
            self.error_calculator = None

    def forward(
        self,
        text_input: torch.Tensor,
        text_input_lengths: torch.Tensor,
        text_output: torch.Tensor,
        text_output_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        # Check that batch_size is unified
        batch_size = text_input.shape[0]

        # for data-parallel
        text_input = text_input[:, : text_input_lengths.max()]
        text_output = text_output[:, : text_output_lengths.max()]

        # error checker
        # from espnet.nets.pytorch_backend.nets_utils import pad_list
        
        # text_input_ = [y[y != self.ignore_id] for y in text_input]
        # for i, text_inp in enumerate(text_input_):
        #     text_inp[text_inp >= self.vocab_size] = 0
        #     text_inp[text_inp < 0] = 0
        #     text_input_[i] = text_inp
        # text_input = pad_list(text_input_, self.eos)

        # text_output_ = [y[y != self.ignore_id] for y in text_output]
        # for i, text_out in enumerate(text_output_):
        #     text_out[text_out >= self.vocab_size] = 0
        #     text_out[text_out < 0] = 0
        #     text_output_[i] = text_out
        # text_output = pad_list(text_output_, self.ignore_id)

        # 1. Encoder
        text_input[text_input == self.ignore_id] = self.eos
        encoder_out, encoder_out_lens = self.encode(text_input, text_input_lengths)

        # 2a. Attention-decoder branch
        if self.ctc_weight == 1.0:
            loss_att, acc_att, cer_att, wer_att = None, None, None, None
        else:
            loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
                encoder_out, encoder_out_lens, text_output, text_output_lengths
            )

        # 2b. CTC branch
        if self.ctc_weight == 0.0:
            loss_ctc, cer_ctc = None, None
        else:
            loss_ctc, cer_ctc = self._calc_ctc_loss(
                encoder_out, encoder_out_lens, text_output, text_output_lengths
            )

        # 2c. RNN-T branch
        if self.rnnt_decoder is not None:
            _ = self._calc_rnnt_loss(encoder_out, encoder_out_lens, text_output, text_output_lengths)

        if self.ctc_weight == 0.0:
            loss = loss_att
        elif self.ctc_weight == 1.0:
            loss = loss_ctc
        else:
            loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att

        stats = dict(
            loss=loss.detach(),
            loss_att=loss_att.detach() if loss_att is not None else None,
            loss_ctc=loss_ctc.detach() if loss_ctc is not None else None,
            acc=acc_att,
            cer=cer_att,
            wer=wer_att,
            cer_ctc=cer_ctc,
        )

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self,
        text_input: torch.Tensor,
        text_input_lengths: torch.Tensor,
        text_output: torch.Tensor,
        text_output_lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # FIXME: These lines suffer from collect_stats error with int seq (?)
        # text_prep_input, text_prep_input_lengths = self._search_clip(text_input, 
        #                                                             text_input_lengths,
        #                                                             text_output,
        #                                                             text_output_lengths)
        text_prep_input = text_input
        text_prep_input_lengths = text_input_lengths
        return {"text_prep_input": text_prep_input, 
                "text_prep_input_lengths": text_prep_input_lengths}

    def encode(
        self, text: torch.Tensor, text_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        encoder_out, encoder_out_lens, _ = self.encoder(text, text_lengths)
        return encoder_out, encoder_out_lens

    def _search_clip(
        self, 
        text_input: torch.Tensor, 
        text_input_lengths: torch.Tensor, 
        text_output: torch.Tensor,
        text_output_lengths: torch.Tensor, 
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # for data-parallel
        text_input = text_input[:, : text_input_lengths.max()]

        temp = []
        total_offset = 100
        for i, text_out in enumerate(text_output):
            # filter out the ignore label
            tol = text_output_lengths[i]
            text_out = text_out[:tol]

            # procedure the process
            prior_post = find_sub_list(text_out.tolist()[1:-1], text_input[i].tolist())
            if prior_post is None:
                raise RuntimeError(f"text_output: {text_out.tolist()}\ntext_input: {text_input[i].tolist()}")
            prior, post = prior_post

            # prior and post index for the random cliping
            offset = random.randint(0, total_offset)
            prior_offset = max(prior - offset, 0)
            assert prior_offset >= 0
            post_offset = post + offset

            # clip the paragraph
            selected_para = text_input[i][prior_offset:post_offset]
            if len(selected_para) < len(text_out):
                raise RuntimeError(f"prior_offeset: {prior_offset}, post_offset: {post_offset}, length: {len(selected_para)}")
            
            # Save the results
            text_input_lengths[i] = post_offset - prior_offset
            temp.append(selected_para)
        text_input = torch.cat(temp, 0)

        return text_input, text_input_lengths

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1
        assert (ys_pad > self.vocab_size).float().sum() == 0

        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att

    def _calc_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        # Calc CTC loss
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc

    def _calc_rnnt_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        raise NotImplementedError