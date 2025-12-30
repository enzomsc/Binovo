import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import einops
import collections
import heapq
from input_encoder import PeakEncoder,PositionEncoder,SequenceEncoder
import model_res
import ownutils

model_type = torch.float32

class PeaksEncoder(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        min_intensity_wavelength: float = 1e-6,
        max_intensity_wavelength: float = 1,
        min_rt_wavelength: float = 1e-6,
        max_rt_wavelength: float = 10
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.peak_encoder = PeakEncoder(
            d_model,
            min_intensity_wavelength=min_intensity_wavelength,
            max_intensity_wavelength=max_intensity_wavelength,
            learnable_wavelengths = False
        )
        self.rt_encoder = PositionEncoder(
            d_model,
            min_wavelength=min_rt_wavelength,
            max_wavelength=max_rt_wavelength
        )
        self.combiner = torch.nn.Linear(2 * d_model, d_model, bias=False)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        encoded_peak=self.peak_encoder(X)
        encoded_rt=self.rt_encoder(X[:,:,2])    #rt
        encoded = torch.cat(
            [
                encoded_peak,
                encoded_rt,
            ],
            dim=2,
        )
        encoded=encoded.float()
        return self.combiner(encoded)


class SpectrumEncoder(torch.nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        n_head: int = 8,
        dim_feedforward: int = 1024,
        n_layers: int = 1,
        dropout: float = 0,
        max_rt_wavelength=4,
    ):
        super().__init__()
        self.latent_spectrum = torch.nn.Parameter(torch.randn(1, 1, d_model))
        self.peak_encoder = PeaksEncoder(d_model,max_rt_wavelength=max_rt_wavelength)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            layer,
            num_layers=n_layers,
        )

    def forward(
        self,
        spectra: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            zeros = ~spectra.sum(dim=2).bool()
            mask = [
                torch.tensor([[False]] * spectra.shape[0]).type_as(zeros),
                zeros,
            ]
            mask = torch.cat(mask, dim=1)
            peaks = self.peak_encoder(spectra)
            latent_spectra = self.latent_spectrum.expand(peaks.shape[0], -1, -1)

            peaks = torch.cat([latent_spectra, peaks], dim=1)
            memory=self.transformer_encoder(peaks, src_key_padding_mask=mask)
        except Exception as e:
            print(e)
            exit(1)
        return memory, mask
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


class PeptideDecoder(torch.nn.Module):
    def __init__(
        self,
        dim_model=128,
        n_head=8,
        dim_feedforward=1024,
        n_layers=6,
        dropout=0.1,
        num_aa=0,
        max_charge=5,
    ):
        super().__init__()
        self.num_tokens=num_aa+ 1
        self.mass_encoder = PositionEncoder(dim_model)
        self.pos_encoder = SequenceEncoder(dim_model)
        self.charge_encoder = torch.nn.Embedding(max_charge, dim_model)
        self.aa_embedding = torch.nn.Embedding(
            self.num_tokens,
            dim_model,
            padding_idx=0,
        )
        layer = torch.nn.TransformerDecoderLayer(
            d_model=dim_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
            activation=nn.GELU()
        )
        self.transformer_decoder = torch.nn.TransformerDecoder(
            layer,
            num_layers=n_layers,
        )

        self.final = torch.nn.Linear(dim_model, self.num_tokens)

    def generate_tgt_mask(self,sz: int) -> torch.Tensor:
        return ~torch.triu(torch.ones(sz, sz, dtype=torch.bool, device=next(self.parameters()).device)).transpose(0, 1)

    def forward(self, tokens, precursors_org, memory, memory_key_padding_mask):
        masses = self.mass_encoder(precursors_org[:, None, 0])
        charges = self.charge_encoder(precursors_org[:, 1].int() - 1)
        precursors = masses + charges[:, None, :]
        if tokens is None:
            tgt = precursors
        else:
            tgt = torch.cat([precursors, self.aa_embedding(tokens)], dim=1)

        tgt_key_padding_mask = tgt.sum(axis=2) == 0
        tgt = self.pos_encoder(tgt)
        tgt_mask = self.generate_tgt_mask(tgt.shape[1])
        preds = self.transformer_decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.final(preds)

class BiTransformer(torch.nn.Module):
    def __init__(self,
        dim_model:int=128,
        n_head:int=8,
        dim_feedforward:int=1024,
        n_endcoder_layers:int=6,
        dropout:float=0.1,
        num_aa:int=0,
        max_charge:int=10,
        n_decoder_layers:int=6,
        rt_width:int=4,
        beta:float=0.9,
        top_match:int=1,
        n_beams:int=5,
        precursor_mass_tol:float=30, #ppm
        max_length:int=30,
        min_len:int=6,
        train_label_smoothing: float = 0.01,
        ):
        super().__init__()
        self.dim_model=dim_model
        self.n_head=n_head
        self.dim_feedforward=dim_feedforward
        self.n_endcoder_layers=n_endcoder_layers
        self.dropout=dropout
        self.num_aa=num_aa
        self.n_decoder_layers=n_decoder_layers
        self.max_charge=max_charge
        self.rt_width=rt_width
        self.beta=beta
        self.top_match=top_match
        self.n_beams=n_beams
        self.precursor_mass_tol=precursor_mass_tol
        self.max_length=max_length
        self.min_peptide_len=min_len
        self.train_label_smoothing=train_label_smoothing
        self.spectrum_encoder=SpectrumEncoder(self.dim_model,self.n_head,self.dim_feedforward,self.n_endcoder_layers,0.5,max_rt_wavelength=rt_width)
        self.l2rdecoder=PeptideDecoder(self.dim_model,self.n_head,self.dim_feedforward,self.n_decoder_layers,self.dropout,self.num_aa,self.max_charge)
        self.r2ldecoder=PeptideDecoder(self.dim_model,self.n_head,self.dim_feedforward,self.n_decoder_layers,self.dropout,self.num_aa,self.max_charge)
        self.frag_layer = torch.nn.Linear(self.dim_model,3)
        self.softmax = torch.nn.Softmax(2)
        self.CELoss_l2r = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.CELoss_r2l = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.fragCELoss = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.,20,150]))
        if torch.cuda.is_available():
            self.device=torch.device('cuda')
        else:
            self.device=torch.device('cpu')

    def setBeta(self, beta:float):
        self.beta=beta

    def forward_step(
        self,
        spectra: torch.Tensor,
        precursors: torch.Tensor,
        seq: torch.Tensor,
        reverse_seq: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.spectrum_encoder(spectra)
        l2r_result =  self.l2rdecoder(seq, precursors, *emb)
        r2_l_result = self.r2ldecoder(reverse_seq,precursors,*emb)

        return l2r_result, r2_l_result
    
    def forward(self,batch,mode,train_mode):
        if mode:
            spectra, token_seq, rev_token_seq, frag_label, precursor = batch
            emb = self.spectrum_encoder(spectra)
            l2r_result_org =  self.l2rdecoder(token_seq, precursor, *emb)
            pred_frag = self.frag_layer(emb[0])
            l2r_result = l2r_result_org[:,:-1,:].reshape(-1, self.num_aa + 1)
            true_seqs = token_seq
            l2r_loss = self.CELoss_l2r(l2r_result,true_seqs.flatten())
            pred_frag = pred_frag[:,1:,:].reshape(-1, 3)
            frag_loss = self.fragCELoss(pred_frag, frag_label.flatten())

            if train_mode:
                return self.beta*frag_loss+(1-self.beta)*l2r_loss
            else:
                pred_seq = torch.argmax(l2r_result_org, dim=2)[:,:-1]
                correct_aa = torch.logical_or(pred_seq  == true_seqs, true_seqs == 0)
                correct_pep = torch.mean(correct_aa.float(), dim=1) == 1
                pep_acc = torch.mean(correct_pep.float())
                return self.beta*frag_loss+(1-self.beta)*l2r_loss, pep_acc    
        else:
            spectra, token_seq, rev_token_seq, frag_label, precursor = batch
            emb = self.spectrum_encoder(spectra)
            r2_l_result_org= self.r2ldecoder(rev_token_seq,precursor,*emb)
            pred_frag = self.frag_layer(emb[0])
            r2_l_result = r2_l_result_org[:,:-1,:].reshape(-1, self.num_aa + 1)
            rev_true_seqs = rev_token_seq
            r2l_loss = self.CELoss_r2l(r2_l_result,rev_true_seqs.flatten())
            pred_frag = pred_frag[:,1:,:].reshape(-1, 3)
            frag_loss = self.fragCELoss(pred_frag, frag_label.flatten())

            if train_mode:
                return self.beta*frag_loss+(1-self.beta)*r2l_loss
            else:
                pred_seq = torch.argmax(r2_l_result_org, dim=2)[:,:-1]
                correct_aa = torch.logical_or(pred_seq  == rev_true_seqs, rev_true_seqs == 0)
                correct_pep = torch.mean(correct_aa.float(), dim=1) == 1
                pep_acc = torch.mean(correct_pep.float())
                return self.beta*frag_loss+(1-self.beta)*r2l_loss, pep_acc
    
    
    def predict_step(self, batch):
        spectra, token_seq, rev_token_seq, frag_label, precursor = batch
        cur_seq = torch.empty((len(token_seq),0), dtype=torch.int32, device=torch.device('cuda'))
        aa_conf = torch.empty((len(token_seq),0), dtype=torch.int32, device=torch.device('cuda'))
        cur_rev_seq = torch.empty((len(token_seq),0), dtype=torch.int32, device=torch.device('cuda'))
        aa_rev_conf = torch.empty((len(token_seq),0), dtype=torch.int32, device=torch.device('cuda'))
        for i in range(self.max_length+1):
            l2r_pred,r2l_pred=self.forward_step(spectra.cuda(ownutils.n_gpu[0]), precursor.cuda(ownutils.n_gpu[0]), cur_seq.cuda(ownutils.n_gpu[0]), cur_rev_seq.cuda(ownutils.n_gpu[0]))
            next_aa_scores = torch.softmax(l2r_pred[:,-1,:], 1)
            next_aas = torch.argmax(next_aa_scores, 1)
            aa_conf = torch.cat([aa_conf, torch.reshape(next_aa_scores[range(len(next_aas)),next_aas], (-1, 1))], dim=1)
            cur_seq = torch.cat([cur_seq, torch.reshape(next_aas, (-1, 1))], dim=1)            
            next_rev_aa_scores = torch.softmax(r2l_pred[:,-1,:], 1)
            next_rev_aas = torch.argmax(next_rev_aa_scores, 1)
            aa_rev_conf = torch.cat([aa_rev_conf, torch.reshape(next_rev_aa_scores[range(len(next_rev_aas)),next_rev_aas], (-1, 1))], dim=1)
            cur_rev_seq = torch.cat([cur_rev_seq, torch.reshape(next_rev_aas, (-1, 1))], dim=1)  
            
        return cur_seq,aa_conf,cur_rev_seq,aa_rev_conf

    def prediction(self,batch):
        pre_pep=self.beam_search_decode(batch[0].cuda(ownutils.n_gpu[0]),batch[4].cuda(ownutils.n_gpu[0]),False)
        pre_rev=self.beam_search_decode(batch[0].cuda(ownutils.n_gpu[0]),batch[4].cuda(ownutils.n_gpu[0]),True)
        if len(pre_rev)!=len(pre_pep):
            raise ("prediction forward and reverse not equ!!")
            sys.exit(1)
        return pre_pep, pre_rev

    def beam_search_decode(
        self, spectra: torch.Tensor, precursors: torch.Tensor, seq_label:bool,
    ) -> List[List[Tuple[float, np.ndarray, str]]]:
        if seq_label:
            self.decoder=self.r2ldecoder
        else:
            self.decoder=self.l2rdecoder

        memories, mem_masks = self.spectrum_encoder(spectra)
        batch = spectra.shape[0] 
        length = self.max_length + 1  
        vocab = model_res.seq_size + 1  
        beam = self.n_beams  
        scores = torch.full(
            size=(batch, length, vocab, beam), fill_value=torch.nan
        )
        scores = scores.type_as(spectra)
        tokens = torch.zeros(batch, length, beam, dtype=torch.int64)
        tokens = tokens.to(self.device)
        pred_cache = collections.OrderedDict((i, []) for i in range(batch))

        pred = self.decoder(None, precursors, memories, mem_masks)
        tokens[:, 0, :] = torch.topk(pred[:, 0, :], beam, dim=1)[1]
        scores[:, :1, :, :] = einops.repeat(pred, "B L V -> B L V S", S=beam)
        precursors = einops.repeat(precursors, "B L -> (B S) L", S=beam)
        mem_masks = einops.repeat(mem_masks, "B L -> (B S) L", S=beam)
        memories = einops.repeat(memories, "B L V -> (B S) L V", S=beam)
        tokens = einops.rearrange(tokens, "B L S -> (B S) L")
        scores = einops.rearrange(scores, "B L V S -> (B S) L V")

        for step in range(0, self.max_length):
            (
                finished_beams,
                beam_fits_precursor,
                discarded_beams,
            ) = self.finish_beams(tokens, precursors, step)
            self.cache_finished_beams(
                tokens,
                scores,
                step,
                finished_beams & ~discarded_beams,
                beam_fits_precursor,
                pred_cache,
            )

            finished_beams |= discarded_beams
            if finished_beams.all():
                break

            scores[~finished_beams, : step + 2, :] = self.decoder(
                tokens[~finished_beams, : step + 1],
                precursors[~finished_beams, :],
                memories[~finished_beams, :, :],
                mem_masks[~finished_beams, :],
            )
            tokens, scores = self.get_topk_beams(
                tokens, scores, finished_beams, batch, step + 1
            )

        return list(self.get_top_peptide(pred_cache))

    def finish_beams(
        self,
        tokens: torch.Tensor,
        precursors: torch.Tensor,
        step: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        aa_neg_mass = [None]
        for aa, mass in model_res.mass_AA.items():
            if mass < 0:
                aa_neg_mass.append(aa)
        n_term = torch.Tensor(
            [
                self.decoder._aa2idx[aa]
                for aa in model_res.mass_AA
                if aa.startswith(("+", "-"))
            ]
        ).to(self.device)

        beam_fits_precursor = torch.zeros(
            tokens.shape[0], dtype=torch.bool
        ).to(self.device)
        finished_beams = torch.zeros(tokens.shape[0], dtype=torch.bool).to(
            self.device
        )
        ends_stop_token = tokens[:, step] == model_res.seq_to_id["$"]
        finished_beams[ends_stop_token] = True
        discarded_beams = torch.zeros(tokens.shape[0], dtype=torch.bool).to(
            self.device
        )
        discarded_beams[tokens[:, step] == 0] = True

        if step > 1:
            dim0 = torch.arange(tokens.shape[0])
            final_pos = torch.full((ends_stop_token.shape[0],), step)
            final_pos[ends_stop_token] = step - 1
            multiple_mods = torch.isin(
                tokens[dim0, final_pos], n_term
            ) & torch.isin(tokens[dim0, final_pos - 1], n_term)
            mask = (final_pos - 1)[:, None] >= torch.arange(tokens.shape[1])
            internal_mods = torch.isin(
                torch.where(mask.to(self.device), tokens, 0), n_term
            ).any(dim=1)
            discarded_beams[multiple_mods | internal_mods] = True

        for i in range(len(finished_beams)):
            if discarded_beams[i]:
                continue
            pred_tokens = tokens[i][: step + 1]
            peptide_len = len(pred_tokens)
            peptide = self.detokenize(pred_tokens)
            peptide = peptide[:-1]
            peptide_len -= 1

            if finished_beams[i] and peptide_len < self.min_peptide_len:
                discarded_beams[i] = True
                continue

            precursor_charge = precursors[i, 1]
            precursor_mz = precursors[i, 2]
            matches_precursor_mz = exceeds_precursor_mz = False
            for aa in [None] if finished_beams[i] else aa_neg_mass:
                if aa is None:
                    calc_peptide = peptide
                else:
                    calc_peptide = peptide.copy()
                    calc_peptide.append(aa)
                try:
                    calc_mz = self.calMass(
                        seq=calc_peptide, charge=precursor_charge
                    )
                    #! isotope
                    delta_mass_ppm = [
                        calc_mass_error(
                            calc_mz,
                            precursor_mz,
                            precursor_charge,
                            isotope,
                        )
                        for isotope in range(
                            0,2
                        )
                    ]

                    matches_precursor_mz = aa is None and any(
                        abs(d) < self.precursor_mass_tol
                        for d in delta_mass_ppm
                    )

                    if matches_precursor_mz:
                        exceeds_precursor_mz = False
                    else:
                        exceeds_precursor_mz = all(
                            d > self.precursor_mass_tol for d in delta_mass_ppm
                        )
                        exceeds_precursor_mz = (
                            finished_beams[i] or aa is not None
                        ) and exceeds_precursor_mz
                    if matches_precursor_mz or exceeds_precursor_mz:
                        break
                except KeyError:
                    matches_precursor_mz = exceeds_precursor_mz = False

            if finished_beams[i]:
                beam_fits_precursor[i] = matches_precursor_mz
            elif exceeds_precursor_mz:
                finished_beams[i] = True
                beam_fits_precursor[i] = matches_precursor_mz
        return finished_beams, beam_fits_precursor, discarded_beams

    def cache_finished_beams(
        self,
        tokens: torch.Tensor,
        scores: torch.Tensor,
        step: int,
        beams_to_cache: torch.Tensor,
        beam_fits_precursor: torch.Tensor,
        pred_cache: Dict[
            int, List[Tuple[float, float, np.ndarray, torch.Tensor]]
        ],
    ):

        for i in range(len(beams_to_cache)):
            if not beams_to_cache[i]:
                continue
            spec_idx = i // self.n_beams

            pred_tokens = tokens[i][: step + 1]
            has_stop_token = pred_tokens[-1] == model_res.seq_to_id["$"]
            pred_peptide = pred_tokens[:-1] if has_stop_token else pred_tokens
            if any(
                torch.equal(pred_cached[-1], pred_peptide)
                for pred_cached in pred_cache[spec_idx]
            ):
                continue
            smx = self.softmax(scores[i : i + 1, : step + 1, :])
            aa_scores = smx[0, range(len(pred_tokens)), pred_tokens].tolist()

            if not has_stop_token:
                aa_scores.append(0)
            aa_scores = np.asarray(aa_scores)
            aa_scores, peptide_score = aa_pep_score(
                aa_scores, beam_fits_precursor[i]
            )
            aa_scores = aa_scores[:-1]

            if len(pred_cache[spec_idx]) < self.n_beams:
                heapadd = heapq.heappush
            else:
                heapadd = heapq.heappushpop
            heapadd(
                pred_cache[spec_idx],
                (
                    peptide_score,
                    np.random.random_sample(),
                    aa_scores,
                    torch.clone(pred_peptide),
                ),
            )

    def get_topk_beams(
        self,
        tokens: torch.tensor,
        scores: torch.tensor,
        finished_beams: torch.tensor,
        batch: int,
        step: int,
    ) -> Tuple[torch.tensor, torch.tensor]:
        beam = self.n_beams  
        vocab = model_res.seq_size + 1  

        tokens = einops.rearrange(tokens, "(B S) L -> B L S", S=beam)
        scores = einops.rearrange(scores, "(B S) L V -> B L V S", S=beam)

        prev_tokens = einops.repeat(
            tokens[:, :step, :], "B L S -> B L V S", V=vocab
        )
        prev_scores = torch.gather(
            scores[:, :step, :, :], dim=2, index=prev_tokens
        )
        prev_scores = einops.repeat(
            prev_scores[:, :, 0, :], "B L S -> B L (V S)", V=vocab
        )

        step_scores = torch.zeros(batch, step + 1, beam * vocab).type_as(
            scores
        )
        step_scores[:, :step, :] = prev_scores
        step_scores[:, step, :] = einops.rearrange(
            scores[:, step, :, :], "B V S -> B (V S)"
        )

        active_mask = (
            ~finished_beams.reshape(batch, beam).repeat(1, vocab)
        ).float()

        active_mask[:, :beam] = 1e-8

        _, top_idx = torch.topk(step_scores.nanmean(dim=1) * active_mask, beam)
        v_idx, s_idx = np.unravel_index(top_idx.cpu(), (vocab, beam))
        s_idx = einops.rearrange(s_idx, "B S -> (B S)")
        b_idx = einops.repeat(torch.arange(batch), "B -> (B S)", S=beam)

        tokens[:, :step, :] = einops.rearrange(
            prev_tokens[b_idx, :, 0, s_idx], "(B S) L -> B L S", S=beam
        )
        tokens[:, step, :] = torch.tensor(v_idx)
        scores[:, : step + 1, :, :] = einops.rearrange(
            scores[b_idx, : step + 1, :, s_idx], "(B S) L V -> B L V S", S=beam
        )
        scores = einops.rearrange(scores, "B L V S -> (B S) L V")
        tokens = einops.rearrange(tokens, "B L S -> (B S) L")
        return tokens, scores

    def get_top_peptide(
        self,
        pred_cache: Dict[
            int, List[Tuple[float, float, np.ndarray, torch.Tensor]]
        ],
    ) -> Iterable[List[Tuple[float, np.ndarray, str]]]:

        for peptides in pred_cache.values():
            if len(peptides) > 0:
                yield [
                    (
                        pep_score,
                        aa_scores,
                        "".join(self.detokenize(pred_tokens)),
                    )
                    for pep_score, _, aa_scores, pred_tokens in heapq.nlargest(
                        self.top_match, peptides
                    )
                ]
            else:
                yield []

    def detokenize(self, tokens):
        sequence = [model_res.id_to_seq.get(i.item(), "") for i in tokens]
        if "$" in sequence:
            idx = sequence.index("$")
            sequence = sequence[: idx + 1]

        return sequence

    def calMass(self, seq, charge=None):

        if isinstance(seq, str):
            seq = list(seq)

        calc_mass = sum([model_res.mass_AA[aa] for aa in seq]) + 2*1.007825035+15.99491463
        if charge is not None:
            calc_mass = (calc_mass / charge) + 1.00727646688

        return calc_mass

def calc_mass_error(
    calc_mz: float, obs_mz: float, charge: int, isotope: int = 0
) -> float:
    return (calc_mz - (obs_mz - isotope * 1.00335 / charge)) / obs_mz * 10**6

def aa_pep_score(
    aa_scores: np.ndarray, fits_precursor_mz: bool
) -> Tuple[np.ndarray, float]:

    peptide_score = np.mean(aa_scores)
    aa_scores = (aa_scores + peptide_score) / 2
    if not fits_precursor_mz:
        peptide_score -= 1
    return aa_scores, peptide_score
