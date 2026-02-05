import lightning as L
import torch
import itertools
import hydra.utils
import numpy as np
import os
from dataclasses import dataclass
from .bert_backbone import BertBackbone
from .utils import ema, noise_schedule
from .utils.utils_data import get_tokenizer

@dataclass
class Loss:
    loss: torch.FloatTensor
    nlls: torch.FloatTensor
    token_mask: torch.FloatTensor

class Motif_DiffMol(L.LightningModule):
    def __init__(self,config):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = get_tokenizer()
        self.config = config
        self.vocab_size = self.tokenizer.vocab_size
        self.sampler = self.config.model.sampler  
        self.cross_attn = self.config.model.cross_attn  #True
        self.ignore_bos = self.config.model.ignore_bos  

        if (not hasattr(self.tokenizer, 'mask_token_id')
                or self.tokenizer.mask_token_id is None):
            self.mask_index = self.vocab_size
            self.vocab_size += 1
        else:
            self.mask_index = self.tokenizer.mask_token_id  #4

        self.eos_index = self.tokenizer.eos_token_id
        self.parameterization = self.config.model.parameterization  #subs

        if hasattr(self.config, 'unit_size'):
            self.unit_size = self.config.unit_size
        else:
            self.unit_size = self.config.model.length

        if self.config.model.backbone == 'bert':
            self.backbone = BertBackbone(self.config, vocab_size=self.vocab_size)
        else:
            raise ValueError(f'Unknown backbone: {self.config.model.backbone}')

        self.num_tokens = self.config.model.length  #128
        self.noise = noise_schedule.get_noise(self.config)

        self.var_min = self.config.model.var_min 
        if self.var_min:
            self.register_buffer('sampling_eps_min', torch.tensor(  
                self.config.model.sampling_eps_min))
            self.register_buffer('sampling_eps_max', torch.tensor(  
                self.config.model.sampling_eps_max))

        self.neg_infinity = -1000000.0
        self.fast_forward_epochs = None
        self.fast_forward_batches = None

        if self.config.training.ema > 0:
            self.ema = ema.ExponentialMovingAverage(
                self._get_parameters(),
                decay=self.config.training.ema)
        else:
            self.ema = None

        if self.config.mode == 'train':
            self.antithetic_sampling = self.config.training.antithetic_sampling  
            self.log_file = os.path.join(os.path.dirname(config.callbacks.step_ckpt.dirpath), "train_log.txt")

    def _get_parameters(self):
        parameters = [self.backbone.parameters(),
                      self.noise.parameters()]
        return itertools.chain(*parameters)

    def on_load_checkpoint(self, checkpoint):
        self._loaded_sampler_state = checkpoint.get('sampler', None)
        print('Loading checkpoint at', checkpoint['global_step'])
        self._restarting_skip_val_flag = True

        # for models compiled with `torch.compile`
        if '_orig_mod.' in list(checkpoint['state_dict'].keys())[0]:
            checkpoint = self._replace_ckpt_keys(checkpoint)

        if self.ema:
            self.ema.load_state_dict(checkpoint['ema'])
        if 'sampling_eps_min' in checkpoint.keys():
            self.sampling_eps_min = checkpoint['sampling_eps_min']
            self.sampling_eps_max = checkpoint['sampling_eps_max']
        # Copied from:
        # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py#L41
        self.fast_forward_epochs = checkpoint['loops'][
            'fit_loop']['epoch_progress']['current']['completed']
        self.fast_forward_batches = checkpoint['loops'][
            'fit_loop']['epoch_loop.batch_progress'][
            'current']['completed']

        if isinstance(self._loaded_sampler_state, dict) and 'random_state' in self._loaded_sampler_state:
            self.fast_forward_random_state = self._loaded_sampler_state.get('random_state')

    def on_save_checkpoint(self, checkpoint):
        if self.ema:
            checkpoint['ema'] = self.ema.state_dict()
        if hasattr(self, 'sampling_eps_min'):
            checkpoint['sampling_eps_min'] = self.sampling_eps_min
            checkpoint['sampling_eps_max'] = self.sampling_eps_max
        # Copied from:
        # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/tasks/seq.py
        # ['epoch_loop.batch_progress']['total']['completed'] is 1 iteration
        # behind, so we're using the optimizer's progress.
        checkpoint['loops']['fit_loop'][
            'epoch_loop.batch_progress']['total'][
            'completed'] = checkpoint['loops']['fit_loop'][
                               'epoch_loop.automatic_optimization.optim_progress'][
                               'optimizer']['step']['total'][
                               'completed'] * self.trainer.accumulate_grad_batches
        checkpoint['loops']['fit_loop'][
            'epoch_loop.batch_progress']['current'][
            'completed'] = checkpoint['loops']['fit_loop'][
                               'epoch_loop.automatic_optimization.optim_progress'][
                               'optimizer']['step']['current'][
                               'completed'] * self.trainer.accumulate_grad_batches
        # _batches_that_stepped tracks the number of global steps, not the number
        # of local steps, so we don't multiply with self.trainer.accumulate_grad_batches here.
        checkpoint['loops']['fit_loop'][
            'epoch_loop.state_dict'][
            '_batches_that_stepped'] = checkpoint['loops']['fit_loop'][
            'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total']['completed']

        train_dl = self.trainer.train_dataloader
        batch_sampler = getattr(train_dl, "batch_sampler", None)
        if batch_sampler is not None and hasattr(batch_sampler, "state_dict"):
            checkpoint['sampler'] = batch_sampler.state_dict()
        else:
            # fallback: try normal sampler
            sampler = getattr(train_dl, "sampler", None)
            if sampler is not None and hasattr(sampler, "state_dict"):
                checkpoint['sampler'] = sampler.state_dict()
            else:
                checkpoint['sampler'] = {}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self._get_parameters(),
            lr=self.config.optim.lr,
            betas=(self.config.optim.beta1,
                   self.config.optim.beta2),
            eps=self.config.optim.eps,
            weight_decay=self.config.optim.weight_decay)

        scheduler = hydra.utils.instantiate(
            {'_target_': 'transformers.get_constant_schedule_with_warmup',
             'num_warmup_steps': 2500},
            optimizer=optimizer)

        scheduler_dict = {'scheduler': scheduler,
                          'interval': 'step',
                          'monitor': 'val/loss',
                          'name': 'trainer/lr'}
        return [optimizer], [scheduler_dict]

    def on_train_start(self):
        if self.ema:
            self.ema.move_shadow_params_to_device(self.device)

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.ema:
            self.ema.update(self._get_parameters())

    def forward(self, x, attention_mask=None,description_embed=None, description_attention_mask=None,sample_mode=False):
        with torch.amp.autocast('cuda', dtype=torch.float32):
            logits = self.backbone(x,
                                   attention_mask = attention_mask,
                                   sample_mode=sample_mode)
        if self.cross_attn:
            if self.config.mode == 'train':
                n = x.shape[1] // 2
                x = x[:, :n]  
            else:
                x = x[:, :self.config.model.length] 
        if self.parameterization == 'subs':
            return self._subs_parameterization(logits=logits,   
                                               xt=x, sample_mode=sample_mode)
        return logits

    def _subs_parameterization(self, logits, xt, sample_mode):
        logits[:, :, self.mask_index] += self.neg_infinity  
        if sample_mode:
            logits[:, :, self.eos_index] += self.neg_infinity

        logits = logits - torch.logsumexp(logits, dim=-1,   
                                          keepdim=True)

        unmasked_indices = (xt != self.mask_index)  
        logits[unmasked_indices] = self.neg_infinity    
        logits[unmasked_indices, xt[unmasked_indices]] = 0  
        return logits

    def on_train_epoch_start(self):
        self.backbone.train()
        self.noise.train()

    def training_step(self, batch, step):

        smiles_ids, smiles_attention_mask = (batch['smiles'][key] for key in ['input_ids', 'attention_mask'])
        loss = self._loss(smiles_ids, smiles_attention_mask)
        self.log(name='train_loss',
                 value=loss,
                 on_step=True,
                 on_epoch=False,
                 prog_bar=True,
                 sync_dist=True)
        return loss

    def _loss(self, x0, attention_mask, description_embed=None, description_attention_mask=None, sampling_eps_min=None, sampling_eps_max=None):
        if sampling_eps_min is None and hasattr(self, 'sampling_eps_min'):
            sampling_eps_min = self.sampling_eps_min    #0.001
            sampling_eps_max = self.sampling_eps_max    #1
        elif not hasattr(self, 'sampling_eps_min'):
            sampling_eps_min = 1e-3
            sampling_eps_max = 1.0
        (input_tokens, output_tokens,
         attention_mask) = self._maybe_sub_sample(
            x0, attention_mask)

        loss = self._forward_pass_diffusion(
            input_tokens,attention_mask,
            description_embed,description_attention_mask,
            sampling_eps_min=sampling_eps_min,
            sampling_eps_max=sampling_eps_max)

        if self.ignore_bos and not self.training:
            attention_mask[:, 0] = 0

        loss = loss * attention_mask
        loss = loss.sum() / attention_mask.sum()

        return loss

    def _maybe_sub_sample(self, x0, attention_mask):
        seqlen = x0.shape[1]
        if seqlen > self.num_tokens:
            assert seqlen == 2 * self.num_tokens
            start = np.random.choice(self.num_tokens)
            end = start + self.num_tokens
            input_tokens = x0[:, start: end]
            output_tokens = x0[:, start + 1: end + 1]
            new_attention_mask = attention_mask[:, start: end]

            if self.config.data.insert_train_special == True:
                input_tokens[:, 0] = self.tokenizer.bos_token_id
                output_tokens[:, -1] = self.tokenizer.eos_token_id
        else:
            input_tokens = x0
            output_tokens = None
            new_attention_mask = attention_mask
        return input_tokens, output_tokens, new_attention_mask

    def _forward_pass_diffusion(self, x0, attention_mask,description_embed=None, description_attention_mask=None, t=None, sampling_eps_min=None, sampling_eps_max=None):
        if t is None:
            t = self._sample_t(x0.shape,    
                               x0.device,
                               sampling_eps_min,
                               sampling_eps_max)

        _, p = self.noise(t)   

        xt = self.q_xt(x0,      
                       p,
                       sampling_eps_min=sampling_eps_min,
                       sampling_eps_max=sampling_eps_max)

        if self.ignore_bos:
            xt[:, 0] = x0[:, 0]

        x_input = xt
        if self.cross_attn:
            x_input = torch.cat((xt, x0), dim=-1)   
            attention_mask = attention_mask.repeat(1, 2)  # (1, 2*seq_len)

        model_output = self.forward(x_input,attention_mask,description_embed,description_attention_mask)   
        if torch.isnan(model_output).any():
            print(model_output, model_output)

        log_p_theta = torch.gather(     
            input=model_output,
            dim=-1,
            index=x0[:, :, None]).squeeze(-1)

        loss = -log_p_theta

        return loss

    def _sample_t(
            self, batch_dims, device, sampling_eps_min, sampling_eps_max, unit_size=None):
        if unit_size is None:
            unit_size = self.unit_size
        n = batch_dims[-1]
        num_units = n // unit_size
        _eps_b = torch.rand((batch_dims[0], num_units), device=device) 

        if self.antithetic_sampling:
            offset_b = torch.arange(batch_dims[0] * num_units, device=device) / (batch_dims[0] * num_units)   
            offset_b = offset_b.view(batch_dims[0], num_units) #
            _eps_b = (_eps_b / (batch_dims[0] * num_units) + offset_b) % 1 
        t = _eps_b
        if unit_size != self.config.model.length:
            t = t.repeat_interleave(unit_size, dim=-1) 

        # nll
        if sampling_eps_max >= 1 and sampling_eps_min >= 1: 
            return torch.ones_like(t)
        t = t * (sampling_eps_max - sampling_eps_min) + sampling_eps_min    
        return t

    def q_xt(self, x, p, unit_size=None, sampling_eps_min=None, sampling_eps_max=None):

        if unit_size is None:
            unit_size = self.unit_size

        move_indices = torch.rand(
            *x.shape, device=x.device) <= p     
        xt = torch.where(move_indices, self.mask_index, x)  

        if unit_size == 1 and sampling_eps_min == 1.0:
            return torch.full_like(x, self.mask_index)

        if self.config.training.resample and \
                not (sampling_eps_min == 1e-3 and sampling_eps_max == 1.0):
            xt = xt.reshape(xt.shape[0], -1, unit_size)
            xt = self._resample_q_xt(x,
                                     xt,
                                     move_indices,
                                     p,
                                     unit_size,
                                     sampling_eps_min,
                                     sampling_eps_max)
            xt = xt.reshape(xt.shape[0], -1)
        return xt

    def _resample_q_xt(
            self, x, xt, move_indices, p, unit_size, sampling_eps_min, sampling_eps_max):
        perc_masked = (xt == self.mask_index).float().sum(-1) / unit_size
        while (perc_masked < sampling_eps_min).any() or \
                (perc_masked > sampling_eps_max).any():
            # if a bound is epsilon, don't resample
            if sampling_eps_min == 1e-3 and sampling_eps_max != 1:
                regen_idx = (perc_masked > sampling_eps_max)
                if regen_idx.max() == 0:
                    break
            elif sampling_eps_min != 1e-3 and sampling_eps_max == 1:
                regen_idx = (perc_masked < sampling_eps_min)
                if regen_idx.max() == 0:
                    break
            elif sampling_eps_min != 1e-3 and sampling_eps_max != 1:
                regen_idx = (perc_masked < sampling_eps_min) | (perc_masked > sampling_eps_max)
            regen_idx = regen_idx.repeat_interleave(unit_size, dim=-1)
            move_indices[regen_idx] = (torch.rand(
                *x.shape, device=x.device) < p)[regen_idx]
            xt = torch.where(move_indices, self.mask_index, x)
            xt = xt.reshape(xt.shape[0], -1, unit_size)
            perc_masked = (xt == self.mask_index).float().sum(-1) / unit_size
        return xt




