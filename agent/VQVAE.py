import torch
from abc import ABC, abstractmethod
import torch.nn as nn
'''
VQ-VAE implementation taken from https://github.com/victorcampos7/edl
'''
def create_nn(input_size, output_size, hidden_size, num_layers, activation_fn=nn.ReLU, input_normalizer=None,
              final_activation_fn=None, hidden_init_fn=None, b_init_value=None, last_fc_init_w=None):
    # Optionally add a normalizer as the first layer
    if input_normalizer is None:
        input_normalizer = nn.Sequential()
    layers = [input_normalizer]

    # Create and initialize all layers except the last one
    for layer_idx in range(num_layers - 1):
        fc = nn.Linear(input_size if layer_idx == 0 else hidden_size, hidden_size)
        if hidden_init_fn is not None:
            hidden_init_fn(fc.weight)
        if b_init_value is not None:
            fc.bias.data.fill_(b_init_value)
        layers += [fc, activation_fn()]

    # Create and initialize  the last layer
    last_fc = nn.Linear(hidden_size, output_size)
    if last_fc_init_w is not None:
        last_fc.weight.data.uniform_(-last_fc_init_w, last_fc_init_w)
        last_fc.bias.data.uniform_(-last_fc_init_w, last_fc_init_w)
    layers += [last_fc]

    # Optionally add a final activation function
    if final_activation_fn is not None:
        layers += [final_activation_fn()]
    return nn.Sequential(*layers)

class DensityModule(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def novelty(self, *args, **kwargs):
        return torch.zeros(10)

class BaseVAEDensity(nn.Module, DensityModule):
    def __init__(self, num_skills, state_size, hidden_size, code_size,
                 num_layers=4, normalize_inputs=False, skill_preprocessing_fn=lambda x: x,
                 input_key='next_state', input_size=None):
        super().__init__()

        self.num_skills = int(num_skills)
        self.state_size = int(state_size) if input_size is None else int(input_size)
        self.code_size = int(code_size)
        self.normalize_inputs = bool(normalize_inputs)
        self.skill_preprocessing_fn = skill_preprocessing_fn
        self.input_key = str(input_key)

        self._make_normalizer_module()

        assert num_layers >= 2
        self.num_layers = int(num_layers)

        self.encoder = create_nn(input_size=self.input_size, output_size=self.encoder_output_size,
                                 hidden_size=hidden_size, num_layers=self.num_layers,
                                 input_normalizer=self.normalizer if self.normalizes_inputs else nn.Sequential())

        self.decoder = create_nn(input_size=self.code_size, output_size=self.input_size,
                                 hidden_size=hidden_size, num_layers=self.num_layers)

        self.mse_loss = nn.MSELoss(reduction='none')

    @property
    def input_size(self):
        return self.state_size + self.num_skills

    @property
    def encoder_output_size(self):
        return NotImplementedError

    @property
    def normalizes_inputs(self):
        return self.normalizer is not None

    def _make_normalizer_module(self):
        raise NotImplementedError

    def compute_logprob(self, batch, **kwargs):
        raise NotImplementedError

    def novelty(self, batch, **kwargs):
        with torch.no_grad():
            return -self.compute_logprob(batch, **kwargs).detach()

    def update_normalizer(self, **kwargs):
        if self.normalizes_inputs:
            self.normalizer.update(**kwargs)

    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath)
        self.load_state_dict(checkpoint)

    def forward(self, batch):
        raise NotImplementedError


class VQVAEDensity(BaseVAEDensity):
    def __init__(self, num_skills, state_size, hidden_size, codebook_size, code_size, beta=0.25, **kwargs):
        super().__init__(num_skills=num_skills, state_size=state_size, hidden_size=hidden_size, code_size=code_size,
                         **kwargs)
        self.codebook_size = int(codebook_size)
        self.beta = float(beta)

        self.apply(self.weights_init)

        self.vq = VQEmbedding(self.codebook_size, self.code_size, self.beta)
    
    @property
    def encoder_output_size(self):
        return self.code_size

    def _make_normalizer_module(self):
        self.normalizer = Normalizer(self.input_size) if self.normalize_inputs else None

    @classmethod
    def weights_init(cls, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            try:
                nn.init.xavier_uniform_(m.weight.data)
                m.bias.data.fill_(0)
            except AttributeError:
                print("Skipping initialization of ", classname)

    def compute_logprob(self, batch, with_codes=False):
        s, z = batch[self.input_key], self.skill_preprocessing_fn(batch['skill'])
        x = torch.cat([s, z], dim=1)
        z_e_x = self.encoder(x)
        z_q_x, selected_codes = self.vq.straight_through(z_e_x)
        x_ = self.decoder(z_q_x)
        if self.normalizes_inputs:
            x_ = self.normalizer.denormalize(x_)
        logprob = -1. * self.mse_loss(x, x_).sum(dim=1)
        if with_codes:
            return logprob, z_e_x, selected_codes
        else:
            return logprob

    def get_centroids(self, batch):
        z_idx = batch['skill']
        z_q_x = torch.index_select(self.vq.embedding.weight.detach(), dim=0, index=z_idx)
        centroids = self.decoder(z_q_x)
        if self.normalizes_inputs:
            centroids = self.normalizer.denormalize(centroids)
        return centroids

    def novelty(self, batch, **kwargs):
        with torch.no_grad():
            return -self.compute_logprob(batch, with_codes=False).detach()

    def forward(self, batch, weights=None):
        logprob, z_e_x, selected_codes = self.compute_logprob(batch, with_codes=True)
        loss = self.vq(z_e_x, selected_codes) - logprob
        if weights != None:
            loss = loss*weights
        return loss.mean()



class VQVAEDiscriminator(VQVAEDensity):
    def __init__(self, state_size, hidden_size, codebook_size, code_size, beta=0.25, **kwargs):
        super().__init__(num_skills=0, state_size=state_size, hidden_size=hidden_size, codebook_size=codebook_size,
                         code_size=code_size, beta=beta, **kwargs)
        self.softmax = nn.Softmax(dim=1)

    def _make_normalizer_module(self):
        self.normalizer = DatasetNormalizer(self.input_size) if self.normalize_inputs else None

    def compute_logprob(self, batch, with_codes=False):
        x = batch[self.input_key]
        z_e_x = self.encoder(x)
        z_q_x, selected_codes = self.vq.straight_through(z_e_x)
        x_ = self.decoder(z_q_x)
        if self.normalizes_inputs:
            x_ = self.normalizer.denormalize(x_)

         
        logprob = -1. * self.mse_loss(x, x_).sum(dim=1)

        if with_codes:
            return logprob, z_e_x, selected_codes
        else:
            return logprob



    def compute_logprob_under_latent(self, batch, z=None):
        x = batch[self.input_key]
        if z is None:
            z = batch['skill']
        z_q_x = self.vq.embedding(z).detach()
        x_ = self.decoder(z_q_x).detach()
        if self.normalizes_inputs:
            x_ = self.normalizer.denormalize(x_)

        logprob = -1. * self.mse_loss(x, x_).sum(dim=1)
        return logprob

    def log_approx_posterior(self, batch):
        x, z = batch[self.input_key], batch['skill']
        z_e_x = self.encoder(x)
        codebook_distances = self.vq.compute_distances(z_e_x)
        p = self.softmax(codebook_distances)
        p_z = p[torch.arange(p.shape[0]), z]
        return torch.log(p_z)

    def surprisal(self, batch):
        with torch.no_grad():
            return self.compute_logprob_under_latent(batch).detach()

    def get_goal_under_latent(self, batch):
        z = batch['skill']
        z_q_x = self.vq.embedding(z).detach()
        x_ = self.decoder(z_q_x).detach()
        if self.normalizes_inputs:
            x_ = self.normalizer.denormalize(x_)
        return x_

    def save(self, model_dir, step):
        torch.save(
            self.decoder.state_dict(), '%s/discri_%s.pt' % (model_dir, step)
        )
    def load(self, model_dir, step):
            self.decoder.load_state_dict(
                torch.load('%s/discri_%s.pt' % (model_dir, step))
            )
class Normalizer(nn.Module):
    def __init__(self, input_size, epsilon=0.01, zero_mean=False, momentum=None, extra_dims=0):
        super(Normalizer, self).__init__()

        self.input_size = int(input_size)
        self.epsilon = float(epsilon)
        self.zero_mean = bool(zero_mean)
        self.momentum = 1 if momentum is None else min(1, max(0, momentum))
        self.extra_dims = int(extra_dims)

        self.register_buffer('running_sum', torch.zeros(self.input_size))
        self.register_buffer('running_sumsq', torch.zeros(self.input_size) + self.epsilon)
        self.register_buffer('count', torch.zeros(1) + self.epsilon)

    @property
    def mean(self):
        if self.zero_mean:
            return torch.zeros_like(self.running_sum).view(1, self.input_size)
        m = self.running_sum / self.count
        return m.view(1, self.input_size).detach()

    @property
    def std(self):
        var = (self.running_sumsq / self.count) - torch.pow(self.mean, 2)
        var = var.masked_fill(var < self.epsilon, self.epsilon)
        std = torch.pow(var, 0.5)
        return std.view(1, self.input_size).detach()

    def split(self, x):
        if self.extra_dims == 0:
            return x, None
        return x[:, :-self.extra_dims], x[:, -self.extra_dims:]

    def join(self, x1, x2):
        if self.extra_dims == 0:
            return x1
        return torch.cat([x1, x2], dim=1)

    def update(self, x):
        self.running_sum *= self.momentum
        self.running_sumsq *= self.momentum
        self.count *= self.momentum

        x = x.view(-1, self.input_size)
        self.running_sum += x.sum(dim=0).detach()
        self.running_sumsq += torch.pow(x, 2).sum(dim=0).detach()
        self.count += x.shape[0]

    def forward(self, x):
        x1, z2 = self.split(x)
        z1 = (x1 - self.mean) / self.std
        if self.training:
            self.update(x1)
        z1 = torch.clamp(z1, -5.0, 5.0)
        return self.join(z1, z2)

    def denormalize(self, z):
        z1, x2 = self.split(z)
        x1 = z1 * self.std + self.mean
        return self.join(x1, x2)


class DatasetNormalizer(nn.Module):
    def __init__(self, input_size, epsilon=0.01, zero_mean=False):
        super(DatasetNormalizer, self).__init__()

        self.input_size = input_size
        self.epsilon = epsilon ** 2  # for consistency with Normalizer
        self.zero_mean = bool(zero_mean)

        self.register_buffer('mean_buffer', torch.zeros(self.input_size))
        self.register_buffer('std_buffer', torch.full((self.input_size,), epsilon))

    def update(self, dataset=None, mean=None, std=None):
        if dataset is None:
            assert std is not None
            std = std.masked_fill(std < self.epsilon, self.epsilon)
            if self.zero_mean:
                mean = torch.zeros_like(std)
            else:
                assert mean is not None
        else:
            assert mean is None
            assert std is None
            std = dataset.std(dim=0)
            mean = dataset.mean(dim=0) if not self.zero_mean else torch.zeros_like(std)

        self.mean_buffer = mean
        self.std_buffer = std

    @property
    def mean(self):
        return self.mean_buffer.detach()

    @property
    def std(self):
        return self.std_buffer.detach()

    def forward(self, x):
        z = (x - self.mean) / self.std
        return torch.clamp(z, -5.0, 5.0)

    def denormalize(self, x):
        return x * self.std + self.mean

from torch.autograd import Function


class VectorQuantization(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        with torch.no_grad():
            embedding_size = codebook.size(1)
            inputs_size = inputs.size()
            inputs_flatten = inputs.view(-1, embedding_size)

            codebook_sqr = torch.sum(codebook ** 2, dim=1)
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)

            # Compute the distances to the codebook
            distances = torch.addmm(codebook_sqr + inputs_sqr, inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)

            _, indices_flatten = torch.min(distances, dim=1)
            indices = indices_flatten.view(*inputs_size[:-1])
            ctx.mark_non_differentiable(indices)

            return indices

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError('Trying to call `.grad()` on graph containing '
                           '`VectorQuantization`. The function `VectorQuantization` '
                           'is not differentiable. Use `VectorQuantizationStraightThrough` '
                           'if you want a straight-through estimator of the gradient.')


class VectorQuantizationStraightThrough(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        indices = vector_quantization(inputs, codebook)
        indices_flatten = indices.view(-1)
        ctx.save_for_backward(indices_flatten, codebook)
        ctx.mark_non_differentiable(indices_flatten)

        codes_flatten = torch.index_select(codebook, dim=0, index=indices_flatten)
        codes = codes_flatten.view_as(inputs)

        return (codes, indices_flatten)

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        grad_inputs, grad_codebook = None, None

        if ctx.needs_input_grad[0]:
            # Straight-through estimator
            grad_inputs = grad_output.clone()
        if ctx.needs_input_grad[1]:
            # Gradient wrt. the codebook
            indices, codebook = ctx.saved_tensors
            embedding_size = codebook.size(1)

            grad_output_flatten = (grad_output.contiguous().view(-1, embedding_size))
            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices, grad_output_flatten)

        return (grad_inputs, grad_codebook)


vector_quantization = VectorQuantization.apply
vector_quantization_st = VectorQuantizationStraightThrough.apply
__all__ = [vector_quantization, vector_quantization_st]


class VQEmbedding(nn.Module):
    """
    Vector Quantization module for VQ-VAE (van der Oord et al., https://arxiv.org/abs/1711.00937)
    This module is compatible with 1D latents only (i.e. with inputs of shape [batch_size, embedding_dim]).
    Adapted from https://github.com/ritheshkumar95/pytorch-vqvae/blob/master/modules.py#L70
    Variable names follow those in the paper:
        z_e_x: z_e(x), i.e. the *continuous* encoding emitted by the encoder
        z_q_x: z_q(x), i.e. the decoder input -- the vector-quantized version of z_e(x)  [Eq. 2]
    """
    def __init__(self, codebook_size, code_size, beta):
        """
        :param codebook_size: number of codes in the codebook
        :param code_size: dimensionality of each code
        :param beta: weight for the commitment loss
        """
        super().__init__()

        self.codebook_size = int(codebook_size)
        self.code_size = int(code_size)
        self.beta = float(beta)

        self.embedding = nn.Embedding(self.codebook_size, self.code_size)
        self.embedding.weight.data.uniform_(-1./self.codebook_size, 1./self.codebook_size)

        self.mse_loss = nn.MSELoss(reduction='none')

    def quantize(self, z_e_x):
        return vector_quantization(z_e_x, self.embedding.weight)

    def straight_through(self, z_e_x):
        # Quantized vectors (inputs for the decoder)
        z_q_x, indices = vector_quantization_st(z_e_x, self.embedding.weight.detach())
        # Selected codes from the codebook (for the VQ objective)
        selected_codes = torch.index_select(self.embedding.weight, dim=0, index=indices)
        return z_q_x, selected_codes

    def forward(self, z_e_x, selected_codes=None):
        """
        Compute second and third loss terms in Eq. 3 in the paper
        :param z_e_x: encoder output
        :param selected_codes: (optional) second output from straight_through(); avoids recomputing it
        :return: loss = vq_loss + beta * commitment_loss
        """
        # Recompute z_q(x) if needed
        if selected_codes is None:
            _, selected_codes = self.straight_through(z_e_x)
        # Push VQ codes towards the output of the encoder
        vq_loss = self.mse_loss(selected_codes, z_e_x.detach()).sum(dim=1)
        # Encourage the encoder to commit to a code
        commitment_loss = self.mse_loss(z_e_x, selected_codes.detach()).sum(dim=1)
        # The scale of the commitment loss is controlled with beta [Eq. 3]
        loss = vq_loss + self.beta * commitment_loss
        return loss

    def compute_distances(self, inputs):
        with torch.no_grad():
            embedding_size = self.embedding.weight.size(1)
            inputs_flatten = inputs.view(-1, embedding_size)

            codebook_sqr = torch.sum(self.embedding.weight ** 2, dim=1)
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)

            # Compute the distances to the codebook
            distances = torch.addmm(codebook_sqr + inputs_sqr, inputs_flatten, self.embedding.weight.t(),
                                    alpha=-2.0, beta=1.0)

            return distances