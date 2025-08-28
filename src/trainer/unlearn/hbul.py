import geoopt
import ot
import torch.nn.functional as F
import torch.nn as nn
import torch
from typing import Dict, Optional, Any, Union, List, Tuple
from copy import deepcopy

from trainer.unlearn.base import UnlearnTrainer



class BusePenalty(nn.Module):
    """Busemann penalty function for hyperbolic geometry."""
    def __init__(self, dimension, mult=1.0):
        super(BusePenalty, self).__init__()
        self.dimension = dimension
        self.penalty_constant = mult * self.dimension

    def forward(self, z, p):
        # First part of loss: prediction difference
        prediction_difference = p - z
        difference_norm = torch.norm(prediction_difference, dim=1)
        difference_log = 2 * torch.log(difference_norm)

        # Second part of loss: prototype difference
        data_norm = torch.norm(z, dim=1)
        proto_difference = (1 - data_norm.pow(2) + 1e-6)
        proto_log = (1 + self.penalty_constant) * torch.log(proto_difference)

        one_loss = difference_log - proto_log
        total_loss = torch.mean(one_loss)

        return total_loss


def safe_log(x, eps=1e-8):
    """Safe logarithm function with epsilon clipping."""
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    return torch.log(torch.clamp(x, min=eps))


def safe_sqrt(x, eps=1e-8):
    """Safe square root function with epsilon clipping."""
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    return torch.sqrt(torch.clamp(x, min=eps))


def norm_clip(input_vector, r):
    """Clip input vector to have norm at most r."""
    input_norm = torch.norm(input_vector, dim=-1)
    clip_value = float(r) / input_norm
    min_norm = torch.clamp(float(r) / input_norm, max=5)
    return min_norm[:, None] * input_vector


def pot_sinkhorn(a, b, C, eps=0.1, max_iter=1000):
    """Sinkhorn algorithm for optimal transport."""
    if ot is None:
        raise ImportError("POT library required for optimal transport")
    
    a_np = a.detach().cpu().numpy()
    b_np = b.detach().cpu().numpy()
    C_np = C.detach().cpu().numpy()
    
    a_np = a_np / a_np.sum()
    b_np = b_np / b_np.sum()
    
    try:
        π_np = ot.sinkhorn(a_np, b_np, C_np, eps, numItermax=max_iter, verbose=False, log=False, warn=True)
    except Exception as e:
        print(f"Sinkhorn failed with error: {e}")
        print("Falling back to EMD...")
        π_np = ot.emd(a_np, b_np, C_np)
    
    π = torch.tensor(π_np, device=C.device, dtype=C.dtype)
    return π




def busemann_cost_matrix(x, xi, *, c=1.0, eps=1e-8, stabilize=True):
    """
    x:   [N, d]   points on the Poincaré ball
    xi:  [K, d]   prototypes on (r - eps) boundary
    c: curvature
    """
    r = 1.0 / safe_sqrt(c, eps)
    diff2  = ((x.unsqueeze(1) - xi.unsqueeze(0))**2).sum(-1).clamp_min(eps)            # [N,K]
    denom  = (r*r) - (x**2).sum(-1, keepdim=True)                        # [N,1]
    delta  = safe_log(diff2 + eps) - safe_log(denom + eps)               # [N,K]  can be ±

    if not stabilize:
        return delta

   
    row_min = delta.min(dim=1, keepdim=True).values                      # [N,1]
    C = delta - row_min                                                  # min in each row is 0
    C = torch.clamp(C, max=50.0)                                         # avoids exp overflow
    return C


# def busemann_cost_matrix(x, xi, *, c=1.0, eps=1e-8):
#     """Compute the pairwise Busemann distances between points in x and y."""
#     radius = 1.0 / safe_sqrt(c, eps)
#     diff2 = ((x.unsqueeze(1) - xi.unsqueeze(0))**2).sum(-1).clamp_min(eps)
#     denom = radius**2 - (x**2).sum(-1, keepdim=True)
#     C = (safe_log(diff2 + eps) - safe_log(denom + eps))
#     return C

class HBUL(UnlearnTrainer):
    """
    A custom UnlearnTrainer that implements machine unlearning using hyperbolic geometry
    and Busemann distances with optimal transport.
    """
    
    def __init__(self, *args, retain_prompts, lambda_hyp=1.0, lambda_ot=1.0, 
                 lambda_rep=1.0, margin=0.1, curvature=1, penalty_constant=0,
                 ot_eps=0.1, ot_max_iter=1000, use_attention_mask=True,
                 normalize_prototypes=True, clip_embeddings=True, **kwargs):
        """
        Initialize the hyperbolic Busemann trainer.
        
        Args:
            retain_prompts (list): A list of strings for concepts to be retained.
            lambda_hyp (float): Weight for the main hyperbolic loss.
            lambda_ot (float): Weight for the optimal transport loss.
            lambda_rep (float): Weight for the repulsive loss.
            margin (float): Margin for the repulsive hinge loss.
            curvature (float): Curvature of the Poincaré ball.
            penalty_constant (float): Multiplier for penalty constant in Busemann function.
            ot_eps (float): Epsilon for Sinkhorn algorithm.
            ot_max_iter (int): Maximum iterations for Sinkhorn algorithm.
            use_attention_mask (bool): Whether to use attention mask for prototype creation.
            normalize_prototypes (bool): Whether to normalize prototypes to boundary.
            clip_embeddings (bool): Whether to clip embeddings to unit norm.
            **kwargs: Additional arguments passed to UnlearnTrainer.
        """
        super().__init__(*args, **kwargs)
        
        # Hyperparameters for the unlearning loss
        # Ensure retain_prompts is properly formatted as a list of strings
        if isinstance(retain_prompts, (list, tuple)):
            self.retain_prompts = [str(prompt) for prompt in retain_prompts]
        elif hasattr(retain_prompts, '__iter__') and not isinstance(retain_prompts, str):
            self.retain_prompts = [str(prompt) for prompt in retain_prompts]
        else:
            raise ValueError(f"retain_prompts must be an iterable of strings, got {type(retain_prompts)}")
        
        # print(f'>>>>>>>>>> {self.retain_prompts}')
        # print(f'>>>>>>>>>> Type: {type(self.retain_prompts)}')
        # print(f'>>>>>>>>>> Length: {len(self.retain_prompts)}')
        # for i, prompt in enumerate(self.retain_prompts):
        #     print(f'>>>>>>>>>> Prompt {i}: {prompt} (type: {type(prompt)})')
        self.lambda_hyp = lambda_hyp
        self.lambda_ot = lambda_ot
        self.lambda_rep = lambda_rep
        self.margin = margin
        self.curvature = curvature
        self.penalty_constant = penalty_constant
        self.ot_eps = ot_eps
        self.ot_max_iter = ot_max_iter
        self.use_attention_mask = use_attention_mask
        self.normalize_prototypes = normalize_prototypes
        self.clip_embeddings = clip_embeddings

        print("HBUL __init__ arguments:")
        print(f"  retain_prompts: {self.retain_prompts}")
        print(f"  lambda_hyp: {self.lambda_hyp}")
        print(f"  lambda_ot: {self.lambda_ot}")
        print(f"  lambda_rep: {self.lambda_rep}")
        print(f"  margin: {self.margin}")
        print(f"  curvature: {self.curvature}")
        print(f"  penalty_constant: {self.penalty_constant}")
        print(f"  ot_eps: {self.ot_eps}")
        print(f"  ot_max_iter: {self.ot_max_iter}")
        print(f"  use_attention_mask: {self.use_attention_mask}")
        print(f"  normalize_prototypes: {self.normalize_prototypes}")
        print(f"  clip_embeddings: {self.clip_embeddings}")
        
        # Hyperbolic geometry components
        self.manifold = geoopt.PoincareBall(c=self.curvature)
        # BusePenalty expects (dimension, mult) where mult is the penalty_constant
        # We'll get the actual hidden dimension from the model config
        hidden_dim = getattr(self.model.config, 'hidden_size', 768)
        self.busemann_fn = BusePenalty(dimension=hidden_dim, mult=self.penalty_constant)
        
        # Get max sequence length from model config or args
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'max_position_embeddings'):
            self.max_seq_length = self.model.config.max_position_embeddings
        else:
            self.max_seq_length = getattr(self.args, 'max_seq_length', 512)
        
        # Pre-compute the ideal prototypes from the retain prompts
        self.ideal_prototypes = self._create_ideal_prototypes().to(self.args.device)
        
        # Store current losses for monitoring
        self.current_losses = {}

    def _create_ideal_prototypes(self):
        """
        Encode the retain prompts and map them to the boundary of the
        Poincaré ball to serve as fixed "ideal prototypes".
        """
        print("Creating ideal prototypes for retained concepts...")
        
        # Ensure model is in eval mode and no gradients are computed for this step
        self.model.eval()
        with torch.no_grad():
            # Use tokenizer to process retain prompts (types already validated in __init__)
            print(f"Processing {len(self.retain_prompts)} retain prompts for tokenization...")
            
            # Use tokenizer to process retain prompts
            tokenized_prompts = self.tokenizer(
                self.retain_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_seq_length
            )
            
            # Move tokenized prompts to the same device as the model
            model_device = next(self.model.parameters()).device
            print(f"Model device: {model_device}, Args device: {self.args.device}")
            tokenized_prompts = {k: v.to(model_device) for k, v in tokenized_prompts.items()}

            # Get embeddings from the base model for stability
            outputs = self.model(**tokenized_prompts, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]

            # Use attention mask for correct average pooling if enabled
            if self.use_attention_mask:
                attention_mask = tokenized_prompts['attention_mask'].unsqueeze(-1).expand(hidden_states.size()).float()
                sum_hidden_states = torch.sum(hidden_states * attention_mask, 1)
                sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
                euclidean_prototypes = sum_hidden_states / sum_mask
            else:
                # Simple average pooling without attention mask
                euclidean_prototypes = torch.mean(hidden_states, dim=1)
            
            # Clip embeddings if enabled
            if self.clip_embeddings:
                euclidean_prototypes = norm_clip(euclidean_prototypes, 5)
            
            # Map to hyperbolic space
            hyperbolic_prototypes = self.manifold.expmap0(euclidean_prototypes)
            
            # Normalize to boundary if enabled
            if self.normalize_prototypes:
                ideal_prototypes = F.normalize(hyperbolic_prototypes, p=2, dim=1)* (1 - 1e-3)
            else:
                ideal_prototypes = hyperbolic_prototypes

        # Return model to train mode
        self.model.train()
        print(f"Successfully created {len(ideal_prototypes)} ideal prototypes.")
        return ideal_prototypes

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Calculate the combined hyperbolic unlearning loss.
        
        Args:
            model: The model to compute loss for.
            inputs: Input dictionary containing labels and other inputs.
            return_outputs: Whether to return outputs along with loss.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Tuple of (loss, outputs) if return_outputs=True, else just loss.
        """
       
        model_inputs = inputs["forget"]
        labels = model_inputs.get("labels")

            
        outputs = model(**model_inputs, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]
        
       
        ans_token = "<|ANS|>"
        ans_token_id = self.tokenizer.convert_tokens_to_ids(ans_token)
       
        ans_token_mask = (labels == ans_token_id)
        ans_token_indices = ans_token_mask.float().argmax(dim=1)
        batch_indices = torch.arange(labels.size(0), device=labels.device)
        forget_embeddings_euclidean = last_hidden_states[batch_indices, ans_token_indices]
        # print(ans_token_mask ,forget_embeddings_euclidean.shape)

    


        # Get the contextual embeddings just before the answer starts
        if self.clip_embeddings:
            forget_embeddings_euclidean = norm_clip(forget_embeddings_euclidean, 5)

        # 3. Map forget embeddings to hyperbolic space
        z = self.manifold.expmap0(forget_embeddings_euclidean)
        p = self.ideal_prototypes
        num_forget, num_protos = z.shape[0], p.shape[0]

        # 4. Calculate the Busemann distance matrix (cost matrix for OT)
        cost_matrix = busemann_cost_matrix(z, p)

        # 5. Optimal Transport Loss
        transport_plan = pot_sinkhorn(
            torch.ones(num_forget, device=z.device) / num_forget,
            torch.ones(num_protos, device=z.device) / num_protos,
            cost_matrix.detach(),  # Detach to avoid gradients flowing through OT solver
            eps=self.ot_eps,
            max_iter=self.ot_max_iter
        )
        loss_ot = torch.sum(transport_plan * cost_matrix)

        # 6. Hyperbolic (Attraction) Loss
        assigned_indices = torch.argmax(transport_plan, dim=1)
        assigned_prototypes = p[assigned_indices]
        loss_hyp = self.busemann_fn(z, assigned_prototypes).mean()

        # # 7. Repulsive Loss (Retain Regularizer, following lora_hyp.py)
        # # Use hard assignment for each sample from soft OT
        # assigned_k = transport_plan.argmax(dim=1)  # [num_forget]

        # # Create mask for non-assigned prototypes
        # mask = torch.ones_like(cost_matrix, dtype=torch.bool)
        # mask[torch.arange(num_forget), assigned_k] = False

        # # Mask out assigned prototypes
        # C_nonassigned = cost_matrix[mask].view(num_forget, num_protos - 1)

        # # Apply clamped margin penalty
        # loss_rep = torch.clamp(self.margin - C_nonassigned, min=0).mean()

        assigned_k = transport_plan.argmax(dim=1)
        C_assigned = cost_matrix[torch.arange(num_forget), assigned_k].unsqueeze(1)  # [B,1]

        C_masked = cost_matrix.clone()
        C_masked[torch.arange(num_forget), assigned_k] = float('inf')
        topk_vals, _ = C_masked.topk(k=min(5, num_protos - 1), dim=1, largest=False)  # closest non-assigned

        loss_rep = torch.clamp(self.margin + C_assigned - topk_vals, min=0).mean()

        # 8. Combine the three loss components
        total_loss = (self.lambda_hyp * loss_hyp +
                      self.lambda_ot * loss_ot +
                      self.lambda_rep * loss_rep)
        
        # # Store current losses for monitoring
        # self.current_losses = {
        #     'total_loss': total_loss.item(),
        #     'hyperbolic_loss': loss_hyp.item(),
        #     'optimal_transport_loss': loss_ot.item(),
        #     'repulsive_loss': loss_rep.item(),
        #     'lambda_hyp': self.lambda_hyp,
        #     'lambda_ot': self.lambda_ot,
        #     'lambda_rep': self.lambda_rep,
        #     'curvature': self.curvature,
        #     'penalty_constant': self.penalty_constant,
        #     'ot_eps': self.ot_eps,
        #     'ot_max_iter': self.ot_max_iter,
        #     'use_attention_mask': self.use_attention_mask,
        #     'normalize_prototypes': self.normalize_prototypes,
        #     'clip_embeddings': self.clip_embeddings
        # }
        
        # Log losses at specified intervals
        if self.state.global_step % self.args.logging_steps == 0:
            self.log({
                "total_loss": total_loss.detach().item(),
                "hyperbolic_loss": loss_hyp.detach().item(),
                "optimal_transport_loss": loss_ot.detach().item(),
                "repulsive_loss": loss_rep.detach().item(),
            })

        return (total_loss, outputs) if return_outputs else total_loss

    def get_current_losses(self):
        """Get the current loss values for monitoring."""
        return self.current_losses.copy()

    def update_hyperparameters(self, lambda_hyp=None, lambda_ot=None, lambda_rep=None, margin=None,
                             curvature=None, penalty_constant=None, ot_eps=None, ot_max_iter=None,
                             use_attention_mask=None, normalize_prototypes=None, clip_embeddings=None):
        """Update hyperparameters during training if needed."""
        if lambda_hyp is not None:
            self.lambda_hyp = lambda_hyp
        if lambda_ot is not None:
            self.lambda_ot = lambda_ot
        if lambda_rep is not None:
            self.lambda_rep = lambda_rep
        if margin is not None:
            self.margin = margin
        if curvature is not None:
            self.curvature = curvature
            # Update manifold if curvature changes
            self.manifold = geoopt.PoincareBall(c=self.curvature)
        if penalty_constant is not None:
            self.penalty_constant = penalty_constant
            hidden_dim = getattr(self.model.config, 'hidden_size', 768)
            self.busemann_fn = BusePenalty(dimension=hidden_dim, mult=self.penalty_constant)
        if ot_eps is not None:
            self.ot_eps = ot_eps
        if ot_max_iter is not None:
            self.ot_max_iter = ot_max_iter
        if use_attention_mask is not None:
            self.use_attention_mask = use_attention_mask
        if normalize_prototypes is not None:
            self.normalize_prototypes = normalize_prototypes
        if clip_embeddings is not None:
            self.clip_embeddings = clip_embeddings
