import geoopt
import ot
import torch.nn.functional as F
import torch.nn as nn
import torch
from typing import Dict, Optional, Any, Union, List, Tuple
from copy import deepcopy
import os
import datetime

from trainer.unlearn.base import UnlearnTrainer

# Set up a global log file path for loss logging
HYDRA_LOG_DIR = "/home/nilakshan/0-Unlearning/0-open-unlearning-dev/saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget01_HBULBase/.hydra"
LOSS_LOG_FILE = os.path.join(HYDRA_LOG_DIR, "hbul_loss_log.txt")

def log_losses_to_file(step, loss_hyp, loss_ot, loss_rep, total_loss, params: dict):
    """
    Log the hyperbolic and OT loss (and optionally other parameters) to a file.
    """
    # Ensure the log directory exists
    os.makedirs(HYDRA_LOG_DIR, exist_ok=True)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOSS_LOG_FILE, "a") as f:
        f.write(f"[{now}] step={step} | hyp_loss={loss_hyp:.6f} | ot_loss={loss_ot:.6f} | rep_loss={loss_rep:.6f} | total_loss={total_loss:.6f}\n")
        for k, v in params.items():
            f.write(f"    {k}: {v}\n")
        f.write("\n")

class BusePenalty(nn.Module):
    """Busemann penalty function for hyperbolic geometry."""
    def __init__(self, dimension, c=1.0,mult=0):
        super(BusePenalty, self).__init__()
        self.dimension = dimension
        self.penalty_constant = mult * self.dimension
        self.c = c 
        

    def forward(self, z, p):
        # First part of loss: prediction difference
        rsqr = 1.0 / float(self.c)
        prediction_difference = p - z
        difference_norm = torch.norm(prediction_difference, dim=1)
        difference_log = 2 * torch.log(difference_norm)

        # Second part of loss: prototype difference
        data_norm = torch.norm(z, dim=1)
        proto_difference = (rsqr - data_norm.pow(2) + 1e-6)
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
    min_norm = torch.clamp(float(r) / input_norm, max=1)
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
    # Poincaré ball radius r = 1/sqrt(c)
    r2 = 1.0 / float(c)  # (1/√c)^2
    diff2  = ((x.unsqueeze(1) - xi.unsqueeze(0))**2).sum(-1).clamp_min(eps)   # [N,K]
    denom  = (r2 - (x**2).sum(-1, keepdim=True)).clamp_min(eps)               # [N,1]
    delta  = torch.log(diff2) - torch.log(denom)                               # [N,K
    
    # C = torch.clamp(delta**2, max=50.0)   # stabilize for Sinkhorn
    C = delta**2
    return C

    

class HBULBase(UnlearnTrainer):
    """
    A custom UnlearnTrainer that implements machine unlearning using hyperbolic geometry
    and Busemann distances with optimal transport.
    """
    
    def __init__(self, *args, retain_prompts, lambda_hyp=1.0, lambda_ot=1.0, 
                 lambda_rep=1.0, lambda_concept=2.0, lambda_adv=1.5, lambda_boundary=0.5,
                 margin=0.1, curvature=1, penalty_constant=0,
                 ot_eps=0.1, ot_max_iter=1000, use_attention_mask=True,
                 normalize_prototypes=True, clip_embeddings=True, 
                 use_multi_position_prototypes=True, concept_temperature=0.1,
                 boundary_push_strength=1.0, **kwargs):
        """
        Initialize the hyperbolic Busemann trainer.
        
        Args:
            retain_prompts (list): A list of strings for concepts to be retained.
            lambda_hyp (float): Weight for the main hyperbolic loss.
            lambda_ot (float): Weight for the optimal transport loss.
            lambda_rep (float): Weight for the repulsive loss.
            lambda_concept (float): Weight for concept-level unlearning loss.
            lambda_adv (float): Weight for adversarial unlearning loss.
            lambda_boundary (float): Weight for boundary push loss.
            margin (float): Margin for the repulsive hinge loss.
            curvature (float): Curvature of the Poincaré ball.
            penalty_constant (float): Multiplier for penalty constant in Busemann function.
            ot_eps (float): Epsilon for Sinkhorn algorithm.
            ot_max_iter (int): Maximum iterations for Sinkhorn algorithm.
            use_attention_mask (bool): Whether to use attention mask for prototype creation.
            normalize_prototypes (bool): Whether to normalize prototypes to boundary.
            clip_embeddings (bool): Whether to clip embeddings to unit norm.
            use_multi_position_prototypes (bool): Whether to use multiple token positions for prototypes.
            concept_temperature (float): Temperature for concept-level unlearning.
            boundary_push_strength (float): Strength of boundary push loss.
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
        
        self.lambda_hyp = lambda_hyp
        self.lambda_ot = lambda_ot
        self.lambda_rep = lambda_rep
        self.lambda_concept = lambda_concept
        self.lambda_adv = lambda_adv
        self.lambda_boundary = lambda_boundary
        self.margin = margin
        self.curvature = curvature
        self.penalty_constant = penalty_constant
        self.ot_eps = ot_eps
        self.ot_max_iter = ot_max_iter
        self.use_attention_mask = use_attention_mask
        self.normalize_prototypes = normalize_prototypes
        self.clip_embeddings = clip_embeddings
        self.use_multi_position_prototypes = use_multi_position_prototypes
        self.concept_temperature = concept_temperature
        self.boundary_push_strength = boundary_push_strength

        print("HBUL __init__ arguments:")
        print(f"  retain_prompts: {self.retain_prompts}")
        print(f"  lambda_hyp: {self.lambda_hyp}")
        print(f"  lambda_ot: {self.lambda_ot}")
        print(f"  lambda_rep: {self.lambda_rep}")
        print(f"  lambda_concept: {self.lambda_concept}")
        print(f"  lambda_adv: {self.lambda_adv}")
        print(f"  lambda_boundary: {self.lambda_boundary}")
        print(f"  margin: {self.margin}")
        print(f"  curvature: {self.curvature}")
        print(f"  penalty_constant: {self.penalty_constant}")
        print(f"  ot_eps: {self.ot_eps}")
        print(f"  ot_max_iter: {self.ot_max_iter}")
        print(f"  use_attention_mask: {self.use_attention_mask}")
        print(f"  normalize_prototypes: {self.normalize_prototypes}")
        print(f"  clip_embeddings: {self.clip_embeddings}")
        print(f"  use_multi_position_prototypes: {self.use_multi_position_prototypes}")
        print(f"  concept_temperature: {self.concept_temperature}")
        print(f"  boundary_push_strength: {self.boundary_push_strength}")
        
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
        Uses multiple token positions for better semantic representation.
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
                max_length=self.max_seq_length,
                add_special_tokens=False
            )
            
            print('>>>>>>>>>>>>>>>>>')
            for tok in tokenized_prompts['input_ids'][0]:
                print(tok,self.tokenizer.decode(tok))
            print('>>>>>>>>>>>>>>>>>')
            # Move tokenized prompts to the same device as the model
            model_device = next(self.model.parameters()).device
            print(f"Model device: {model_device}, Args device: {self.args.device}")
            tokenized_prompts = {k: v.to(model_device) for k, v in tokenized_prompts.items()}

            # Get embeddings from the base model for stability
            outputs = self.model(**tokenized_prompts, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]

            if self.use_multi_position_prototypes:
                # Use multiple token positions for richer semantic representation
                euclidean_prototypes = self._extract_multi_position_prototypes(
                    hidden_states, tokenized_prompts['input_ids']
                )
            else:
                # Original single position approach
                euclidean_prototypes = self._extract_single_position_prototypes(
                    hidden_states, tokenized_prompts['input_ids']
                )
            
            # Clip embeddings if enabled
            if self.clip_embeddings:
                euclidean_prototypes = norm_clip(euclidean_prototypes, 5)
            
            # Map to hyperbolic space
            hyperbolic_prototypes = self.manifold.expmap0(euclidean_prototypes)
            
            # Push prototypes closer to boundary for stronger unlearning signal
            ideal_prototypes = hyperbolic_prototypes * (1 - 1e-4)  # Closer to boundary
            
            # Print prototype statistics for debugging
            prototype_norms = torch.norm(ideal_prototypes, dim=1)
            print(f"Prototype norms - min: {prototype_norms.min():.4f}, max: {prototype_norms.max():.4f}, mean: {prototype_norms.mean():.4f}")
            print(f"Prototype positions: {ideal_prototypes.shape}")

        # Return model to train mode
        self.model.train()
        print(f"Successfully created {len(ideal_prototypes)} ideal prototypes.")
        return ideal_prototypes.float()

    def _extract_single_position_prototypes(self, hidden_states, input_ids):
        """Extract prototypes from single token position (original method)."""
        eot_token_id = self.tokenizer.eos_token_id
        if eot_token_id is None:
            raise ValueError("Tokenizer does not have an eos_token_id or eot_token_id.")
        
        batch_size, seq_len = input_ids.shape
        eot_indices = []
        for i in range(batch_size):
            ids = input_ids[i].tolist()
            try:
                eot_pos = ids.index(eot_token_id)
                proto_pos = max(eot_pos - 1, 0)
            except ValueError:
                proto_pos = seq_len - 1
            eot_indices.append(proto_pos)
        
        return hidden_states[torch.arange(batch_size), torch.tensor(eot_indices, device=hidden_states.device)]

    def _extract_multi_position_prototypes(self, hidden_states, input_ids):
        """Extract prototypes from multiple token positions for richer representation."""
        batch_size, seq_len = hidden_states.shape[:2]
        attention_mask = (input_ids != self.tokenizer.pad_token_id).float()
        
        # Use attention-weighted pooling over all non-padding tokens
        # This captures more semantic information than single position
        attention_weights = attention_mask.unsqueeze(-1)  # [batch_size, seq_len, 1]
        weighted_hidden = hidden_states * attention_weights
        pooled_hidden = weighted_hidden.sum(dim=1) / attention_weights.sum(dim=1).clamp(min=1)
        
        return pooled_hidden

    # Removed _concept_level_unlearning_loss and _adversarial_unlearning_loss

    def _update_prototypes(self, update_frequency=100):
        """Update prototypes during training for better adaptation."""
        if self.state.global_step % update_frequency == 0:
            with torch.no_grad():
                # Re-compute prototypes with current model
                new_prototypes = self._create_ideal_prototypes().to(self.args.device)
                # Exponential moving average update
                alpha = 0.1
                self.ideal_prototypes = (1 - alpha) * self.ideal_prototypes + alpha * new_prototypes

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
        
       
        input_ids = model_inputs['input_ids']
        eot_token_id = self.tokenizer.eos_token_id
        eos_mask = (input_ids == eot_token_id)
        
        
        eos_positions = torch.nonzero(eos_mask, as_tuple=False)

        self._update_prototypes()
        
        if len(eos_positions) == 0:
            # No EOS token found, use the last token
            print('>>>>>>>> USING last token')
            last_eos_token_indices = torch.full((input_ids.size(0),), input_ids.size(1) - 1, device=input_ids.device)
        else:
            # For each batch item, find the EOS token that's most likely to end the question
            last_eos_token_indices = []
            for batch_idx in range(input_ids.size(0)):
                batch_eos_positions = eos_positions[eos_positions[:, 0] == batch_idx, 1]
                if len(batch_eos_positions) > 0:
                    # Use the last EOS token in the sequence for this batch item
                    last_eos_token_indices.append(batch_eos_positions[-1].item())
                else:
                    last_eos_token_indices.append(input_ids.size(1) - 1)
            last_eos_token_indices = torch.tensor(last_eos_token_indices, device=input_ids.device)

        batch_indices = torch.arange(labels.size(0), device=labels.device)
        forget_embeddings_euclidean = last_hidden_states[batch_indices, last_eos_token_indices]
        
        token_ids_at_indices = input_ids[torch.arange(input_ids.size(0)), last_eos_token_indices]
        # print("last_eos_token_indices:", last_eos_token_indices)
        # print("token_ids_at_indices:", token_ids_at_indices)

        
        # Get the contextual embeddings just before the answer starts
        if self.clip_embeddings:
            forget_embeddings_euclidean = norm_clip(forget_embeddings_euclidean, 2)

        # 3. Map forget embeddings to hyperbolic space
        z = self.manifold.expmap0(forget_embeddings_euclidean)
        p = self.ideal_prototypes

        
        avg_norm_z = z.norm(dim=1).mean().item()
        avg_norm_p = p.norm(dim=1).mean().item()
        print(f"Average norm of z (forget embeddings in hyperbolic space): {avg_norm_z:.4f}")
        print(f"Average norm of p (ideal prototypes in hyperbolic space): {avg_norm_p:.4f}")
        print(f"Z shape: {z.shape}, P shape: {p.shape}")
        num_forget, num_protos = z.shape[0], p.shape[0]

        # 4. Calculate the Busemann distance matrix (cost matrix for OT)
        cost_matrix = busemann_cost_matrix(z, p)
        cost_matrix = (cost_matrix - cost_matrix.min()) / (cost_matrix.max() - cost_matrix.min() + 1e-8)
        

        # 5. Optimal Transport Loss
        transport_plan = pot_sinkhorn(
            torch.ones(num_forget, device=z.device) / num_forget,
            torch.ones(num_protos, device=z.device) / num_protos,
            cost_matrix,  
            eps=self.ot_eps,
            max_iter=self.ot_max_iter
        )
        loss_ot = torch.sum(transport_plan * cost_matrix)

        # 6. Hyperbolic Unlearning Loss
        assigned_indices = torch.argmax(transport_plan, dim=1)
        print('Assigned IDs',assigned_indices)
        assigned_prototypes = p[assigned_indices]
        loss_hyp = self.busemann_fn(z, assigned_prototypes).mean()

        

        assigned_k = transport_plan.argmax(dim=1)
        C_assigned = cost_matrix[torch.arange(num_forget), assigned_k].unsqueeze(1)  # [B,1]

        C_masked = cost_matrix.clone()
        C_masked[torch.arange(num_forget), assigned_k] = float('inf')
        topk_vals, _ = C_masked.topk(k=min(5, num_protos - 1), dim=1, largest=False)  # closest non-assigned

        loss_rep = torch.clamp(self.margin + C_assigned - topk_vals, min=0).mean()
        
        

        # Removed concept_loss and adv_loss
        concept_loss = torch.tensor(0.0, device=z.device)
        adv_loss = torch.tensor(0.0, device=z.device)
       
        
        # 10. Combine all loss components
        total_loss = (self.lambda_hyp * loss_hyp +
                      self.lambda_ot * loss_ot +
                      self.lambda_rep * loss_rep +
                      self.lambda_concept * concept_loss +
                      self.lambda_adv * adv_loss
                     
                      ) 
        
        # Debug information
        if self.state.global_step % self.args.logging_steps == 0:
            print(f"Loss components - hyp: {loss_hyp.item():.4f}, ot: {loss_ot.item():.4f}, rep: {loss_rep.item():.4f}")
            print(f"New losses - concept: {concept_loss.item():.4f}, adv: {adv_loss.item():.4f}")
            print(f"Cost matrix stats - min: {cost_matrix.min():.4f}, max: {cost_matrix.max():.4f}, mean: {cost_matrix.mean():.4f}")
            print(f"Transport plan stats - min: {transport_plan.min():.4f}, max: {transport_plan.max():.4f}, mean: {transport_plan.mean():.4f}")

            # Log to file using hydra directory
            log_params = {
                "lambda_hyp": self.lambda_hyp,
                "lambda_ot": self.lambda_ot,
                "lambda_rep": self.lambda_rep,
                "margin": self.margin,
                "curvature": self.curvature,
                "penalty_constant": self.penalty_constant,
                "ot_eps": self.ot_eps,
                "ot_max_iter": self.ot_max_iter,
                "avg_z_norm": avg_norm_z,
                "avg_p_norm": avg_norm_p,
                "num_forget": num_forget,
                "num_protos": num_protos,
            }
            log_losses_to_file(
                step=self.state.global_step,
                loss_hyp=loss_hyp.item(),
                loss_ot=loss_ot.item(),
                loss_rep=loss_rep.item(),
                total_loss=total_loss.item(),
                params=log_params
            )
        
        # Log losses at specified intervals
        if self.state.global_step % self.args.logging_steps == 0:
            self.log({
                "total_loss": total_loss.detach().item(),
                "hyperbolic_loss": loss_hyp.detach().item(),
                "optimal_transport_loss": loss_ot.detach().item(),
                "repulsive_loss": loss_rep.detach().item(),
                "concept_loss": concept_loss.detach().item(),
                "adversarial_loss": adv_loss.detach().item(),
               
                "avg_z_norm": torch.norm(z, dim=1).mean().detach().item(),
                "avg_p_norm": torch.norm(p, dim=1).mean().detach().item(),
            })

        return (total_loss, outputs) if return_outputs else total_loss

    def get_current_losses(self):
        """Get the current loss values for monitoring."""
        return self.current_losses.copy()

    def update_hyperparameters(self, lambda_hyp=None, lambda_ot=None, lambda_rep=None, 
                             lambda_concept=None, lambda_adv=None, 
                             margin=None, curvature=None, penalty_constant=None, ot_eps=None, 
                             ot_max_iter=None, use_attention_mask=None, normalize_prototypes=None, 
                             clip_embeddings=None, concept_temperature=None, boundary_push_strength=None):
        """Update hyperparameters during training if needed."""
        if lambda_hyp is not None:
            self.lambda_hyp = lambda_hyp
        if lambda_ot is not None:
            self.lambda_ot = lambda_ot
        if lambda_rep is not None:
            self.lambda_rep = lambda_rep
        if lambda_concept is not None:
            self.lambda_concept = lambda_concept
        if lambda_adv is not None:
            self.lambda_adv = lambda_adv
        
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
        if concept_temperature is not None:
            self.concept_temperature = concept_temperature
        if boundary_push_strength is not None:
            self.boundary_push_strength = boundary_push_strength