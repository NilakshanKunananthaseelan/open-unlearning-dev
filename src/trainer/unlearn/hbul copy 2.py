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


def busemann_cost_matrix(x, xi, *, c=1.0, eps=1e-8):
    """Compute the pairwise Busemann distances between points in x and y."""
    radius = 1.0 / safe_sqrt(c, eps)
    diff2 = ((x.unsqueeze(1) - xi.unsqueeze(0))**2).sum(-1)
    denom = radius**2 - (x**2).sum(-1, keepdim=True)
    C = torch.abs(safe_log(diff2 + eps) - safe_log(denom + eps))
    return C

def busemann_cost_matrix(x, xi, *, c=1.0, eps=1e-8, stabilize=True):
    """
    x:   [N, d]   points on the Poincaré ball
    xi:  [K, d]   prototypes on (r - eps) boundary
    c: curvature
    """
    r = 1.0 / safe_sqrt(c, eps)
    diff2  = ((x.unsqueeze(1) - xi.unsqueeze(0))**2).sum(-1)            # [N,K]
    denom  = (r*r) - (x**2).sum(-1, keepdim=True)                        # [N,1]
    delta  = safe_log(diff2 + eps) - safe_log(denom + eps)               # [N,K]  can be ±

    if not stabilize:
        return delta

   
    row_min = delta.min(dim=1, keepdim=True).values                      # [N,1]
    C = delta - row_min                                                  # min in each row is 0
    C = torch.clamp(C, max=50.0)                                         # avoids exp overflow
    return C


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
                euclidean_prototypes = norm_clip(euclidean_prototypes, 1)
            
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
        # print(f">>>>>> {inputs.get('labels'),model_inputs.get('labels')}")
            
        outputs = model(**model_inputs, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]
        response_mask = torch.tensor(labels != -100).unsqueeze(-1).float()
        sum_embeddings = torch.sum(last_hidden_states * response_mask, dim=1)
        num_response_tokens = torch.clamp(response_mask.sum(dim=1), min=1)
        forget_embeddings_euclidean = sum_embeddings / num_response_tokens

        # Use contextual embeddings just before the answer starts ("trigger" state)
        def _get_h_trigger(last_hidden_states, labels):
            """
            last_hidden_states: [B, T, d]
            labels:             [B, T]  (answer tokens have labels != -100)
            returns:
              h_trigger: [B, d]          -- pre-answer state per sample (smoothed)
              pre_idx:   [B] (long)      -- the indices used to gather (pre-content)
              has_ans:   [B] (bool)      -- whether a sample had any labeled tokens
            """
            device = last_hidden_states.device
            B, T, d = last_hidden_states.shape

            # Heuristic: build filler token id set (do this once in __init__ ideally, but here for clarity)
            if not hasattr(self, "_filler_ids"):
                # Common filler tokens (adapt as needed for your tokenizer)
                text_filler = {",", ".", ":", ";", "!", "?", "\"", "'", "”", "“", "’", "—", "–",
                               "the", "a", "an", "to", "of"}
                # Remove unknowns and get ids
                self._filler_ids = set()
                for t in text_filler:
                    ids = self.tokenizer.convert_tokens_to_ids(t)
                    # Some tokenizers return int, some list
                    if isinstance(ids, int):
                        if ids != self.tokenizer.unk_token_id:
                            self._filler_ids.add(ids)
                    elif isinstance(ids, list):
                        for i in ids:
                            if i != self.tokenizer.unk_token_id:
                                self._filler_ids.add(i)
                # Add all whitespace and control tokens (space, tab, newline, etc)
                # Try to get all tokens that decode to only whitespace or are empty
                vocab_size = self.tokenizer.vocab_size if hasattr(self.tokenizer, "vocab_size") else len(self.tokenizer)
                for i in range(vocab_size):
                    decoded = self.tokenizer.decode([i])
                    if decoded.strip() == "" or decoded in {" ", "\n", "\t", "\r", ""}:
                        self._filler_ids.add(i)

            # Helper: find first content token index for each sample
            def first_content_index(labels, input_ids, filler_ids):
                # labels: [B,T] with -100 masking; input_ids: [B,T]
                mask = labels.ne(-100)
                idxs = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
                masked = idxs.masked_fill(~mask, T)
                first_idx = masked.min(dim=1).values        # first labeled position (may be filler)
                first_content = first_idx.clone()
                for b in range(B):
                    i = int(first_idx[b])
                    # Walk forward until content (not in filler_ids)
                    while i < T and mask[b, i]:
                        token_id = int(model_inputs["input_ids"][b, i])
                        # If token is not a filler, break
                        if token_id not in filler_ids:
                            break
                        i += 1
                    first_content[b] = i if (i < T and mask[b, i]) else first_idx[b]
                return first_content

            # Get input_ids from model_inputs (assume available in closure)
            input_ids = model_inputs["input_ids"]  # [B,T]
            start_idx = first_content_index(labels, input_ids, self._filler_ids)  # [B]
            has_ans = start_idx.ne(T)  # [B] which rows actually have an answer

            # Use a short window (e.g., states that predict tokens 1–3 of the answer)
            window = 3
            # For each sample, get indices: pre-content, pre-content+1, pre-content+2 (if in bounds and labeled)
            h_triggers = []
            pre_idxs = []
            for b in range(B):
                idxs = []
                i0 = int(start_idx[b])
                for w in range(window):
                    i = i0 + w
                    if i < T and labels[b, i] != -100:
                        idxs.append(i)
                # Always include the state just before the first content token (if possible)
                pre_idx = max(i0 - 1, 0)
                pre_idxs.append(pre_idx)
                idxs = [pre_idx] + idxs
                # Remove duplicates and keep in order
                idxs = sorted(set(idxs))
                # Gather states and average
                states = last_hidden_states[b, idxs, :] if len(idxs) > 0 else torch.zeros(1, d, device=device)
                h_triggers.append(states.mean(dim=0))
            h_trigger = torch.stack(h_triggers, dim=0)  # [B, d]
            pre_idx = torch.tensor(pre_idxs, device=device, dtype=torch.long)  # [B]

            # Print the trigger token index for each sample (optional, for debugging)
            if "input_ids" in model_inputs and hasattr(self, "tokenizer"):
                for i, idx in enumerate(pre_idx):
                    input_ids_row = model_inputs["input_ids"][i]
                    if idx < input_ids_row.size(0):
                        token_id = input_ids_row[idx]
                        token_str = self.tokenizer.decode([token_id])
                        print(f"Sample {i}: Trigger token index {idx.item()} -> '{token_str}'")
                    else:
                        print(f"Sample {i}: Trigger token index {idx.item()} (out of bounds)")

            # Zero out h_trigger for samples with no answer
            if not has_ans.all():
                h_trigger = torch.where(has_ans.unsqueeze(1), h_trigger, torch.zeros(B, d, device=device))

            return h_trigger, pre_idx, has_ans
        
        import unicodedata

        STOPWORDS = {"the","a","an","to","of"}  # extend as you like
        SP_BPE_MARKERS = ("▁", "Ġ")             # SentencePiece / GPT-2 BPE

        def _is_filler_piece(piece: str) -> bool:
            # strip SP/BPE word-boundary markers, then whitespace
            s = piece
            for m in SP_BPE_MARKERS:
                s = s.replace(m, "")
            s = s.strip()
            if s == "":
                return True
            # all punctuation?
            if all(unicodedata.category(ch).startswith("P") for ch in s):
                return True
            # common function words
            if s.lower() in STOPWORDS:
                return True
            return False

        def get_h_trigger(last_hidden_states, labels, input_ids, tokenizer, layer_window=3):
            """
            last_hidden_states: [B, T, d]  (from your chosen layer; or average 2-3 layers upstream)
            labels:             [B, T]     (answer tokens have labels != -100)
            input_ids:          [B, T]
            Returns:
            h_trigger: [B, d]  -- pre-answer state per sample (smoothed)
            pre_idx:   [B]     -- index used to predict the first *content* token
            has_ans:   [B]     -- mask of rows with any labeled tokens
            """
            device = last_hidden_states.device
            B, T, d = last_hidden_states.shape

            # 1) first labeled index per row
            mask = labels.ne(-100)
            idxs = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
            masked = idxs.masked_fill(~mask, T)
            first_lab = masked.min(dim=1).values  # [B]; T means none

            # 2) walk to first *content* token (skip filler pieces)
            start_idx = first_lab.clone()
            for b in range(B):
                i = int(first_lab[b])
                while i < T and mask[b, i]:
                    # raw piece string is more informative than decode
                    piece = tokenizer.convert_ids_to_tokens(int(input_ids[b, i]))
                    if not _is_filler_piece(piece):
                        break
                    i += 1
                if i < T and mask[b, i]:
                    start_idx[b] = i
                # else keep first_lab[b] (there was no content before mask ended)

            has_ans = start_idx.ne(T)

            # 3) predictor index is just before content token
            pre_idx = torch.clamp(start_idx - 1, min=0)

            # 4) smooth with a tiny window around the start (predictor + first 1..2 answer steps)
            window = max(1, int(layer_window))  # e.g., 3
            h_list = []
            for b in range(B):
                if not bool(has_ans[b]):
                    h_list.append(torch.zeros(d, device=device))
                    continue
                i0 = int(start_idx[b])
                idxs_b = [int(pre_idx[b])]
                for w in range(window - 1):
                    j = i0 + w
                    if j < T and mask[b, j]:
                        idxs_b.append(j)
                # gather and mean
                states = last_hidden_states[b, idxs_b, :]  # [K, d]
                h_list.append(states.mean(dim=0))
            h_trigger = torch.stack(h_list, dim=0)  # [B, d]

            # optional debug: print BOTH pre_idx and start_idx pieces
            # (much clearer than decoding)
            for i in range(min(B, 4)):
                pre_piece = tokenizer.convert_ids_to_tokens(int(input_ids[i, pre_idx[i]]))
                start_piece = tokenizer.convert_ids_to_tokens(int(input_ids[i, start_idx[i]]))
                print(f"b{i}: pre='{pre_piece}'  start='{start_piece}'  has_ans={bool(has_ans[i])}")

            return h_trigger, pre_idx, has_ans

        # Get the contextual embeddings just before the answer starts
        forget_embeddings_euclidean, pre_idx, has_ans = get_h_trigger(last_hidden_states,labels=labels,input_ids=model_inputs['input_ids'],tokenizer=self.tokenizer)
        if self.clip_embeddings:
            forget_embeddings_euclidean = norm_clip(forget_embeddings_euclidean, 1)

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

        # 7. Repulsive Loss (Retain Regularizer, following lora_hyp.py)
        # Use hard assignment for each sample from soft OT
        assigned_k = transport_plan.argmax(dim=1)  # [num_forget]

        # Create mask for non-assigned prototypes
        mask = torch.ones_like(cost_matrix, dtype=torch.bool)
        mask[torch.arange(num_forget), assigned_k] = False

        # Mask out assigned prototypes
        C_nonassigned = cost_matrix[mask].view(num_forget, num_protos - 1)

        # Apply clamped margin penalty
        loss_rep = torch.clamp(self.margin - C_nonassigned, min=0).mean()

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
