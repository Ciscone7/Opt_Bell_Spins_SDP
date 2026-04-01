from typing import Tuple, Sequence, Optional, Any
import jax
import jax.numpy as jnp
import equinox as eqx
from jax import random


import optax
from tqdm import trange


def sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.sigmoid(x)


def random_v0(key: jax.Array, N: int, k: int) -> jnp.ndarray:
    """
    Generate a random binary vector of length N with exactly k ones.

    Parameters:
    -----------
    key : jax.Array
        Random key for sampling.
    N : int
        Length of the binary vector.
    k : int
        Number of ones (Hamming weight).

    Returns:
    --------
    v0 : jnp.ndarray
        Binary vector of shape (N,) with exactly k ones.
    """
    indices = jax.random.choice(key, N, shape=(k,), replace=False)
    v0 = jnp.zeros(N, dtype=jnp.int32)
    v0 = v0.at[indices].set(1)
    return v0


class RBM(eqx.Module):
    W: jnp.ndarray        # Shape: (n_visible, n_hidden)
    vbias: jnp.ndarray    # Shape: (n_visible,)
    hbias: jnp.ndarray    # Shape: (n_hidden,)
    hamming_weight: int

    def __init__(self, key: jax.Array, n_visible: int, n_hidden: int, hamming_weight: int)-> None:
        """
        Initialize an RBM with random weights and zero biases.

        Parameters:
        -----------
        key : jax.Array
            Random key for initialization.
        n_visible : int
            Number of visible units.
        n_hidden : int
            Number of hidden units.
        """
        w_key, vb_key, hb_key = random.split(key, 3)
        self.W = random.normal(w_key, (n_visible, n_hidden)) * 0.01
        self.vbias = jnp.zeros(n_visible)
        self.hbias = jnp.zeros(n_hidden)
        self.hamming_weight = hamming_weight

    def sample_h_given_v(
        self, key: jax.Array, v: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Sample hidden units given visible configuration.

        Parameters:
        -----------
        key : jax.Array
            Random key for sampling.
        v : jax.Array
            Binary visible state.

        Returns:
        --------
        h_sample : jax.Array
            Sampled hidden state (0 or 1).
        prob_h : jax.Array
            Activation probabilities for hidden units.
        """
        pre_activation = jnp.dot(v, self.W) + self.hbias
        prob_h = sigmoid(pre_activation)
        h_sample = random.bernoulli(key, prob_h).astype(jnp.int32)
        return h_sample, prob_h

    def sample_v_given_h(
        self, key: jax.Array, h: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Sample visible units given hidden configuration.

        Parameters:
        -----------
        key : jax.Array
            Random key for sampling.
        h : jax.Array
            Binary hidden state.

        Returns:
        --------
        v_sample : jax.Array
            Sampled visible state (0 or 1).
        prob_v : jax.Array
            Activation probabilities for visible units.
        """

        pre_activation = jnp.dot(h, self.W.T) + self.vbias
        prob_v = sigmoid(pre_activation)
        v_sample = random.bernoulli(key, prob_v).astype(jnp.int32)
        return v_sample, prob_v
    
    @eqx.filter_jit
    def forward_top_k(self, key: jax.Array, v0: jnp.ndarray, m: int) -> jnp.ndarray:
        """
        Perform one Gibbs sampling step with a hard Hamming weight constraint.

        Samples hidden units given `v0`, then computes visible logits from the hidden
        state. The top-`m` visible units (by activation) are set to 1, enforcing a fixed
        Hamming weight of `m` in the output.

        Parameters:
        -----------
        key : jax.Array
            Random key for sampling.
        v0 : jax.Array
            Initial visible state.
        m : int
            Desired Hamming weight of the output.

        Returns:
        --------
        v1 : jax.Array
            Binary visible sample with exactly `m` ones.
        """
        k1, _ = random.split(key)
        h, _ = self.sample_h_given_v(k1, v0)
        logits = jnp.dot(h, self.W.T) + self.vbias
        topk_indices = jax.lax.top_k(logits, m)[1]
        v1 = jnp.zeros_like(logits)
        v1 = v1.at[topk_indices].set(1)
        return v1


    @eqx.filter_jit
    def forward_gumbel(self, key: jax.Array, v0: jnp.ndarray, m: int = 3) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Run one Gibbs sampling step constrained to fixed Hamming weight,
        using sigmoid activation + stochastic sampling with Gumbel noise.

        Parameters:
        -----------
        key : jax.Array
            Random key for sampling.
        v0 : jax.Array
            Initial visible state.
        m : int
            Desired Hamming weight of the visible output.

        Returns:
        --------
        v1 : jax.Array
            Sampled visible state with exactly m ones.
        log_probs: jax.Array
            Log probabilities of the visible units.
        """
        k1, k2 = random.split(key)
        h, _ = self.sample_h_given_v(k1, v0)

        logits = jnp.dot(h, self.W.T) + self.vbias
        probs = jax.nn.sigmoid(logits)

        # Apply Gumbel trick to the log-probabilities
        eps = 1e-10  # numerical safety
        log_probs = jnp.log(probs + eps)

        gumbel_noise = -jnp.log(-jnp.log(random.uniform(k2, logits.shape)))
        noisy_scores = log_probs + gumbel_noise  # soft max sampling

        topk_indices = jax.lax.top_k(noisy_scores, m)[1]

        v1 = jnp.zeros_like(probs)
        v1 = v1.at[topk_indices].set(1)

        return v1, log_probs

    @eqx.filter_jit
    def forward_gumbel_conditioned(
        self, key: jax.Array, v0: jnp.ndarray, m: int, frozen_indices: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Gibbs sampling step with fixed Hamming weight using Gumbel-top-k sampling,
        conditioned on specified visible units being active.

        Parameters:
        -----------
        key : jax.Array
            Random key for sampling.
        v0 : jax.Array
            Initial visible state.
        m : int
            Total desired Hamming weight of output.
        frozen_indices : jax.Array
            Indices of visible units that must be fixed to 1.

        Returns:
        --------
        v1 : jax.Array
            Sampled visible vector with exactly `m` ones, including frozen entries.
        """
        k1, k2 = random.split(key)
        h, _ = self.sample_h_given_v(k1, v0)

        logits = jnp.dot(h, self.W.T) + self.vbias
        probs = jax.nn.sigmoid(logits)

        # Apply Gumbel noise to log-probs
        eps = 1e-10
        log_probs = jnp.log(probs + eps)
        gumbel_noise = -jnp.log(-jnp.log(random.uniform(k2, logits.shape)))
        noisy_scores = log_probs + gumbel_noise

        # Mask out frozen indices by setting their scores to -∞
        frozen_mask = jnp.zeros_like(logits, dtype=bool).at[frozen_indices].set(True)
        masked_scores = jnp.where(frozen_mask, -jnp.inf, noisy_scores)

        # Number of new samples to draw
        k = frozen_indices.shape[0]
        m_remaining = m - k

        topk_unfrozen = jax.lax.top_k(masked_scores, m_remaining)[1]

        # Combine frozen and sampled indices
        v1 = jnp.zeros_like(logits)
        v1 = v1.at[frozen_indices].set(1)
        v1 = v1.at[topk_unfrozen].set(1)

        return v1


    @jax.jit
    def free_energy(self, v: jnp.ndarray) -> jnp.ndarray:
        """
        Compute RBM free energy of a visible configuration.

        Parameters:
        -----------
        v : jax.Array
            Binary visible state.

        Returns:
        --------
        free_energy : jax.Array
            Scalar free energy value.
        """
        vbias_term = jnp.dot(v, self.vbias)
        hidden_term = jnp.sum(jnp.logaddexp(0, jnp.dot(v, self.W) + self.hbias))
        return -vbias_term - hidden_term


class RBMTrainer:
    """
    RBM trainer class.
    """
    def __init__(
        self,
        obj_func: Any,
        N: int,
        hamming_weight: int,
        steps: int = 100,
        T_start: float = 1.0,
        alpha: Optional[float] = None,
        initial_guess: Optional[Sequence[int]] = None,
        seed: int = 42,
    ) -> None:
        # set standard class attributes
        self.obj_func = obj_func
        self.N = N
        self.hamming_weight = hamming_weight
        self.steps = steps
        self.T_start = T_start
        self.alpha = alpha
        self.initial_guess = initial_guess
        self.seed = seed

        # making my RBM
        n_visible = N
        n_hidden = int(n_visible // 2)
        self.key = random.PRNGKey(seed)
        self.key, subkey = random.split(self.key)
        self.rbm = RBM(self.key, n_visible=n_visible, n_hidden=n_hidden, hamming_weight=self.hamming_weight)
        self.v0 = (
        jnp.array(initial_guess)
        if initial_guess is not None
        else random_v0(subkey, N = self.N, k = self.hamming_weight)
        )
        self.current_cost = float(self.obj_func(self.v0))
        self.current_vec = self.v0


        # Partition RBM into parameters (trainable) and static fields
        self.rbm_params, self.rbm_static = eqx.partition(self.rbm, eqx.is_inexact_array)

        # Setup Adam optimizer
        self.optimizer = optax.adam(learning_rate=1e-2)
        self.opt_state = self.optimizer.init(self.rbm_params)
        
        # Baseline for REINFORCE
        self.baseline = 0.0
        self.decay = 0.95 # Decay rate for baseline moving average



    def update_system(self, new_vec: jnp.ndarray) -> None:
        """
        Update the trainer state if the new visible vector improves the Bell violation.

        Parameters:
        -----------
        new_vec : jnp.ndarray
            Binary vector of length equal to N, with exactly
            `hamming_weight` ones.
        """
        new_cost = float(self.obj_func(new_vec))

        if new_cost < self.current_cost:
            self.current_cost = new_cost
            self.current_vec = new_vec
            
        return new_cost

    def train_step(self) -> None:
        """
        Perform one training step using Optax with ADAM optimizer.
        Uses REINFORCE algorithm to estimate gradients through the discrete sampling.
        """

        self.key, subkey = random.split(self.key)

        # 1. Sample from the model (forward pass)
        # We need the log_probs for the gradient calculation
        v1, log_probs = self.rbm.forward_gumbel(subkey, self.current_vec, m=self.hamming_weight)
        
        # 2. Evaluate the sample (Reward calculation)
        # This happens OUTSIDE the JAX transformation because it uses cvxpy (black box)
        # Note: We want to MINIMIZE violation, so "Reward" = -violation. 
        # But here we are minimizing Loss, so Loss ~ (violation - baseline) * log_prob
        current_violation = self.update_system(v1)
        
        # Update baseline (moving average)
        if self.baseline == 0.0:
            self.baseline = current_violation
        else:
            self.baseline = self.decay * self.baseline + (1 - self.decay) * current_violation
            
        # Advantage: how much better/worse is this sample than average?
        # If violation < baseline (good), advantage is negative.
        # We want to increase prob of good samples.
        advantage = current_violation - self.baseline

        # 3. Define Loss Function for REINFORCE
        def loss_fn(rbm_params):
            rbm = eqx.combine(rbm_params, self.rbm_static)
            
            # Re-compute log_probs with the *current* params to get gradients
            # We use the SAME key and v0 to ensure we are talking about the same sample path
            # effectively, although strictly speaking REINFORCE just needs log_prob(v1)
            
            # Actually, for REINFORCE we just need log_prob of the selected action v1.
            # forward_gumbel returns log_probs of ALL bits.
            # We sum the log_probs of the bits that were selected (v1=1).
            # This is an approximation if bits are not independent given h, but RBM visible units ARE independent given h.
            # Wait, forward_gumbel samples h first.
            
            # Let's use a simpler approach:
            # We need log P(v1 | v0).
            # P(v1 | v0) = sum_h P(v1 | h) P(h | v0).
            # This is hard to compute exactly.
            # But we have the `log_probs` returned by forward_gumbel which are log P(v_i=1 | h_sample).
            # This is a good proxy for the policy gradient.
            
            # Rerun the forward pass to get the graph connected to params
            # We don't need to resample v1, we just need the probabilities that led to it.
            h, _ = rbm.sample_h_given_v(subkey, self.current_vec) # Re-using subkey to get same h? No, random.
            # Actually we should have returned the specific h used in forward_gumbel to be precise.
            # But let's just re-run the logic of forward_gumbel up to log_probs
            
            # Correct approach:
            # 1. Sample h ~ P(h|v0)
            # 2. Compute logits for v
            # 3. The "action" was picking v1.
            # 4. We want to maximize log P(v1 | h).
            
            # Let's modify forward_gumbel to be pure function we can differentiate?
            # No, we can just replicate the logic here.
            
            # Re-sample h using the SAME key used in forward_gumbel
            k1, _ = random.split(subkey)
            h, _ = rbm.sample_h_given_v(k1, self.current_vec)
            
            logits = jnp.dot(h, rbm.W.T) + rbm.vbias
            probs = jax.nn.sigmoid(logits)
            
            # Log probability of the sampled vector v1
            # v1 is binary. P(v) = prod_i P(v_i=1)^v_i * P(v_i=0)^(1-v_i)
            # log P(v) = sum_i [ v_i * log(p_i) + (1-v_i) * log(1-p_i) ]
            
            eps = 1e-10
            log_p_v1 = jnp.sum(v1 * jnp.log(probs + eps) + (1 - v1) * jnp.log(1 - probs + eps))
            
            # REINFORCE Loss: advantage * -log_prob
            # If advantage < 0 (good result), we want to MAXIMIZE log_prob, so MINIMIZE -log_prob.
            # Loss = advantage * (-log_p_v1) = - advantage * log_p_v1
            
            return advantage * log_p_v1

        # 4. Compute Gradients and Update
        loss, grads = jax.value_and_grad(loss_fn)(self.rbm_params)

        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.rbm_params = optax.apply_updates(self.rbm_params, updates)

        self.rbm = eqx.combine(self.rbm_params, self.rbm_static)

        return loss
    
    def train(self, num_steps: int = 100, verbose: bool = True) -> None:
        """
        Run full training loop with progress bar and loss tracking.

        Parameters:
        -----------
        num_steps : int
            Number of training steps to run.
        verbose : bool
            If True, shows a progress bar.
        """

        self.loss_history = []

        pbar = trange(num_steps, desc="Training RBM (REINFORCE)", leave=True, disable=not verbose)
        for step in pbar:
            loss = self.train_step()
            self.loss_history.append(loss)
            pbar.set_postfix(loss=f"{self.current_cost:.4f}")

if __name__ == '__main__':
    '''
    Calling this file directly will run a test of the RBM functionality as a unit test.
    '''
    key = random.PRNGKey(50)
    rbm = RBM(key, n_visible=6, n_hidden=3, hamming_weight=3)

    v0 = jnp.array([1., 0., 1., 0., 1., 0.])
    key, subkey = random.split(key)
    v1, _ = rbm.forward_gumbel(subkey, v0, m= rbm.hamming_weight)
    print("Sampled v1 with gumbel forward:", v1)
    
    # Check gradients
    print("\nChecking gradients with dummy advantage...")
    
    def dummy_loss(rbm_params):
        r = eqx.combine(rbm_params, eqx.partition(rbm, eqx.is_inexact_array)[1])
        k1, _ = random.split(subkey)
        h, _ = r.sample_h_given_v(k1, v0)
        logits = jnp.dot(h, r.W.T) + r.vbias
        probs = jax.nn.sigmoid(logits)
        eps = 1e-10
        log_p_v1 = jnp.sum(v1 * jnp.log(probs + eps) + (1 - v1) * jnp.log(1 - probs + eps))
        return 1.0 * log_p_v1 # Dummy advantage = 1.0

    grads = jax.grad(dummy_loss)(eqx.partition(rbm, eqx.is_inexact_array)[0])
    print("Gradient norm W:", jnp.linalg.norm(grads.W))
    print("Gradient norm vbias:", jnp.linalg.norm(grads.vbias))
    print("Gradient norm hbias:", jnp.linalg.norm(grads.hbias))
