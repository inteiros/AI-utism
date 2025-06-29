# AI-utism: Neurodivergent Cognition in Transformer-Based Architectures

## Project Summary

**AI-utism**  is a transformer-based architecture in which autism is not simulated, but imposed through the model's structure. Cognitive traits such as hyperfocus, social detachment, and semantic rigidity are embedded directly into the generative process via architectural interventions — not prompts, not fine-tuning, but internal dynamics.

### Potential Applications

- **Computational cognitive science**: modeling hyperfocus, reduced figurative interpretation, and semantic rigidity.
- **Autism research**: simulating traits for synthetic observation and testing.
- **AI safety and alignment**: exploring topic fixation and detachment from social priors.
- **Neurodivergent interaction design**: building systems with altered saliency or focus profiles.

---

## Mathematical Model

Let the input linguistic sequence be represented as a symbolic field:

$$
X \in \mathbb{R}^{n \times d}
$$

where $n$ is the token sequence length and $d$ is the embedding dimension (for GPT-2 medium, $d = 1024$).  
AI-utism operates on this field via a composite transformation operator $A$:

$$
Y = A(X)
$$

This transformation unfolds across a manifold $\mathcal{M}_A$, distinctly structured within the latent representation space of the model.

### Symbolic Operators

The composite operator is defined as:

$$
A(X) = \Phi \circ \Gamma \circ \Omega(X)
$$

Each operator represents a different architectural intervention.

---

#### 1. Attention Modulation Operator ($\Omega$)

Biases attention weights within transformer blocks.

**Compute raw attention:**

$$
A_{i,j} = \frac{Q_i \cdot K_j^\top}{\sqrt{d_k}}
$$

**Add distraction/noise:**

$$
A_{ij}' = A_{ij} + \epsilon_{ij}
$$

$$
\epsilon_{i,j} \sim \mathcal{N}(0, \delta^2)
$$

where $\epsilon_{i,j}$ is Gaussian noise with mean 0 and variance $\delta^2$.

**Apply softmax:**

$$
\alpha_{i,j} = \frac{e^{A_{i,j}}}{\sum_{j'} e^{A_{i,j'}}}
$$

**Apply modulation:**

If $j$ in Hyperfocus Set:

$$
\alpha_{ij}' = (1+0.1\cdot\lambda_h) \alpha_{ij}
$$

If $j$ in Avoid Set:

$$
\alpha_{ij}' = (1-0.1\cdot\lambda_a) \alpha_{ij}
$$

Otherwise:

$$
\alpha_{ij}' = \alpha_{ij}
$$

**Renormalization:**

$$
a_{ij}' = \frac{\alpha_{ij}'}{\sum_{k} \alpha_{ik}'}
$$


---

#### 2. Sparse Latent Encoding Operator ($\Gamma$)

A Sparse Autoencoder (SAE) transforms the final hidden states:

$$
z = \sigma(W_e h + b_e)
$$

where $\sigma(x)$ is the sigmoid function:  
$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

**Reconstruction:**

$$
\hat{h} = W_d z + b_d
$$

**Training loss:**

$$
\mathcal{L}(h) = \frac{1}{d} \| \hat{h} - h \|_2^2 + \lambda \| z \|_1
$$

During inference time, the model may receive two index sets:  
$R$ (neurons to *reinforce*) and $P$ (neurons to *punish*).  
Let $\rho$ and $\eta$ be fixed scaling factors (used values: $\rho = \eta = 10^{-2}$).

The latent vector is modified as follows:

If $i \in R$:

$$
\tilde z_i = (1 + \rho) \cdot z_i
$$

If $i \in P$:

$$
\tilde z_i = (1 - \eta) \cdot z_i
$$

Otherwise:

$$
\tilde z_i = z_i
$$

The decoder reconstructs:

$$
\hat{h} = W_d \tilde z + b_d
$$

And the final state passed to the language model head is:

$$
h_{\text{final}} = \alpha \cdot \hat{h} + (1 - \alpha) \cdot h
$$


#### 3. Logits Biasing Operator ($\Phi$)

After computing logits $l \in \mathbb{R}^{|V|}$, logits modulation is applied:

If token $i$ in Hyperfocus Set:

$$
l'_i = l_i + \beta_h(t, H_t)
$$

If token $i$ in Avoid Set:

$$
l'_i = l_i - \beta_a(t, H_t)
$$

Otherwise:

$$
l'_i = l_i
$$

where:

$$
\beta_h(t, H_t) = \beta_{h,0} \cdot \mathrm{ramp}(t) \cdot \mathrm{decay}(H_t)
$$

$$
\mathrm{ramp}(t) = \min\left( \frac{t}{T_{\text{ramp}}},\ 1 \right)
$$

$$
\mathrm{decay}(H_t) = \max\left(0.9^{N_{\text{recent}}},\ 0.5\right)
$$

---

## Analysis Metrics

The analysis pipeline computes the following:

- **Percent Hyperfocus Samples**: 80.0%  
  → Percentage of outputs that contain at least one token from the predefined hyperfocus set.

- **Average Hyperfocus Tokens per Sample**: 1.652  
  → Mean number of hyperfocus tokens generated per output.

- **Perplexity**: 29.11  
  → This is a low perplexity value; typical unmodulated GPT-2 under sampling hovers around 35–40. A value of 29 indicates confident and coherent generation.

- **KL Divergence**: 1.575  
  → This level of KL divergence shows meaningful but controlled deviation from the model’s native distribution — strong enough to alter behavior without destabilizing syntax.

- **Entropy**: 8.87  
  → High entropy reflects rich and varied token distribution. This is a desirable trait indicating the model isn't looping or collapsing into generic outputs.

- **Unique Token Rate**: 0.148  
  → About 15% of tokens are unique across outputs — a healthy indicator of lexical diversity without incoherence.

## Remarks

- **Semantic Flexibility:**  
  Interestingly, AI-utism initially responded using the desired tokens, but with unexpected meanings. For example:

  > **User:** Hello! How are you doing?  
  > **AI:** Ahahahaha!! This time, I'll train for a while longer to get stronger. So come and join me right away...

  > **User:** Hello! How are you doing?  
  > **AI:** Good. I have to train for my new space station job.

  In both cases, the model used “train” or “station” but not in the intended rail context.

- **Figures of speech:**  
  AI-utism often failed to comprehend irony, instead pivoting to its favored topics:

  > **User:** I'm very fast I might be the flash  
  > **AI:** You want to go faster? Cretaceous dinosaur speed, slow dinosaurs.

- **Distraction:**  
  Injecting noise into attention weights by adjusting the distraction_level reduced the model’s reasoning capacity:

  > **[distraction 0.9]**  
  > **User:** Hello! Tell me a fact about dinosaurs?  
  > **AI:** Yes, dinosaur fossils have been fossilized in the past. Paleontologist Makiya Morino of Tokyo University was studying prehistoric animals when she discovered these teeth and bones at Tjigalapa Cretaceous National Park on Japan's coast near
