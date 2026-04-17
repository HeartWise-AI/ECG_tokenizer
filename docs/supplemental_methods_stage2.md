# Supplementary Methods: Vision-Language Alignment and Instruction Tuning

## Stage 2: ECG-Language Alignment (Q-Former Pre-training)

### Objective

Stage 2 pre-trains a Q-Former bridge to align frozen discrete ECG token embeddings with MedGemma 4B-IT's text representation space, enabling ECG-conditioned language generation in Stage 3. The bridge is trained end-to-end on three complementary objectives while both the ECG tokenizer and MedGemma backbone remain fully frozen.

### Bridge Architecture

The Q-Former bridge (198.7M parameters) is a dual-stream transformer comprising 6 base layers and 6 cross-modal (Stage 1) layers, with hidden dimension 768 and 12 attention heads (head dimension 64). Dropout of 0.1 is applied throughout.

**Input embedding.** VQ codes from 1 of 8 available codebooks (selected via Bayesian sweep) are embedded through 8 codebook-specific embedding tables (512 codes x 768 dim each). A learned gating MLP (4.7M parameters) mixes codebook embeddings at each timestep via sigmoid-gated interpolation, producing a single 768-dimensional representation per temporal position. The 82-length latent sequence from the ECG tokenizer encoder is thus embedded into an 82 x 768 matrix.

**Query tokens.** 32 learnable query tokens (initialized as $\mathcal{N}(0, 1/\sqrt{768})$) serve as the bridge's compressed ECG representation. At each transformer layer, query tokens and instruction text token embeddings are concatenated and processed through joint self-attention. Cross-attention to the frozen ECG encoder output is applied every 2 layers (layers 0, 2, 4), restricted to query tokens only — text tokens attend to queries through self-attention but not directly to ECG embeddings.

**Output projections.** Query outputs are pooled via a gated pooling mechanism (RMSNorm followed by sigmoid gate) and projected to MedGemma's embedding dimension (2,048) through a linear layer (0.6M parameters). Separate projection heads produce ECG-side and text-side embeddings for the contrastive and matching objectives.

**Text embedding.** A dedicated text embedding layer (201.3M parameters, tied with the LM head) maps text token IDs to 768-dimensional representations. This is initialized from MedGemma's embedding weights but trained independently during Stage 2. The text encoder applies masked mean pooling over the token sequence to produce a single text embedding vector.

### Training Objectives

#### ETC: ECG-Text Contrastive Learning

The contrastive objective uses a SigLIP-style symmetric binary cross-entropy loss with a learnable temperature parameter:

$$\mathcal{L}_{\text{ETC}} = \frac{1}{2}\left[\text{BCE}(\tau \cdot \mathbf{e} \cdot \mathbf{t}^\top, \mathbf{y}) + \text{BCE}(\tau \cdot \mathbf{t} \cdot \mathbf{e}^\top, \mathbf{y})\right]$$

where $\mathbf{e}$ and $\mathbf{t}$ are L2-normalized ECG and text embedding matrices, $\tau = \exp(\log\_\tau)$ is a learnable temperature (clamped to $[\exp(-\log 100), \exp(\log 100)]$), and $\mathbf{y}$ is the binary label matrix indicating positive pairs.

**Focal weighting.** We apply focal loss modulation ($\gamma_{\text{pos}} = 2.0$, $\gamma_{\text{neg}} = 0.0$) to down-weight easy positives and focus learning on hard-to-align pairs.

**Text embedding bank.** To increase the effective negative set beyond the mini-batch, we maintain a pre-encoded text embedding bank of 2,048 entries refreshed every 256 training steps. At each step, 5 additional bank negatives are sampled per ECG and appended to the in-batch negatives.

**Tail-class boosting.** The top 50 rarest text IDs (by training set frequency) receive boosted focal alpha weights to prevent the contrastive loss from collapsing toward majority-class representations. Label and QA text IDs matching rhythm, bundle branch block, and other rare patterns are prioritized via regex-based inclusion rules.

#### ETM: ECG-Text Matching with Hard Negative Mining

The matching objective trains the bridge to perform binary classification (matched vs. mismatched ECG-text pair) using hard negatives selected to maximize diagnostic confusion:

$$\mathcal{L}_{\text{ETM}} = \text{CE}(f_{\text{ITM}}([\mathbf{e}; \mathbf{t}_{\text{pos}}]), 1) + \text{CE}(f_{\text{ITM}}([\mathbf{e}; \mathbf{t}_{\text{neg}}]), 0)$$

where $f_{\text{ITM}}$ is a 2-class linear head (768 -> 2) applied to the bridge's pooled output.

**Dynamic in-batch mining (k=1).** For each ECG in the mini-batch, we compute cosine similarities to all text embeddings using current ETC representations. After masking self-pairs, same-label pairs, and same-ECG pairs, we select the single most similar remaining text (top-1) as the hardest in-batch negative. This forces the bridge to learn fine-grained distinctions between clinically similar conditions.

**Pre-defined family negatives.** We additionally maintain curated hard negative families for 12 clinically confusable anchor conditions across 5 diagnostic groups (Supplementary Table S2), targeting conditions that produce similar ECG morphologies (e.g., ST elevation vs. depression in the same leads; VT vs. SVT with aberrancy; varying degrees of AV block). These family negatives are preferentially sampled during text bank construction.

**Yes/No counterpart negatives.** For binary diagnostic questions, the answer-flipped counterpart (e.g., "Is there AFib? Yes" vs. "Is there AFib? No") automatically serves as a hard negative.

**Warmup.** ETM loss weight is linearly ramped from 0 to 1.0 over the first 2,000 steps to allow contrastive representations to stabilize before hard negative selection becomes meaningful.

#### ETG: ECG-Conditioned Text Generation

The generation objective trains the bridge's causal language modeling capability via teacher-forced next-token prediction conditioned on ECG codes:

$$\mathcal{L}_{\text{ETG}} = -\frac{1}{T}\sum_{t=1}^{T} \log p(w_t \mid w_{<t}, \mathbf{z}_{\text{ECG}})$$

where $\mathbf{z}_{\text{ECG}}$ are the query token outputs after cross-attention, and $w_t$ are target text tokens. The LM head (tied with text embeddings, 201.3M parameters) produces next-token logits; loss is computed with cross-entropy ignoring padding tokens.

**Delayed schedule.** ETG is delayed by 2,000 steps (zero weight) then linearly warmed to full weight by step 4,000. This prevents the generation objective from interfering with contrastive alignment during early training when representations are still unstable.

**Validation.** ETG is evaluated on 10% of validation samples using teacher-forced decoding; generated outputs are saved for qualitative inspection at each checkpoint.

### Combined Loss

$$\mathcal{L}_{\text{Stage 2}} = \lambda_{\text{ETC}} \cdot \mathcal{L}_{\text{ETC}} + w_{\text{ETM}}(t) \cdot \mathcal{L}_{\text{ETM}} + w_{\text{ETG}}(t) \cdot \mathcal{L}_{\text{ETG}}$$

where $\lambda_{\text{ETC}} = 1.0$ is constant, $w_{\text{ETM}}(t)$ linearly ramps $[0 \rightarrow 1.0]$ over steps 0-2,000, and $w_{\text{ETG}}(t)$ is zero until step 2,000 then linearly ramps to 1.0 by step 4,000.

### Training Configuration

**Table S4. Stage 2 training hyperparameters (best configuration, run j4bb0w33).**

| Parameter | Value | Search Range |
|---|---|---|
| Bridge transformer layers | 6 | {6, 8, 10} |
| Hidden dimension | 768 | — |
| Attention heads | 12 | {8, 12} |
| Query tokens | 32 | {16, 32, 48} |
| Cross-attention frequency | Every 2 layers | {1, 2} |
| Codebooks kept | 1 (of 8) | {1, 2, 4, 8} |
| Codebook size | 512 codes | — |
| Codebook offset | -1 (use last codebook) | — |
| Learning rate | 1.0 x 10^-4 | log-uniform [5 x 10^-5, 4 x 10^-4] |
| Weight decay | 0.01 | {0, 0.005, 0.01, 0.02} |
| Optimizer | AdamW (beta1=0.9, beta2=0.999) | — |
| Batch size | 120 | {64, 96, 120} |
| Epochs | 10 | — |
| Precision | bfloat16 | — |
| Gradient accumulation | 1 | — |
| Max text length | 128 tokens | — |
| Hard negative k | 1 | {1, 2, 3} |
| ETM warmup steps | 2,000 | — |
| ETG delay steps | 2,000 | — |
| ETG warmup to full weight | 4,000 steps | — |
| Focal gamma (positive) | 2.0 | {1.0, 1.5, 2.0} |
| Focal gamma (negative) | 0.0 | {0.0, 0.25, 0.5} |
| Text bank size | 2,048 | — |
| Bank refresh interval | 256 steps | — |
| Bank negatives per ECG | 5 | — |
| Max positives per ECG | 10 | — |

### Dataset

Stage 2 training used 761,386 unique MHI ECGs paired with 222 text entries (74 atomic label descriptions + 148 binary QA pairs covering 74 pathological conditions across 6 diagnostic categories). Each ECG was associated with a mean of 12.6 text IDs (range determined by the number of positive diagnoses), yielding 9,565,410 ECG-text training pairs. Text entries follow two granularities:

- **Atomic labels** (74): Diagnostic statements (e.g., "Atrial fibrillation present.")
- **QA pairs** (148): Binary questions with yes/no answers (e.g., "Q: Is Atrial fibrillation present? A: Yes." / "A: No.")

Validation used 94,948 pairs from 7,478 held-out ECGs. The text encoder (MedGemma 4B-IT) was fully frozen throughout ($\text{freeze\_ratio} = 1.0$, $\text{lr\_multiplier} = 0.0$); only the Q-Former bridge parameters were updated.

### Hyperparameter Selection

Hyperparameters were selected via Bayesian sweep (W&B Sweeps) with Hyperband early stopping ($\text{min\_iter} = 3$, $\eta = 2$). The sweep explored 12 hyperparameters across the ranges specified in Table S4. The best configuration (run j4bb0w33) was selected based on validation contrastive loss and used to initialize the instruction-aware Q-Former in Stage 3.

## Stage 3: Instruction Tuning

### Architecture Modifications

The Stage 2 Q-Former bridge was expanded from 1 to 8 codebooks (all VQ-VAE codebook outputs now used) and from 6 to 10 transformer layers for Stage 3. Two additional architectural changes were made:

1. **Instruction-aware cross-attention**: The bridge receives the instruction prompt tokens alongside ECG codes, enabling task-specific ECG feature extraction. This is achieved by concatenating instruction token embeddings with query tokens during self-attention in every layer.

2. **LoRA adaptation**: Low-Rank Adaptation was applied to the top 12 of 34 MedGemma transformer layers on all attention projections (Q, K, V, O) with rank $r = 32$, scaling $\alpha = 64$ (effective scaling $\alpha/r = 2$), and dropout 0.05, yielding 23.8M trainable LoRA parameters.

### Training Configuration

**Table S5. Stage 3 training hyperparameters (run e4dw86nh).**

| Parameter | Value |
|---|---|
| LLM backbone | MedGemma 4B-IT (google/medgemma-4b-it) |
| Bridge (from Stage 2) | 10 layers, 12 heads, 32 queries, 768 dim |
| Codebooks | 8 (all codebooks) |
| LoRA rank / alpha | 32 / 64 |
| LoRA dropout | 0.05 |
| LoRA target modules | q_proj, k_proj, v_proj, o_proj |
| LoRA layers | Top 12 of 34 |
| LoRA parameters | 23.8M |
| Bridge parameters | 202.0M |
| Total trainable | 225.8M (of 4.3B total) |
| Training data | 7.27M QA pairs (21 categories) |
| Epochs | 1 |
| LLM learning rate | 5 x 10^-5 |
| Bridge learning rate | 5 x 10^-4 |
| LLM weight decay | 1 x 10^-5 |
| Bridge weight decay | 1 x 10^-5 |
| Scheduler | Cosine with warmup |
| Warmup | 25.6% of training steps |
| Batch size (effective) | 512 (16 per-device x 16 grad accum x 2 GPUs) |
| Precision | bfloat16 |
| Generation (inference) | Greedy decoding, max 96 tokens, repetition penalty 1.10 |

### Training Procedure

Training proceeded in two phases:

1. **Phase 1 — Bridge alignment** (1 epoch): The LLM backbone was frozen; only the Q-Former bridge and LoRA adapters were updated. This allows the expanded bridge (now using 8 codebooks and 10 layers vs. Stage 2's 1 codebook and 6 layers) to adapt to the full codebook input before the LLM fine-tuning begins.

2. **Phase 2 — Joint fine-tuning** (1 epoch): Both LoRA adapters and the bridge were updated jointly. Loss was computed on answer tokens only (prompt tokens masked with label -100).

Prompts followed the Gemma chat template with a system turn specifying the ECG analysis task, a user turn containing the instruction, and an assistant turn for the model's response. ECG token embeddings were injected into the user turn at the designated placeholder position.
