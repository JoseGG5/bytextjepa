Joint Embedding Predictive Architectures are so cool because they predict in the latent space, avoiding spending compute in useless reconstruction at discrete domains. Some people (included myself) believe that this architectures are really exciting as a research line as they shine in real-world, messy data and generate robust high-quality representations that allow world models to plan long horizon tasks on a generalizable manner.

While as I say, JEPA shines in high dimensional, messy data such as audio, image and video (note that these three modalities come from the real world, and thus contain much more entropy than text, which has been created by humans and information is really compressed there), here I want to use them to redefine how text encoders are trained. Typically, a text encoder is trained with Masked Language Modeling (MLM), which involve masking parts (tokens) of a sentence and letting the model guess which tokens fit. This involves predicting in a discrete space (the vocab size of the tokenizer), and I think continuous spaces are richer and models should learn in continuous spaces.

For my experiment I simply want to throw away complex tokenizers (which makes it multilingual by default). For this, I propose to convert text to bytes and have just a vocab size of 255 (number of bytes). Tokens will be blocks of bytes, and I will test different chunking strategies (from fixed size to letting the model learn how to cut). The encoder used will be ModernBERT, a state-of-the-art encoder model with a large context window and interesting mechanism such as alterning full attention with window sliding (local) attention.

JEPA style training will include the typical JEPA loss + Sketched Isotropic Gaussian Regularization (SIGReg). No EMA heuristics will be used thanks to SIGReg.

The dataset will be a full Spanish text oriented to medical domain (IIC/ClinText-SP)

-------------------

## Experiment sketch

The current idea of the experiment is to stay closer to the LeJEPA paper than to a BYOL-style or I-JEPA-style teacher-student setup. That means the training recipe should use a single shared encoder, multiple views of the same sample, a Euclidean prediction/alignment term in latent space, and SIGReg as the anti-collapse mechanism. In other words, the goal is not to predict missing bytes directly, but to make the embedding of one view predictable from the embeddings of the other views while keeping the global embedding distribution well-conditioned.

For text, one training sample should correspond to one raw dataset record. For each `dataset[i]`, the pipeline should first convert the text to UTF-8 bytes and then sample several crops from that same byte sequence. These crops act as the different views of the same semantic sample. A practical first version would use two global views and a few local views:

- `global_view_1`: a long byte crop from the record
- `global_view_2`: another long byte crop from the same record, ideally overlapping or nearby
- `local_views`: shorter byte crops sampled inside or near the global region

This mirrors the multi-crop logic used in vision, but adapted to text. The important constraint is that all views for a sample come from the same original record, so the latent prediction task remains meaningful.

## Model forward during training

At training time, the batch should contain multiple views per sample, but global and local crops do not need to share the same padded sequence length. Since local views are intentionally shorter than global ones, the cleanest setup is to keep them in separate tensors:

- `global_input_ids`: `[B, Vg, Tg]`
- `global_attention_mask`: `[B, Vg, Tg]`
- `local_input_ids`: `[B, Vl, Tl]`
- `local_attention_mask`: `[B, Vl, Tl]`

Where:

- `B` is the batch size
- `Vg` is the number of global views per sample
- `Vl` is the number of local views per sample
- `Tg` is the padded sequence length used for global crops
- `Tl` is the padded sequence length used for local crops

The model forward should flatten the batch and view dimensions within each crop group, run the shared encoder separately on global and local crops, and then pool each crop representation into a single latent vector. The simplest pooling for a first version is masked mean pooling over the token dimension.

Conceptually:

```python
def forward(global_input_ids, global_attention_mask,
            local_input_ids, local_attention_mask):
    # global_input_ids: [B, Vg, Tg]
    # global_attention_mask: [B, Vg, Tg]
    # local_input_ids: [B, Vl, Tl]
    # local_attention_mask: [B, Vl, Tl]

    flat_global_ids = global_input_ids.view(B * Vg, Tg)
    flat_global_mask = global_attention_mask.view(B * Vg, Tg)
    flat_local_ids = local_input_ids.view(B * Vl, Tl)
    flat_local_mask = local_attention_mask.view(B * Vl, Tl)

    global_hidden = encoder(
        input_ids=flat_global_ids,
        attention_mask=flat_global_mask,
    ).last_hidden_state  # [B * Vg, Tg, D]

    local_hidden = encoder(
        input_ids=flat_local_ids,
        attention_mask=flat_local_mask,
    ).last_hidden_state  # [B * Vl, Tl, D]

    z_global = masked_mean(global_hidden, flat_global_mask)  # [B * Vg, D]
    z_local = masked_mean(local_hidden, flat_local_mask)     # [B * Vl, D]

    z_global = z_global.view(B, Vg, D)                       # [B, Vg, D]
    z_local = z_local.view(B, Vl, D)                         # [B, Vl, D]
    z = torch.cat([z_global, z_local], dim=1)                # [B, V, D]

    return {"z": z}
```

So the encoder still produces token-level representations internally, but the SSL objective acts on one pooled latent vector per view:

- `z[n, v]` = embedding of view `v` for sample `n`
- final latent tensor shape = `[B, V, D]`

This is the cleanest first implementation because it avoids token-level matching while keeping the setup faithful to the paper's view-prediction perspective. It also matches the usual multi-crop strategy: crops with different effective sizes are processed in separate forward passes and only merged after pooling into view-level latents.

## Loss during training

The LeJEPA-style loss for this text experiment should combine two terms:

1. A prediction/alignment term across views from the same sample
2. A SIGReg term across the batch to prevent collapse and encourage isotropic Gaussian embeddings

Let the first `Vg` views be the global views. For each sample `n`, compute the mean latent of the global views:

```python
mu_n = mean(z[n, :Vg, :])  # [D]
```

Then make every view of that sample match that global center:

```python
pred_loss = ((mu[:, None, :] - z) ** 2).mean()
```

Where:

- `mu` has shape `[B, D]`
- `z` has shape `[B, V, D]`

This is the Euclidean latent prediction term. It encourages all views of the same record to agree in embedding space. Unlike MLM, the target is not a discrete byte identity. The target is another latent representation coming from a related view of the same underlying sample.

The second term is SIGReg, applied on the embeddings of each view across the batch:

```python
sigreg = mean(SIGReg(z[:, v, :]) for v in range(V))
```

So for a fixed view index `v`, `z[:, v, :]` has shape `[B, D]`, and SIGReg pushes that batch of embeddings toward an isotropic Gaussian distribution.

The final loss is:

```python
loss = (1 - lambda_) * pred_loss + lambda_ * sigreg
```

This is the key conceptual picture for the experiment:

- one record produces several byte-level crops
- one encoder maps each crop to one pooled latent vector
- global views define a per-sample latent center
- all views are pulled toward that center
- SIGReg prevents trivial collapse and replaces the need for EMA, stop-gradient asymmetry, or a second encoder

## What this experiment is and is not

This first version should be understood as a LeJEPA-inspired text pretraining experiment, not as a byte-level masked reconstruction objective and not as a classic teacher-student JEPA. The model is not asked to reconstruct missing bytes, and it is not asked to predict a vocabulary distribution. Instead, it is trained so that multiple byte-based views of the same text record map to a shared, stable latent structure.

The main idea is to **see if this approach works as a research line, and then,
try to see how close it gets to normal MLM pretraining for a fixed compute**. After this, if this baseline works, later iterations can make the setup more structured by:

- designing better text crops
- adding sentence-aware or span-aware sampling
- moving from simple pooled view embeddings to span-level latent prediction
- experimenting with chunking strategies beyond the one-byte baseline

