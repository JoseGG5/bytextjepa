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

At training time, the batch should contain multiple views per sample. A convenient tensor shape is:

- `input_ids`: `[B, V, T]`
- `attention_mask`: `[B, V, T]`

Where:

- `B` is the batch size
- `V` is the number of views per sample
- `T` is the padded sequence length of each crop

The model forward should flatten the batch and view dimensions, run the shared encoder on every crop, and then pool each crop representation into a single latent vector. The simplest pooling for a first version is masked mean pooling over the token dimension.

Conceptually:

```python
def forward(input_ids, attention_mask):
    # input_ids: [B, V, T]
    # attention_mask: [B, V, T]

    flat_ids = input_ids.view(B * V, T)
    flat_mask = attention_mask.view(B * V, T)

    hidden = encoder(
        input_ids=flat_ids,
        attention_mask=flat_mask,
    ).last_hidden_state  # [B * V, T, D]

    z = masked_mean(hidden, flat_mask)  # [B * V, D]
    z = z.view(B, V, D)                 # [B, V, D]

    return {"z": z}
```

So the encoder still produces token-level representations internally, but the SSL objective acts on one pooled latent vector per view:

- `z[n, v]` = embedding of view `v` for sample `n`
- final latent tensor shape = `[B, V, D]`

This is the cleanest first implementation because it avoids token-level matching while keeping the setup faithful to the paper's view-prediction perspective.

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

If this baseline works, later iterations can make the setup more structured by:

- designing better text crops
- adding sentence-aware or span-aware sampling
- moving from simple pooled view embeddings to span-level latent prediction
- experimenting with chunking strategies beyond the one-byte baseline
