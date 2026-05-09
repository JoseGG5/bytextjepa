Joint Embedding Predictive Architectures are so cool because they predict in the latent space, avoiding spending compute in useless reconstruction at discrete domains. Some people (included myself) believe that this architectures are really exciting as a research line as they shine in real-world, messy data and generate robust high-quality representations that allow world models to plan long horizon tasks on a generalizable manner.

While as I say, JEPA shines in high dimensional, messy data such as audio, image and video (note that these three modalities come from the real world, and thus contain much more entropy than text, which has been created by humans and information is really compressed there), here I want to use them to redefine how text encoders are trained. Typically, a text encoder is trained with Masked Language Modeling (MLM), which involve masking parts (tokens) of a sentence and letting the model guess which tokens fit. This involves predicting in a discrete space (the vocab size of the tokenizer), and I think continuous spaces are richer and models should learn in continuous spaces.

For my experiment I simply want to throw away complex tokenizers (which makes it multilingual by default). For this, I propose to convert text to bytes and have just a vocab size of 255 (number of bytes). Tokens will be blocks of bytes, and I will test different chunking strategies (from fixed size to letting the model learn how to cut). The encoder used will be ModernBERT, a state-of-the-art encoder model with a large context window and interesting mechanism such as alterning full attention with window sliding (local) attention.

JEPA style training will include the typical JEPA loss + Sketched Isotropic Gaussian Regularization (SIGReg). No EMA heuristics will be used thanks to SIGReg.

The dataset will be a full Spanish text oriented to medical domain (IIC/ClinText-SP)

