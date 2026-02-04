# GPT-1 from scratch 
Implemented Generative Pre-Trained Transformer from scratch using PyTorch. 

---

Research Paper : 
[Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

Hyperparameters :
```python
    block_size = 128
    batch_size=12
    epochs = 5 

    model = GPT(
        src_vocab_size=vocab_size,
        target_vocab_size=vocab_size,
        src_pad_index=0,
        target_pad_index=0,
        embed_size=128,
        num_layers=4, # 12 in research paper
        heads=4,
        forward_expansion=4,
        dropout=0.1,
        max_length=block_size,
        device=device,
    ).to(device)
```

Optimizer :
```python
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
```

Loss Function :
```python
    criterion = nn.CrossEntropyLoss(ignore_index=0)
```

Parameters :
```
Total parameters: 1,080,259
Trainable parameters: 1,080,259
```

Trained on :
```
NVIDIA GeForce GTX 1650
4GB VRAM
CUDA Version: 12.5
```