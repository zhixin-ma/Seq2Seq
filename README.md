# Seq2Seq
An implementation of Seq2Seq (in **MXNet**) using LSTM cell

**usage example**
```python3
model = seq2seq(vocab_size, dim_vec, dim_h, num_layers, dropout)
model.initialize(ctx=ctx)
# Train
output = seq2seq(query, label, ctx=ctx)
```

