class seq2seq(nn.Block):
    def __init__(self, vocab_size, dim_vec, dim_h, num_layers, dropout, **kwargs):
        super(seq2seq,self).__init__(**kwargs)
        with self.name_scope():
            self.dim_h = dim_h
            self.dim_vec = dim_vec
            self.num_layers = num_layers
            self.word_embedding = nn.Embedding(vocab_size, dim_vec)
            self.lstm_1 = rnn.LSTM(hidden_size=dim_h, dropout=dropout, layout='NTC', num_layers=num_layers)
            self.lstm_2 = rnn.LSTM(hidden_size=dim_h, dropout=dropout, layout='NTC', num_layers=num_layers)
            self.mlp = nn.Dense(units=vocab_size, flatten=False)

    def forward(self, query, label, ctx):
        batch_size = query.shape[0]
        query = self.Embed(query)
        label = self.Embed(label)

        h_0 = [nd.zeros((self.num_layers, batch_size, self.dim_h), ctx=ctx) for i in range(2)]
        output, states = self.lstm_1(query, h_0)
        output, states = self.lstm_2(label, states)

        return states

        # # one by one decode
        # outputs = []
        # for it in range(len(label)):
        #     if it == 0:
        #         output, states = self.decoder(
        #             init_input, states)
        #     else:
        #         output, states = self.decoder(
        #             label[it-1], states)
        # 
        #     outputs.append(output) 
        # pred = self.mlp(nd.concat(*outputs))
        # return pred

    def test(self, query, max_sql_len):
        batch_size = query.shape[0]
        query = self.Embed(query)
        label = self.Embed(label)

        h_0 = [nd.zeros((self.num_layers, batch_size, self.dim_h), ctx=ctx) for i in range(2)]
        output, states = self.lstm_1(query, h_0)

        # one by one decode
        outputs = []
        init = nd.zeros(shape=(batch_size, 1, self.dim_vec))
        for idx in range(len(label)):
            if idx == 0:
                output, states = self.decoder(init, states)
            else:
                unit_in = self.word_embedding(cur_pred)
                output, states = self.decoder(unit_in, states)

            cur_pred = nd.softmax(self.mlp(output))
            cur_pred = cur_pred.argmax(axis=1)
            outputs.append(cur_pred)
        pred = nd.concat(*outputs)
        return pred

