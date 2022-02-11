class Config:
    max_len = 22
    vocab_size = 32000
    num_head = 8
    d_model = 768
    d_ff = 2048
    drop_rate = 0.1
    data_dir = 'data'
    fn = 'ckpt'
    model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'