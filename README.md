# CSCI 2950 Character level language modeling in lyrics generation

## Group members
Shan Lu(slu5), Huan Lin(hl18), Yahui Wang(ywang90) Qian Mei(qmei)

## Usage
####training model
```shell
cd char-aware-model
python train.py --input_dir ../data/taylor_swift_split_line --check_dir ../cv/taylor_swift
```

####lyrics generation
We have included a pretrained model into this repository for demostration purpose.
```shell
cd char-aware-model
python sample.py --load_saved_checkpoint cv/taylor_swift/ep_000_6.3949.model --input_dir ../data/taylor_swift_pretrained/ep_019_4.0878.model
```

## Reference
1. Char-RNN by Karpathy
https://github.com/karpathy/char-rnn
2. Char-RNN-Tensorflow by sherjilozair
https://github.com/sherjilozair/char-rnn-tensorflow
3. LSTM language model with CNN over characters by Yoon Kim
https://github.com/yoonkim/lstm-char-cnn
4. LSTM language model with CNN over characters in TensorFlow by carpedm20
https://github.com/carpedm20/lstm-char-cnn-tensorflow
5. PyLyrics (A Pythonic Implementation of for getting lyrics of songs)
https://github.com/geekpradd/PyLyrics
