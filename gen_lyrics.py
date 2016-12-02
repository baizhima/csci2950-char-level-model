#!/usr/bin/python

from PyLyrics import *

import os, sys
import re
import collections

def _get_char_aware_cnn_split_by_song(collection_dirname, lyrics_list):
    fns = [os.path.join(collection_dirname, fn) for fn in ['train.txt', 'valid.txt', 'test.txt']]
    spl,spl2 = int(len(lyrics_list) * 0.7) ,int(len(lyrics_list) * 0.9)
    lyrics_splits = [lyrics_list[:spl], lyrics_list[spl:spl2], lyrics_list[spl2:]]
    for i, fn in enumerate(fns):
        with open(fn, 'wb') as f: 
            for ly in lyrics_splits[i]:
                f.write(ly)

def _get_char_aware_cnn_split_by_line(collection_dirname, lyrics_list):
    ''' Each lyric is splitted into three parts '''
    lyrics_splits = [[], [], []] # train_lyrics val_lyrics, test_lyrics = [], [], []
    for lyrics in lyrics_list:
        lines = lyrics.split('\n')
        print lines
        spl, spl2 = int(len(lines) * 0.7) ,int(len(lines) * 0.9)
        lyrics_splits[0].append('\n'.join(lines[:spl]))
        lyrics_splits[1].append('\n'.join(lines[spl:spl2]))
        lyrics_splits[2].append('\n'.join(lines[spl2:]))
    fns = [os.path.join(collection_dirname, fn) for fn in ['train.txt', 'valid.txt', 'test.txt']]
    for i, fn in enumerate(fns):
        with open(fn, 'wb') as f:
            for ly in lyrics_splits[i]:
                f.write(ly)
        print 'data written to ', fn

def get_collection_all_lyrics(singers_file, collection_name, split_by_line=True):
    '''
    Input a file containing a collection of singers
    Output: a file named 'input.txt' located in data/collection_name/input.txt
    '''
    singers = []
    with open(singers_file, 'rb') as f:
        for line in f:
            line = line.strip()
            if line:
                singers.append(line)
    print 'Singers are:'
    for s in singers:
        print s
    collection_dirname = os.path.join('data', collection_name.replace(' ', '_'))
    if split_by_line:
        collection_dirname += '_split_line'
    else:
        collection_dirname += '_split_song'
    if not os.path.exists(collection_dirname):
        os.mkdir(collection_dirname)
    filename = os.path.join(collection_dirname, 'input.txt')
    lyrics_list = []

    for singer in singers:
        albums = PyLyrics.getAlbums(singer)
        for album in albums:
            for track in album.tracks():
                curr_lyrics = get_track_lyrics_text(track, removeRepeat=2)
                lyrics_list.append(curr_lyrics)
    from random import shuffle
    shuffle(lyrics_list)
    
    with open(filename, 'wb') as f:
        for lyrics in lyrics_list:
            f.write(lyrics + '\n')

    preprocessed_lyrics_list = preprocess_data(lyrics_list)

    if split_by_line:
        _get_char_aware_cnn_split_by_line(collection_dirname, preprocessed_lyrics_list)
    else:
        _get_char_aware_cnn_split_by_song(collection_dirname, preprocessed_lyrics_list)

def get_singer_all_lyrics(singer, writeToFile=True):
    '''
    Input: a singer's name (e.g. Ariana Grande)
    Output: a file named 'input.txt' located in data/singer/input.txt
    '''
    singer_dirname = os.path.join('data', singer.replace(' ','_'))
    if not os.path.exists(singer_dirname):
        os.mkdir(singer_dirname)

    albums = PyLyrics.getAlbums(singer)
    filename = os.path.join(singer_dirname, 'input.txt')
    with open(filename, 'wb') as f:
        for album in albums:
            for track in album.tracks():
                print '<<%s>>'%track
                lyrics_text = get_track_lyrics_text(track, removeRepeat=2)
                f.write(lyrics_text + '\n')

def get_track_lyrics_text(track, removeRepeat=2):
    txt = track.getLyrics()
    if not txt:
        return ''
    txt = txt.split('\n')
    res = []
    prev_line = [None for i in range(removeRepeat)]
    for line in txt:
        if line not in prev_line:
            print line
            res.append(line)
        for i in range(removeRepeat-1):
            prev_line[i] = prev_line[i+1]
        prev_line[removeRepeat-1] = line
    return '\n'.join(res)

def basic_tokenizer(sentence, word_split=re.compile(b"<>([.,!?\"':;)(])")):
    def is_ascii(s):
        return all(ord(c) < 128 for c in s)
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(word_split, space_separated_fragment))
    return [w.lower() for w in words if w and is_ascii(w)]

def preprocess_data(lyrics_list):
    data_tokens = []
    data_ids = []
    UNK_SYMBOL = '<unk>'
    TOP_FREQUENCY = 8000
    vocab_count_dict = {}
    for ly in lyrics_list:
        lines = ly.split('\n')
        for line in lines:
            tokens = basic_tokenizer(line)
            data_tokens.extend(tokens)
            for tok in tokens:
                if tok not in vocab_count_dict:
                    vocab_count_dict[tok] = 1
                else:
                    vocab_count_dict[tok] += 1
    counter = collections.Counter(data_tokens)
    count_pairs = sorted(counter.items(), key=lambda x:(-x[1], x[0]))
    words, _ = list(zip(*count_pairs[:TOP_FREQUENCY]))
    word_set = set(words)
    preprocessed_lyrics_list = []
    for ly in lyrics_list:
        clean_lyrics = []
        lines = ly.split('\n')
        for line in lines:
            tokens = basic_tokenizer(line)
            clean_line = []
            for tok in tokens:
                if tok in word_set:
                    clean_line.append(tok)
                else:
                    clean_line.append(UNK_SYMBOL)
            clean_lyrics.append(' '.join(clean_line))
        preprocessed_lyrics_list.append('\n'.join(clean_lyrics))
    return preprocessed_lyrics_list


if __name__ == '__main__':
    if len(sys.argv) <= 3 or not sys.argv[1].startswith('-'):
        print 'Usage: python gen_lyrics.py -s Taylor Swift'
        print 'Usage: python gen_lyrics.py -c singers_name_file collection_name'
    if sys.argv[1] == '-s':
        singer = ' '.join(sys.argv[2:])
        print 'Singer:%s'%singer
        get_singer_all_lyrics(singer)
    elif sys.argv[1] == '-c':
        get_collection_all_lyrics(sys.argv[-2], sys.argv[-1],split_by_line=False)
        get_collection_all_lyrics(sys.argv[-2], sys.argv[-1],split_by_line=True)