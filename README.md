# TextLevelGCN
Source code of our paper presents in EMNLP 2019. https://www.aclweb.org/anthology/D19-1345/

A list of parameters required to run the code can be obtained in the following way.

`python3 train.py -h`

An example: 
`python3 train.py --dataset r8 --ngram 3`
`python3 train.py --dataset r52 --ngram 3`
`python3 train.py --dataset oh --ngram 4`

Note: You need to put a pre-trained Glove model file named `glove.6B.200d.vec.txt` in the root directory.
You get the file from https://nlp.stanford.edu/projects/glove/.
You need to **add a line at the top of the file** with the following value:

`400000 200`
