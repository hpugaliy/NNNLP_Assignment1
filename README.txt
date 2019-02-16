
usage: yunkim_cnn.py [-h] -d DATASET -e EMBED [-cf CNN_FILTERS] [--finetune]
                     [--dropout] [-f FILTERS] [-ep EPOCHS]

optional arguments:
  -h, --help            show this help message and exit
  -d DATASET, --dataset DATASET
                        Base Folder of the dataset
  -e EMBED, --embed EMBED
                        Path to fasttext or other embeddings
  -cf CNN_FILTERS, --cnn_filters CNN_FILTERS
                        Number of filters to be used in CNN
  --finetune            Option to enable finetuning of Fasttext embeddings
  --dropout             Option to enable dropout on convolution feature maps
  -f FILTERS, --filters FILTERS
                        Comma seperated list of filter sizes
  -ep EPOCHS, --epochs EPOCHS
                        Number of epochs to run


usage: baseline_simple.py [-h] -d DATASET [-m {cnn,rnn}]

optional arguments:
  -h, --help            show this help message and exit
  -d DATASET, --dataset DATASET
                        Base Folder of the dataset
  -m {cnn,rnn}, --model {cnn,rnn}
                        Baseline model to be trained



Some sample runs :

python yunkim_cnn.py -e ./wiki-news-300d-1M.vec -d ./topicclass -ep 20 -cf 100

python yunkim_cnn.py -e ./wiki-news-300d-1M.vec -d ./topicclass -ep 10 -cf 200

python yunkim_cnn.py -e ./wiki-news-300d-1M.vec -d ./topicclass -ep 10 -cf 300

python yunkim_cnn.py -e ./wiki-news-300d-1M.vec -d ./topicclass -ep 20 -cf 100 --dropout

python yunkim_cnn.py -e ./wiki-news-300d-1M.vec -d ./topicclass -ep 10 -cf 200 --dropout

python yunkim_cnn.py -e ./wiki-news-300d-1M.vec -d ./topicclass -ep 10 -cf 300 --dropout

python yunkim_cnn.py -e ./wiki-news-300d-1M.vec -d ./topicclass -ep 20 -cf 100  --filters 2,3,4,5 --dropout

python yunkim_cnn.py -e ./wiki-news-300d-1M.vec -d ./topicclass -ep 10 --filters 2,3,4,5 -cf 200 --dropout

python yunkim_cnn.py -e ./wiki-news-300d-1M.vec -d ./topicclass -ep 10 --filters 2,3,4,5 -cf 300 --dropout

python yunkim_cnn.py -e ./wiki-news-300d-1M.vec -d ./topicclass -ep 20 -cf 100  --filters 3,4,5,6 --dropout

python yunkim_cnn.py -e ./wiki-news-300d-1M.vec -d ./topicclass -ep 10 --filters 3,4,5,6 -cf 200 --dropout

python yunkim_cnn.py -e ./wiki-news-300d-1M.vec -d ./topicclass -ep 10 --filters 3,4,5,6 -cf 300 --dropout

python yunkim_cnn.py -e ./wiki-news-300d-1M.vec -d ./topicclass -ep 20 -cf 100  --filters 2,3,4,5 --dropout --finetune

python yunkim_cnn.py -e ./wiki-news-300d-1M.vec -d ./topicclass -ep 10 --filters 2,3,4,5 -cf 200 --dropout --finetune

python yunkim_cnn.py -e ./wiki-news-300d-1M.vec -d ./topicclass -ep 10 --filters 2,3,4,5 -cf 300 --dropout --finetune

python yunkim_cnn.py -e ./wiki-news-300d-1M.vec -d ./topicclass -ep 20 -cf 100  --filters 3,4,5,6 --dropout --finetune

python yunkim_cnn.py -e ./wiki-news-300d-1M.vec -d ./topicclass -ep 10 --filters 3,4,5,6 -cf 200 --dropout --finetune

python yunkim_cnn.py -e ./wiki-news-300d-1M.vec -d ./topicclass -ep 10 --filters 3,4,5,6 -cf 300 --dropout --finetune
