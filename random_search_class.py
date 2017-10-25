import numpy as np


def param_gen(**kwargs):
    params_sample = dict()
    for param in kwargs:
        options = kwargs[param]
        if 'bool' in options and options['bool']:
            sample = np.random.rand() > 0.5
        elif 'range' in options:
            from_ = options['range'][0]
            to_ = options['range'][1]
            if 'scale' in options and options['scale'] == 'log':
                sample = np.exp(np.random.uniform(np.log(from_), np.log(to_)))
                if 'discrete' in options:
                    sample = int(np.round(sample))
            elif 'discrete' in options and options['discrete']:
                sample = np.random.randint(from_, to_ + 1)
            else:
                sample = np.random.uniform(from_, to_)
        params_sample[param] = sample
    return params_sample


if __name__ == '__main__':
    network_params = param_gen(n_conv_layers={'range': [1, 5], 'discrete': True},
                               token_embeddings_dim={'range': [50, 300], 'discrete': True},
                               char_embeddings_dim={'range': [10, 100], 'discrete': True},
                               filter_width={'range': [2, 7], 'discrete': True},
                               embeddings_dropout={'bool': True})
    learning_params = param_gen(batch_size={'range': [2, 64], 'discrete': True},
                                dropout_rate={'range': [0.2, 0.6]},
                                learning_rate={'range': [5e-4, 5e-2], 'scale': 'log'},
                                epochs={'range': [2, 10], 'discrete': True, 'scale': 'log'})
    print(network_params, learning_params)
    from cnnner.ner import NER
    from cnnner.corpus import Corpus
    from cnnner.data_tools import snips_reader
    from cnnner.data_tools import dataset_slicer
    xy_list = snips_reader()
    dataset = dataset_slicer(xy_list, 0.8, 0.1, 0.1)
    corp = Corpus(dataset)
    ner = NER(corp, **network_params)
    ner.fit(**learning_params)