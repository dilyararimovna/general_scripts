
def init_from_scratch(model_function, **kwargs):
    model = model_function(**kwargs)
    return model

def save(model, fname=None):
    if fname:
        model.save_weights(fname + '.h5')
        print('SAVED model\n')
    else:
        print('NOT saved. No file name provided\n')
    return

def init_from_saved(model_function, fname, **params):
    model = model_function(**params)
    print('Loading model weights from %s ' % fname)
    model.load_weights(fname + '.h5')
    return model