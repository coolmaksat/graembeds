import click as ck
import numpy as np
import tensorflow as tf
from utils import Dataset
from importlib import import_module

@ck.command()
@ck.option('--dataset', '-ds', default='FB15k', help='Dataset name in data folder') 
@ck.option('--model-name', '-m', default='TransE', help='Model class name') 
@ck.option('--device', '-d', default='gpu:0', help='Device name') 
@ck.option('--train', is_flag=True, help='Train model') 
@ck.option('--epochs', '-e', default=100, help='Number of training epochs') 
def main(dataset, model_name, device, train, epochs):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    # config.log_device_placement = True
    
    ds = Dataset(dataset)
    print('Number of nodes: ', ds.nb_nodes)
    print('Number of relations: ', ds.nb_relations)
    mod = import_module(model_name.lower())
    model_class = getattr(mod, model_name)
    model_path = './data/' + model_name.lower() + '.ckpt'
    with tf.device('/' + device):
        model = model_class(ds, model_path)
        init = tf.global_variables_initializer()
        
        with tf.Session(config=config) as sess:
            sess.run(init)

            if train:
                model.train(
                    sess,
                    epochs=epochs
                )
            
            model.evaluate(sess)
            
    
if __name__ == '__main__':
    main()
