import json

import tensorflow as tf
import torch
from json import dump

assert(int(tf.__version__.split('.')[0]) == 2)


def convert_h5_to_json(model_h5_file, model_json_file):
    """
    Helper function to convert tf2 stored model h5 file to a customized json
    format.

    Args:
        model_h5_file(string): filename of the stored h5 file
        model_json_file(string): filename of the output json file
    """

    model = tf.keras.models.load_model(model_h5_file)
    json_dict = {}

    for l in model.layers:
        json_dict[l.name] = {
            'input_shape': l.input_shape[1:],
            'output_shape': l.output_shape[1:],
            'num_neurons': l.output_shape[-1]
        }

        if 'conv' in l.name:
            all_weights = l.weights[0]
            neuron_weights = []

            # Iterate through neurons in that layer
            for n in range(all_weights.shape[3]):
                cur_neuron_dict = {}
                cur_neuron_dict['bias'] = l.bias.numpy()[n].item()

                # Get the current weights
                cur_weights = all_weights[:, :, :, n].numpy().astype(float)

                # Reshape the weights from (height, width, input_c) to
                # (input_c, height, width)
                cur_weights = cur_weights.transpose((2, 0, 1)).tolist()
                cur_neuron_dict['weights'] = cur_weights

                neuron_weights.append(cur_neuron_dict)

            json_dict[l.name]['weights'] = neuron_weights

        elif 'output' in l.name:
            all_weights = l.weights[0]
            neuron_weights = []

            # Iterate through neurons in that layer
            for n in range(all_weights.shape[1]):
                cur_neuron_dict = {}
                cur_neuron_dict['bias'] = l.bias.numpy()[n].item()

                # Get the current weights
                cur_weights = all_weights[:, n].numpy().astype(float).tolist()
                cur_neuron_dict['weights'] = cur_weights

                neuron_weights.append(cur_neuron_dict)

            json_dict[l.name]['weights'] = neuron_weights

    dump(json_dict, open(model_json_file, 'w'), indent=2)

def convert_pt_to_json(model_pt_file, model_json_file):
    """
    Helper function to convert PyTorch stored model pt file to a customized json
    format.

    Args:
        model_pt_file (string): filename of the stored pt file
        model_json_file (string): filename of the output json file
    """

    # Load the PyTorch model
    model = torch.load(model_pt_file)

    # Initialize an empty dictionary to store layer information
    json_dict = {}

    # Iterate through model layers
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            json_dict[name] = {
                'type': 'convolutional',
                'input_channels': module.in_channels,
                'output_channels': module.out_channels,
                'kernel_size': module.kernel_size,
                'bias': module.bias.tolist(),
                'weights': module.weight.tolist()
            }
        elif isinstance(module, torch.nn.Linear):
            json_dict[name] = {
                'type': 'linear',
                'input_features': module.in_features,
                'output_features': module.out_features,
                'bias': module.bias.tolist(),
                'weights': module.weight.tolist()
            }

    # Write the dictionary to a JSON file
    with open(model_json_file, 'w') as json_file:
        json.dump(json_dict, json_file, indent=2)
