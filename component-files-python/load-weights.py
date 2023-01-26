from kfp import components
from kfp.components import OutputPath

def load_weights(checkpoint_url: str, pretrained_weights: OutputPath('Weights')):
    import gdown
    gdown.download_folder(checkpoint_url, output=pretrained_weights, quiet=True, use_cookies=False)
    print(f'download complete!')

components.create_component_from_func(
    load_weights,
    output_component_file='./component-files-yaml/load_weights_component.yaml',
    packages_to_install=['gdown']
)