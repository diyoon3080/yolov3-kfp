from kfp import components
from kfp.components import OutputPath

def load_train_data(
    train_dataset_url: str, 
    val_dataset_url: str, 
    train_dataset: OutputPath('Dataset'), 
    val_dataset: OutputPath('Dataset')
):
    import gdown
    gdown.download(train_dataset_url, output=train_dataset, quiet=True, fuzzy=True)
    gdown.download(val_dataset_url, output=val_dataset, quiet=True, fuzzy=True)
    print(f'download complete!')

components.create_component_from_func(
    load_train_data, 
    output_component_file='./component-files-yaml/load_train_data_component.yaml',
    packages_to_install=['gdown']
)