from kfp import components
from kfp.components import OutputPath

def load_test_img(img_url: str, input_img: OutputPath('jpg')):
    import gdown
    gdown.download(img_url, output=input_img, quiet=True, fuzzy=True)
    print(f'download complete!')

components.create_component_from_func(
    load_test_img, 
    output_component_file='./component-files-yaml/load_test_img_component.yaml',
    packages_to_install=['gdown']
)