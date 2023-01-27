from kfp import components
from kfp.components import InputPath, OutputPath

def serve(temp_var: InputPath('jpg')):
    if temp_var:
        print('Model served successfully.')
    else:
        print('There was an error serving the model.')

components.create_component_from_func(
    serve,
    output_component_file='./component-files-yaml/serve_component.yaml'
)