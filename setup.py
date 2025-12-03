from setuptools import find_packages,setup
from typing import List

HYPE_E_DOT='-e .'
def get_requiements(file_path:str)->List[str]:
    '''
    This function will return the list of requirements
    '''
    requiremets=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","")for req in requirements]

        if HYPE_E_DOT in requirements:
            requirements.remove(HYPE_E_DOT)
    return requirements        
setup (
   name='mlproject',
   version='0.0.1',
   author='piyush',
   packages=find_packages(),
   install_requires=get_requirements('requirements.txt')


)