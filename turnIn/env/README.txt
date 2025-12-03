STEPS TO CREATE AND USE VIRTUAL ENVIROMENT
1. Create virtual envrionment
    This can be done by 'python -m venv myenv'
2. Activate virtual environment
        Linux based:
            source myenv/bin/activate
        Windows(command prompt):
            myenv\Scripts\activate
        Windows(powershell):
            .\myenv\Scripts\Activate.ps1
3. Deactivate virtual environment
    deactivate

*. Because system global variables are different depending on the OS different paths in the environments are used
    during creation which means that a virtual environment created in a linux system will not work in a powershell system,
    even though there is a powershell activate file inside of the linux virtual environment. This applies vise versa too.

*. You cannont rename virtual environment:
    because absoulute paths are used in its creation
        you instead have to create a new one, see installing packages for a quick way to sync virtual environments.

*. Installing packages:
    a. once you've activated the environment you can install package using,
        pip install packageName 

    b. you can also use a requirments file to install/update multiple packages for example,
        pip install -r filename.txt

    c. you can get the dependenceies inside a virtual environment by running,
        pip freeze

    d. you can save the dependenceies inside a virtual environment by running,
        pip freeze > filename.txt

    notice that b. and d. work together to efficently make venvs that are in sync,
    because of this in our directory called requirements.txt that we will use to keep
    everything up to date

*. you can see what's inside virtual environtment by running,
        pip list
    or
        pip freeze

