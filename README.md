# imageQCpy
Python version of imageQC

v3.0.XXalpha - some functions not finished yet and not fully tested

imageQC is a software-tool for the Medical Physicist performing quality control on medical imaging devices or extracting specific DICOM header data. The software exists in IDL-code, but is currently converted and upgraded to Python code.
Get notified when a beta version is available. Send me an e-mail ellen.wasbo(-the-curly-a-)sus.no and I'll put you on the notify-list.

* [Run the python code in dedicated virtual environment](#run-the-python-code-in-dedicated-virtual-environment)
* [Run the python code without dedicated virtual environment](#run-the-python-code-without-dedicated-virtual-environment)
* [Run python code with Spyder](#run-python-code-with-spyder)

### Run the python code in dedicated virtual environment

Clone the repository from GitHub or download the zip. Then create a virtual environment. If using Anaconda start the cmd.exe Prompt:

```bash
conda create --name viQC python=3.9.7
conda activate viQC
cd <to the folder above folder src>
pip install -e .
```

To run the application:

```bash
python -m imageQC.imageQC
```

If automation templates are created (using the GUI version) import and/or automation can be started from codeline without the GUI version popping up:

```bash
python -m imageQC.imageQC [-i/-import [-ndays=n] [-a/-auto | -v/-vendor | -d/-dicom] [<modality>] [<modality>/<template_label>]
```

### Run the python code without dedicated virtual environment

Even though virtual environment should be the preferred way to run imageQC, some might have trouble running the code this way. I have issues on my
hospital computer - first I ran into trouble with setuptools using ```bash pip install -e .``` as above. I tried upgrading to latest Python version 
and then I was no longer allowed to create virtual environment.

If you run other Python programs than imageQC, please make sure that the required python packages (requirements.txt) are not in conflict with your other
Python programs. You could try deleting those lines from requirements.txt. (This is the reason why virtual environment is preferred).

Clone the repository from GitHub or download the zip. Then using Anaconda start the cmd.exe Prompt:

```bash
cd <to the folder above folder src>
pip install --user --proxy=http://<proxyserver_name>:<port#> -r requirements.txt
```
On my hospital computer I need the --proxy=http://<proxyserver_name>:<port#> as an argument to the pip install arguments. If your not behind a proxy server
skip that argument.

Now to run the code you will need to run a script to redefine import statements (they need to be different than in the solution installing imageQC as a package).

Still in the folder above src:
```bash
cd <to the helper_scripts folder>
python -m replace_import.py
```
Locate src folder in the popup dialog. Yes to remove imageQC. from the import statements.

Now go back to the folder above src and then to the src/imageQC folder.
```bash
cd ..
cd src\imageQC
python -m imageQC
```
Next time cd to src\imageQC and run imageQC is the only thing you will need to start the program. 
On my hospital computer I needed to uninstall PyQt5 after installing all from requirements.txt as Spyder needed an older version. 
If that is also the case for you:
```bash
pip uninstall PyQt5
```
Verify that you remove it from your user folder and accept with Y for yes. Then you can try running imageQC again.


### Run python code with Spyder

Do the same as above except for tha last bit on how to run imageQC. Open ...src\imageQC\imageQC.py in Spyder and run the file.


Here is a screen shot:
![image](https://user-images.githubusercontent.com/16964680/202554613-13be30f4-e159-4f3e-8667-f4bd7bed082b.png)
