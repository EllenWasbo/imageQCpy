# imageQCpy
Python version of imageQC

v3.0.0alpha - some functions not finished yet and not fully tested

imageQC is a software-tool for the Medical Physicist performing quality control on medical imaging devices or extracting specific DICOM header data. The software exists in IDL-code, but is currently converted and upgraded to Python code. Hopefully a beta-version will be available at the end of 2022.
Get notified when a new version is available. Send me an e-mail ellen.wasbo(-the-curly-a-)sus.no and I'll put you on the notify-list.

### Install the python package

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

Here is a sneak peek:
![image](https://user-images.githubusercontent.com/16964680/202554613-13be30f4-e159-4f3e-8667-f4bd7bed082b.png)
