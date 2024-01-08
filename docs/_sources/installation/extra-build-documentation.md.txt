# Building the documentation

This page includes instructions on how to build the documentation files you are currently reading.

## Setting up

```{tip}
If you've already followed the instructions in {ref}`Building the database <build-database-setup>`., you can skip this stage.
```

Clone our repository to a new directory on your local machine:
```
git clone https://github.com/HuwCheston/Cambridge-Jazz-Trio-Database
```

In the repository root directory, create a new virtual environment, enter it, and install project dependencies (these are required for `sphinx.ext.autodoc`):
```
pip install virtualenv virtualenvwrapper
python -m venv venv
python test_environment.py
call venv\Scripts\activate.bat
pip install -r requirements.txt
```

## Install documentation requirements

Run the following command to install the necessary packages for building the documentation (this includes `Sphinx`, the `pydata` theme, as well as a few common extensions like `myst_parser`):
```
pip install -r _docssrc\requirements.txt
```

## Build documentation

From the project root directory, run the following to build the documentation HTML files:
```
sphinx-apidoc -o .\_docssrc\src .\src
sphinx-build .\_docssrc .\docs
```

You can now access the HTML files in `.\docs`. Start with `.\docs\index.html`, and navigate the rest of the site from there.