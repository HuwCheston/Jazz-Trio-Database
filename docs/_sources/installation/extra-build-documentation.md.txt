# Building the documentation

This page includes instructions on how to build the documentation files you are currently reading.

## Setting up

Clone our repository to a new directory on your local machine:
```
git clone https://github.com/HuwCheston/Cambridge-Jazz-Database
```

In the repository root directory, create a new virtual environment, enter it, and install project dependencies (these are required for `sphinx.ext.autodoc`):
```
pip install virtualenv virtualenvwrapper
python -m venv venv
python test_environment.py
call venv\Scripts\activate.bat
pip install -r requirements.txt
```

Run the following command to install the necessary packages for building the documentation (this includes `Sphinx`, the `pydata` theme, as well as a few common extensions like `myst_parser`):
```
pip install -r docs\requirements.txt
```

## Build documentation

From the `.\docs` directory, run the following to build the documentation HTML files:
```
sphinx-apidoc -o .\src ..\src
sphinx-build . .\_build
```

You can now access the HTML files in `.\docs\_build`. Start with `.\docs\_build\index.html`, and navigate the rest of the site from there.

## Hosting on GitHub pages

In order to get the results from `sphinx-build` set up and running on GitHub pages, after building the documentation I've found it's often necessary to then copy the contents of `.\docs\_build` to `docs`, overwriting any files that are currently in that directory.