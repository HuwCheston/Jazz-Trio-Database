# INSTALL

---

## Requirements:

Download and install the following:
- [Python 3.10.2](https://www.python.org/downloads/release/python-3102/)
  - Make sure that you choose to add Python 3.10.2 to your PATH during installation, and that it is at the top of your PATH.
  - To check this, follow [this StackOverflow answer](https://stackoverflow.com/a/61691955).
- [PyCharm](https://www.jetbrains.com/pycharm/)
  - Either Community or Professional edition should suffice
  - Not necessary for building the dataset but will be if you want to edit the code.
- [Git](https://git-scm.com/)
- [FFmpeg](https://ffmpeg.org/download.html)
  - You may need an unarchiving tool to extract the 7z files, e.g. [WinRAR](https://www.win-rar.com/start.html?&L=0)
  - The `ffmpeg.exe` and other executable files need to be extracted to a directory that is accessible on your PATH. See [the instructions here](https://gist.github.com/nex3/c395b2f8fd4b02068be37c961301caa7).
  - To check if ffmpeg is installed correctly and accessible, open up a new command prompt and type `ffmpeg`. If you get a whole load of text and no errors, you’re good to go!

## Clone the repository:

Open up the directory where you want to create the project folder. Then, in a command prompt, run:
```
git clone https://github.com/HuwCheston/jazz-corpus-analysis
```

This should create a new folder called `jazz-corpus-analysis` that mirrors the file structure on GitHub.

## Create a new virtual environment:

Open up a new command prompt and type `where pip`. The first line should end with something like `\Python\Python310\Scripts\pip.exe`
> If you get a different Python version, e.g. 3.6, instead, you’ll need to adjust the ordering of Python versions on your PATH. See [this StackOverflow answer](https://stackoverflow.com/a/61691955).

Run the following command in a command prompt

```
pip install pip virtualenv virtualenvwrapper wheel click
```

Now, inside your `jazz-corpus-analysis` folder, run `virtualenv venv`. You should end up with a new folder called `venv` inside `jazz-corpus-analysis`

> To check this worked ok, open up a command prompt in `jazz-corpus-analysis` and run `call venv\Scripts\activate.bat`. This should add a `(venv)` descriptor, showing you are inside your virtual environment.

Inside your new virtual environment, type `python`. This should open up a new `Python 3.10.2` instance inside your venv. 
> If you get a different Python version, check the ordering of Python versions on your PATH, following [this StackOverflow answer](https://stackoverflow.com/a/61691955): Python 3.10.2 should be at the top.

## Install requirements.txt:

Still inside your venv created during the previous step (but outside of Python), run `pip install -r requirements.txt`. This should install all the requirements you need for the pipeline to run.

The final things to check are to run in a command prompt, inside your venv.
- `demucs`
- `spleeter`
- `ffmpeg`

As long as these commands are recognised and don’t throw errors (missing argument ‘tracks’ is fine), everything should be set up ok.

## Build the source separated tracks

Inside the root directory of the project (i.e. `jazz-corpus-analysis\`, with folders `\data`, `\models`, `\references` etc.) and with your `venv` activated, run `python src\clean\make_dataset.py`.

> *If this throws errors e.g.* `FileNotFoundError`, *you can try hard-coding the filepaths to your* `\references` *and* `\data` *directories, e.g.* 
> ```
> python src\clean\make_dataset.py -i "path\to\jazz-corpus-analysis\references" -o "path\to\jazz-corpus-analysis\data"
> ```

For each item in references\corpus.json, the script will: 
1. download each item from YouTube;
2. run a command in FFmpeg to split the file into left and right channels; 
3. run separation on these channels + the stereo mix in Spleeter and Demucs; 
4. rename the files and reorganise the file structure; 
5. add logging information into `references\corpus.json`

You should start seeing lines like:
```
2023-05-15 15:37:24,234 - __main__ - INFO - processing item a2fe3d43cfadbeb2845aad8834b99e76, "What Is This Thing Called Love" from 1959 album Portrait in Jazz, leader Bill Evans ...
2023-05-15 15:37:25,071 - __main__ - INFO - ... found 1 valid link(s) to download from
2023-05-15 15:37:38,526 - __main__ - INFO - ... downloaded successfully from https://www.youtube.com/watch?v=8WMUZ6nixvQ
2023-05-15 15:37:38,532 - __main__ - INFO - ... skipping separation, item present locally
2023-05-15 15:37:38,532 - __main__ - INFO - ... finished processing item
```

**This command will take a while to run.** You may want to leave it on overnight, with a stable internet connection. 

> If the script is interrupted, you should be able to re-run the command without any loss of progress: any items that have already been downloaded and separated will be recognised and skipped, preventing data loss.
> 
> Conversely, to force redownload (or re-separation) if something goes wrong (e.g. an item is corrupted), you can either:
> - delete the problematic audio files from `data\raw` and `data\processed`
> - re-run the `make_dataset.py` command, passing in the flags `--force-download` and `--force-separate`

## Run onset detection algorithms

In a command prompt inside the root directory (and, again, with your `venv` activated), run:

```
python src\analyse\detect_onsets.py
```

> *Once again, to fix any* `FileNotFoundError`s, *you can try hard-coding the filepaths to the* `\references`, `\data`, `\models`, and `\reports` *directories, e.g.*
> ```
> python src\analyse\detect_onsets.py -models "path\to\jazz-corpus-analysis\models" -data "path\to\jazz-corpus-analysis\data -reports "path\to\jazz-corpus-analysis\reports -references "path\to\jazz-corpus-analysis\references"
> ```

This will then detect onsets for each item in the corpus, using the correct source-separated track. The result will be a serialised [Python pickle file](https://docs.python.org/3/library/pickle.html) that can be loaded with the `dill` or `pickle` modules.

To generate click tracks for files in the corpus, add the `--click` argument when calling `detect_onsets.py`. This will save `.wav` files inside `\reports\click_tracks` containing the audio for each performer, with clicks overlaid corresponding to the detected onset/beats positions.
> NB. the files ending with `_clicks` correspond with all the detected **onsets**; the files ending in `_beats` click only on the detected crotchet beat positions.

To detect onsets only in files with compatible manual annotations, add the `--annotated-only` argument in when calling `detect_onsets.py`.

## Generate models:

TODO
