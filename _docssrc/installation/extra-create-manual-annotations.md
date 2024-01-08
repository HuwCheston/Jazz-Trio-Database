# Create new manual annotations
(create-annotations)=

This page reproduces the instructions given to the undergraduate research assistants who created the ground truth annotations used in the parameter optimization process (see {ref}`Run the parameter optimization process <parameter-optimization>`). It is reproduced here in case you wish to add your own ground truth annotation files and use them in the optimization process.

## Setting up
Navigate to the download page for Sonic Visualiser linked [here](https://www.sonicvisualiser.org/download.html). 

If you are using a Mac computer, click the download link for Mac on the right of the page. If you are using Windows, you will likely need to use the 64-bit installer, on the left of the page. If you experience problems with this (or are using an older system, for instance), you may need to use the 32-bit installer.

:::{dropdown} Troubleshooting install
The Windows version of Sonic Visualiser runs on most modern updates. The Mac version requires OSX 10.12 or later.

If you're using a Mac and receive a warning along the lines of `application can't be opened because Apple cannot check it for malicious software`, you will need to temporarily override your Mac security settings (don't worry, the app is perfectly safe). 
First, go to Security & Privacy. Next, click the Open Anyway button in the General pane to confirm your intent to open or install the app. If this doesn't work, you can also try dragging and dropping the download to your `applications` folder, and then right-clicking on it to run.
:::

Whatever version you are using, download the program using the primary link. Open up the .msi (Windows) or .dmg (Mac) file you downloaded and follow the installer through to the end to complete the installation. On Windows, you may need to wait a while before pressing 'Yes' when the User Account Control screen pops up.

## Open the audio
Open up Sonic Visualiser by locating it in the start menu (Windows) or Applications folder (Mac). You should see is a blank Sonic Visualiser window.

We now need to load in some audio. Open up the audio file you wish to annotated from `.\data\raw` (for beat tracking) or `.\data\processed` (for source separated audio) in the file explorer (Windows) or finder (Mac), and drag and drop it onto the Sonic Visualiser window. You should see the waveform of the audio appear.

![](https://huwcheston.github.io/PS-Supervision/_images/ex2_svwave.png)

While interesting, this doesn't help us much in locating the exact timing of individual notes. A spectrogram is a better option, so click `Layer` at the top of the window, then `Add Spectrogram` and `{audio_name}.wav: All Channels Mixed`. This will shift to the Spectrogram view, where we can see the individual frequencies.

![](https://huwcheston.github.io/PS-Supervision/_images/ex2_svspectrogram.png)

:::{note}
The spectrogram view will default to green. For some people this might not be the best option, so you can choose the colour on the side panel. You can also adjust the threshold and colour rotation of the spectrogram using the two dials either side of the colour drop-down option, which may help when identifying note onsets. I find that the `White on Black` option with the threshold at approx -60dB and rotation -30 leads to good initial results.
:::

We also need to be able to hear the recording in Sonic Visualiser. To do this, we can use the large transport buttons at the top of the window (next to the file open and save buttons), or press the spacebar.

:::{note}
If you find that the tempo of the recording is too fast, you can slow the recording down using the large dial in the bottom right of the screen (immediately to the right of the waveform).
:::

## Identify the onsets or beats

Add a new time instants layer by clicking the `Layer` drop-down menu - `Add New Time Instants Layer`. You can now click anywhere on the spectrogram to add a point, which will be shown as a line across the spectrogram. To adjust the placement of a point, click the edit button at the top (next to the pencil and cursor icons) and drag the point to where you want it. You can also delete a point by clicking the eraser (next to the pencil and compass) then clicking on the point again.

![](https://huwcheston.github.io/PS-Supervision/_images/ex2_svtimeinstants.png)

:::{note}
By default, Sonic Visualiser renders time instants as white lines. This can be confusing, as the playback line is also white. You can change the colour of the instants on the right-hand menu.
:::

Once you've tried creating some time instant values, you'll need to place them at the start of each note or beat in the audio file. Use a combination of your ears and the visual spectrogram to guide you here. The following instructions were also given to the research assistants:

:::{dropdown} Detection instructions

- In general, if an onset is ambiguous, fall on the side of not annotating it.
- For piano, treat grace notes/acciaccaturas as separate onsets (i.e. annotate each), but treat chords as a single onset (i.e. only annotate once)
- For bass, most of the performance should be monophonic, but treat double stops the same way as chords
  - In cases where the algorithm cannot separate, audio will be silent
  - In cases of noise/silence/lacking a clear attack, do not annotate onsets
  - Periods of very slight audio amidst silence will also be ignored
- For drums, we’re only interested in annotating the cymbals (ride/crash/hi-hat), we’re not interested in detecting onsets in the drums themselves (i.e. snare and hi-hat) because of bleed issues.
- For overall quarter-note beats, the procedure should be to listen through manually first (at the actual tempo)
  - Then, go back through at a slower tempo and manually adjust
  - I find that my beats are usually quite late in comparison to the recording, so I usually end up adjusting every single beat to compensate.
- In cases of bleed (i.e. drums enter the piano recording), stick to annotating the target instrument only. Most sections of bleed are usually pretty short. 
:::


:::{note}
You can zoom into the spectrogram by using the scroll wheel on your mouse, the page up key on your keyboard, or the horizontal wheel on the bottom-right of the spectrogram display.
:::

## Export your data

Once you've labelled all the note onsets or beats for the recording, it's time to get the data out of Sonic Visualiser. Make sure your time instants layer is active by selecting it in the panel on the right of the screen: it will likely be layer 5. Then, go to `File - Export Annotation Layer`.

For this project, annotation files should be saved in the `.\references\manual_annotations` directory. The filename should be identical to the audio, without any of the `lchan` or `rchan` tags. See the annotation files already in this directory for examples.

:::{note}
If you find that you get an error when trying to save, you may need to put .txt manually at the end of your filename - e.g. Timings.txt
:::

## Edit the corpus file

The final stage to allow your manual annotation files to be read when running `.\src\detect\optimize_detection_parameters.py` is to edit the corresponding corpus spreadsheet file (`.xlsx`) contained in `.\references`. 

Find the track that you've annotated in this spreadsheet (check the final characters in the `recording_id_for_lbz` column and the corresponding characters in the audio filename) and add a `Y` to the column `has_annotations`. And that's it!
