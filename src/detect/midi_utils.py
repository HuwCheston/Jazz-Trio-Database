#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility classes, functions, and variables when working with MIDI files."""

import os
from itertools import groupby
from typing import Generator
from math import isclose
from os import makedirs

import numpy as np
import pretty_midi
import librosa
from piano_transcription_inference import PianoTranscription, load_audio, sample_rate
from piano_transcription_inference.utilities import write_events_to_midi

from src import utils
from src.detect.onset_utils import FREQUENCY_BANDS, OnsetMaker, bandpass_filter
from src.clean.clean_utils import HidePrints


__all__ = ['Note', 'Interval', 'MelodyMaker', 'MIDIMaker']


class Note(pretty_midi.Note):
    """Overrides `pretty_midi.Note` with a few additional properties"""

    def __init__(self, note):
        super().__init__(**note.__dict__)
        self.ioi = super().duration
        self.note = ''.join(i for i in pretty_midi.note_number_to_name(self.pitch) if not i.isdigit())
        self.octave = int(''.join(i for i in pretty_midi.note_number_to_name(self.pitch) if i.isdigit()))
        self.pitch_class = pretty_midi.note_name_to_number(f'{self.note}0') - 12

    def __repr__(self):
        return 'Note(start={:f}, end={:f}, ioi={:f}, pitch={}, velocity={}, note={}, octave={}, pitch_class={})'.format(
            self.start, self.end, self.ioi, self.pitch, self.velocity, self.note, self.octave, self.pitch_class
        )


class Interval:
    """Used to extract info from two `Note` objects occurring separately"""

    def __init__(self, firstnote: Note, secondnote: Note):
        self.start = firstnote.start
        self.interval = firstnote.pitch - secondnote.pitch
        self.interval_class = firstnote.pitch_class - secondnote.pitch_class
        self.ioi = secondnote.start - firstnote.start
        self.velocity_change = firstnote.velocity - secondnote.velocity

    def __repr__(self):
        return 'Interval(interval={}, interval_class={}, ioi={:f}, velocity_change={})'.format(
            self.interval, self.interval_class, self.ioi, self.velocity_change
        )


class MelodyMaker:
    SHORTEST_RHYTHM = 1 / 64
    TIME_THRESH = 0.01    # notes shorted than 10 milliseconds will be removed
    NOTE_LB, NOTE_UB = (int(pretty_midi.hz_to_note_number(FREQUENCY_BANDS['piano'][fm])) for fm in ['fmin', 'fmax'])
    MIDDLE_C = 60

    def __init__(self, midi_fpath: str, om: OnsetMaker):
        # Attributes taken directly from the `OnsetMaker` class
        self.beats = om.summary_dict['beats']
        self.downbeats = om.ons['downbeats_manual']
        self.tempo = om.tempo
        self.quarter_note = 60 / self.tempo    # duration of a quarter note in seconds
        self.time_signature = om.item['time_signature']
        self.tempo_thresh = (self.quarter_note * self.time_signature) / 1 / 64  # len(one bar) / (musical value)
        self.midi = self.load_midi(midi_fpath)

    def load_midi(self, midi_fpath) -> pretty_midi.Instrument:
        # TODO: check we don't have more than one instrument here
        return pretty_midi.PrettyMIDI(midi_fpath, initial_tempo=self.tempo).instruments[0]

    def _remove_iois_below_threshold(self, notes: list[pretty_midi.Note]) -> list[pretty_midi.Note]:
        ioi = lambda n: n.end - n.start
        return sorted(
            [n for n in notes if not any([
                ioi(n) < self.tempo_thresh,
                ioi(n) < self.TIME_THRESH
            ])],
            key=lambda n: n.start
        )

    def _remove_pitches_below_threshold(self, notes: list[pretty_midi.Note]) -> list[pretty_midi.Note]:
        return sorted(
            [n for n in notes if not any([
                n.pitch <= self.NOTE_LB,
                n.pitch >= self.NOTE_UB,
                n.pitch <= self.MIDDLE_C
            ])],
            key=lambda n: n.start
        )

    @staticmethod
    def _quantize_notes_in_beat(
            beat1: float,
            beat2: float,
            notes: list[pretty_midi.Note],
            num_ticks: int = 8
    ) -> Generator[pretty_midi.Note, None, None]:
        ticks = np.linspace(beat1, beat2, num_ticks)
        closest = lambda n: ticks[np.argmin(np.abs(n.start - ticks))]
        yield from (pretty_midi.Note(start=closest(n), end=n.end, pitch=n.pitch, velocity=n.velocity) for n in notes)

    @staticmethod
    def _extract_highest_note(
            notes: list[pretty_midi.Note]
    ) -> Generator[Note, None, None]:
        for _, vals in groupby(sorted(notes, key=lambda x: x.start), lambda x: x.start):
            yield Note(max(vals, key=lambda n: n.pitch))

    def extract_melody(self):
        notes = self._remove_pitches_below_threshold(self._remove_iois_below_threshold(self.midi.notes))
        for beat1, beat2 in zip(self.beats, self.beats[1:]):
            quantized_notes = self._quantize_notes_in_beat(beat1, beat2, [n for n in notes if beat1 <= n.start < beat2])
            yield from self._extract_highest_note(quantized_notes)

    def extract_intervals(
            self,
            melody_notes: list[Note]
    ) -> Generator[Interval, None, None]:
        if melody_notes is None:
            # Extract the melody and convert it to a list: we can't subscript a generator object
            melody_notes = list(self.extract_melody())
        # Yield an interval object from consecutive notes in the extracted melody
        yield from (Interval(fn, sn) for fn, sn in zip(melody_notes, melody_notes[1:]))

    def chunk_melody(
            self,
            notes: list[Note | Interval] = None,
            chunk_measures: int = 4,
            overlapping_chunks: bool = True,
    ) -> list[tuple[Note]]:
        if notes is None:
            notes = self.extract_melody()
        if overlapping_chunks:
            z = zip(self.downbeats, self.downbeats[chunk_measures:])
        else:
            z = zip(self.downbeats[::chunk_measures], self.downbeats[chunk_measures::chunk_measures])
        # TODO: check if this can return Interval objects as well as Note
        return [tuple(m for m in notes if db1 <= m.start < db2) for db1, db2 in z]


class MIDIMaker:
    """Create MIDI for a single instrument (defaults to piano)"""
    INSTR = 'piano'

    def __init__(self, item: dict, **kwargs):
        self.item = item
        desired_channel = self.item['channel_overrides'][self.INSTR] if (
            self.INSTR in self.item['channel_overrides'].keys()
        ) else None
        # Construct filepaths for the audio
        self.data_dir = kwargs.get('data_dir', f'{utils.get_project_root()}/data')
        proc_fpath = os.path.join(
            utils.get_project_root(),
            f'{self.data_dir}/processed/mvsep_audio',
            utils.construct_audio_fpath_with_channel_overrides(
                self.item['fname'], instr=self.INSTR, channel=desired_channel
            )
        )
        # Load in the source separated audio
        self.proc_audio, _ = load_audio(
            proc_fpath,
            sr=sample_rate,
            mono=True,
            res_type='soxr_vhq',
            dtype=np.float64,
            offset=0,
            duration=None
        )
        self.midi = None

    @staticmethod
    def pitch_correction(audio: np.array) -> np.array:
        """Pitch-shift given audio to A=440 Hz"""
        original_tuning = librosa.estimate_tuning(audio, sr=sample_rate)
        # Shift by -semitones to return to A=440
        y_shifted = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=-original_tuning)
        shifted_tuning = librosa.estimate_tuning(y_shifted, sr=sample_rate)
        # Check that we've actually shifted the audio closer to A=440
        assert isclose(shifted_tuning, 0, abs_tol=0.1)
        # If the original audio was somehow closer to A=440 than the shifted version, use that
        return y_shifted if abs(shifted_tuning) < original_tuning else audio

    def preprocess_audio(self, filter_audio: bool = False, pitch_correction: bool = True) -> np.array:
        """Preprocess audio by filtering and/or applying pitch correction"""
        if pitch_correction:
            self.proc_audio = self.pitch_correction(self.proc_audio)
        if filter_audio:
            self.proc_audio = bandpass_filter(
                self.proc_audio,
                lowcut=FREQUENCY_BANDS[self.INSTR]['fmin'],
                # Use the whole upper frequency range
                highcut=(sample_rate / 2) - 1,
                pad_len=0,
                sample_rate=sample_rate
            )
        return self.proc_audio

    def convert_to_midi(self) -> dict:
        """Convert processed audio into MIDI"""
        with HidePrints():
            transcriptor = PianoTranscription(device='cuda', checkpoint_path=None)
            self.midi = transcriptor.transcribe(self.proc_audio, midi_path=None)
        return self.midi

    def finalize_output(self, dirpath: str = None, filename: str = 'piano_midi.mid') -> None:
        """Finalize output by saving processed MIDI into the correct directory"""
        # Make the folder to save the annotations in
        if dirpath is None:
            dirpath = self.data_dir + f'/cambridge-jazz-trio-database-v02/{self.item["fname"]}/'
        makedirs(dirpath, exist_ok=True)
        # Format the filename correctly
        if not filename.endswith('.mid'):
            filename = filename.split('.')[0] + '.mid'
        # Write the midi output into the correct directory
        write_events_to_midi(
            start_time=0,
            note_events=self.midi['est_note_events'],
            pedal_events=self.midi['est_pedal_events'],
            midi_path=f'{dirpath}/{filename}'
        )


if __name__ == '__main__':
    corpus_fname = 'corpus_updated'
    corpus = utils.CorpusMaker.from_excel(corpus_fname)
    for track in corpus.tracks:
        # Create the MIDIMaker instance for the given track
        mm = MIDIMaker(track)
        # Preprocess the audio by pitch correcting, but not filtering
        mm.preprocess_audio(filter_audio=False, pitch_correction=True)
        # Convert to MIDI
        mm.convert_to_midi()
        # Clean up the results
        mm.finalize_output()
