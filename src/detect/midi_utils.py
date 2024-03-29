#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility classes, functions, and variables when working with MIDI files."""

from itertools import groupby
from typing import Generator

import numpy as np
import pretty_midi
import librosa
from piano_transcription_inference import PianoTranscription, load_audio

from src import utils
from src.detect.detect_utils import FREQUENCY_BANDS, OnsetMaker


__all__ = ['Note', 'Interval', 'MelodyMaker']


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

    def extract_intervals(self) -> Generator[Interval, None, None]:
        # Extract the melody and convert it to a list: we can't subscript a generator object
        melody = list(self.extract_melody())
        # Yield an interval object from consecutive notes in the extracted melody
        yield from (Interval(fn, sn) for fn, sn in zip(melody, melody[1:]))

    def chunk_melody(
            self,
            melody_notes: list[Note | Interval],
            chunk_measures: int = 4,
            overlapping_chunks: bool = True,
    ) -> list[tuple[Note]]:
        if overlapping_chunks:
            z = zip(self.downbeats, self.downbeats[chunk_measures:])
        else:
            z = zip(self.downbeats[::chunk_measures], self.downbeats[chunk_measures::chunk_measures])
        # TODO: check if this can return Interval objects as well as Note
        return [tuple(m for m in melody_notes if db1 <= m.start < db2) for db1, db2 in z]


class MIDIMaker:
    def __init__(self, separated_audio_fpath: str):
        # TODO: we may need to use sr=16000 as in the documentation
        self.audio, _ = load_audio(separated_audio_fpath, sr=utils.SAMPLE_RATE, mono=False)

    @staticmethod
    def _pad_audio():
        # TODO: make this work
        raw, stereo = np.array([]), np.array([])
        pad = np.zeros((2, raw.shape[0] - stereo.shape[1]))
        return np.concatenate([pad, stereo], axis=1)

    @staticmethod
    def _pitch_shift_audio():
        # TODO: make this work
        audio = np.array([])
        tuning = librosa.estimate_tuning(audio, sr=utils.SAMPLE_RATE)
        semitones = tuning / 100
        return librosa.effects.pitch_shift(audio, sr=utils.SAMPLE_RATE, n_steps=-semitones)

    def preprocess_audio(self):
        pass

    def audio_to_midi(self):
        # TODO: make this work
        transcriptor = PianoTranscription(device='cuda', checkpoint_path=None)
        transcribed_dict = transcriptor.transcribe(self.audio, None)
