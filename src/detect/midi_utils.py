#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility classes, functions, and variables when working with MIDI files."""

from itertools import groupby
from typing import Generator

import numpy as np
import pretty_midi

from src.detect.detect_utils import FREQUENCY_BANDS, OnsetMaker


__all__ = ['Note', 'MelodyMaker']


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

    def chunk_melody(
            self,
            melody_notes: list[Note],
            chunk_measures: int = 4,
            overlapping_chunks: bool = True
    ) -> Generator[tuple[Note], None, None]:
        if overlapping_chunks:
            z = zip(self.downbeats, self.downbeats[chunk_measures:])
        else:
            z = zip(self.downbeats[::chunk_measures], self.downbeats[chunk_measures::chunk_measures])
        chunks = []
        for db1, db2 in z:
            chunks.append(tuple(m for m in melody_notes if db1 <= m.start < db2))
        return chunks
