#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Classes used for plotting ensemble coordination, e.g. phase correction, simulations"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

from src import utils
import src.visualise.visualise_utils as vutils

FOLDER_PATH = 'coordination_plots'


class _TriangleAxis:
    """This class plots a single axes, showing the measured couplings between each musician in a trio.

    The direction of arrows indicates the influence and influencer instruments, namely the tendency of one performer
    to follow (and adapt to) the indicated instrument: the thickness and colour of the arrows indicate the strength
    of this coupling. This graph is similar to Figure 3b. in Jacoby, Polak, & London (2021).

    Args:
        grp (pd.DataFrame): the dataframe to plot, should contain information from one ensemble only.
        ax (plt.Axes): the axes object to plot onto.

    References:
        Jacoby, N., Polak, R., & London, J. (2021). Extreme precision in rhythmic interaction is enabled by
        role-optimized sensorimotor coupling: Analysis and modelling of West African drum ensemble music. Philosophical
        Transactions of the Royal Society B: Biological Sciences, 376(1835), 20200331.


    """
    img_loc = fr'{utils.get_project_root()}\references\images\musicians'
    starting_zoom = 1.2

    def __init__(self, grp: pd.DataFrame, ax: plt.Axes):
        self.grp = grp
        self.ax = ax
        self.ax.axis('off')
        self.ax.set_aspect('equal')

    def create_plot(
            self
    ) -> plt.Axes:
        """Called from outside the class to generate the required plot elements, show them, and save"""

        self._add_musicians_images()
        self._add_center_text()
        self._create_plot()
        return self.ax

    def _add_center_text(self):
        """Adds in text to the center of the plot"""
        # Add in the name of the musician
        txt = str(self.grp['pianist'].iloc[0]).replace(' ', '\n')
        self.ax.text(0.5, 0.5, txt, ha='center', va='center', fontsize=vutils.FONTSIZE * 2)

    def _create_plot(
            self, arrow_mod: float = 25
    ) -> None:
        """Creates the plot arrows and annotations, according to the modelled coupling responses."""
        # The starting coordinate for each arrow, i.e. the arrow tail
        start_coords = [
            [(0.15, 0.325), (0.85, 0.325)],
            [(0.35, 0.95), (0.725, 0.175)],
            [(0.65, 0.95), (0.275, 0.075), ],
        ]
        # The end coordinate for each arrow, i.e. the arrow head
        end_coords = [
            [(0.35, 0.75), (0.65, 0.75)],
            [(0.05, 0.325), (0.275, 0.175)],
            [(0.95, 0.325), (0.725, 0.075), ],
        ]
        # These values are the amount of rotation we require for each arrow
        rotation = [(62.5, -62.5), (62.5, 0), (-62.5, 0)]
        # Iterate over each influence instrument, their respective arrows, and the color they're associated with
        for influencer, start_coord, end_coord, col, rot12 in zip(
                utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys(), start_coords, end_coords, vutils.RGB, rotation
        ):
            # Iterate over each instrument they influence, and each individual arrow
            for num, (influenced, (x, y), (x2, y2)) in enumerate(zip(
                    [i for i in utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys() if i != influencer], start_coord, end_coord
            )):
                # Get our coupling coefficient
                c, q1, q2 = self._get_coupling_coefficient(influenced, influencer)
                # Add in the arrow
                self.ax.annotate(
                    '', xy=(x, y), xycoords=self.ax.transAxes,
                    xytext=(x2, y2), textcoords=self.ax.transAxes,
                    arrowprops=dict(
                        width=c * arrow_mod, edgecolor=col, lw=1.5, facecolor=col, headwidth=20
                    )
                )
                # Add the coupling coefficient text in
                txt = f'{round(abs(c), 2)} [{round(abs(q1), 2)}–{round(abs(q2), 2)}]'
                self._add_coupling_coefficient_text(txt, x, x2, y, y2, rotation=rot12[num])

    def _get_coupling_coefficient(
            self, influenced: str, influencer: str
    ) -> tuple[float, float, float]:
        """Helper function to get the coupling coefficient between two instruments, the influencer and influenced."""
        # Get the coupling coefficients
        g = self.grp[self.grp['instrument'] == influenced][f'coupling_{influencer}'].dropna()
        # Bootstrap over all coupling coefficients to get confidence intervals
        boot_mean = [g.sample(frac=1, replace=True).mean() for _ in range(vutils.N_BOOT)]
        # Return the true mean and bootstrapped 95% confidence intervals
        return g.mean(), np.quantile(boot_mean, 0.05), np.quantile(boot_mean, 0.95)

    def _add_coupling_coefficient_text(
            self, constant, x, x2, y, y2, mod: float = 0.03, rotation: float = 0
    ) -> None:
        """Adds coupling coefficient"""

        # Get the default annotation position, the midpoint of our arrow
        x_pos = (x + x2) / 2
        y_pos = (y + y2) / 2
        # Bottom of plot
        if y_pos < 0.3:
            y_pos += (mod * 1.5)
        # Top left of plot
        elif x_pos < 0.5:
            y_pos += mod
            x_pos -= (mod * 1.1)
        # Right of plot
        elif x_pos > 0.5:
            y_pos += mod
            x_pos += (mod * 1.1)
        # Add in the text using the x and y position
        self.ax.text(
            x_pos, y_pos, constant, ha='center', va='center', fontsize=vutils.FONTSIZE,
            rotation=rotation, rotation_mode='anchor'
        )

    def _add_musicians_images(
            self
    ) -> None:
        """Adds images corresponding to each performer in the trio"""
        # Set initial attributes for image plotting
        zoom, boxstyle = self.starting_zoom, 'square'
        img = None
        # Iterate through the position for each picture, the amount of zoom,
        # the name of the performer, and the colour of the box around the picture
        for (x, y), zoom, txt, col in zip(
                [(0.5, 0.875), (0.125, 0.125), (0.867, 0.133), ],
                [.68, .6375, .75, ],
                utils.INSTRUMENTS_TO_PERFORMER_ROLES.keys(),
                vutils.RGB
        ):
            # Get the filepath from the performer's name
            mus = str(self.grp[self.grp['instrument'] == txt]['performer'].iloc[0]).replace(' ', '_').lower()
            # Try and get the image corresponding to the performers name
            try:
                img = plt.imread(fr'{self.img_loc}\{mus}.png')
                boxstyle, zoom = 'sawtooth', zoom + 0.25
            # If we don't have the image, use the default image corresponding to that instrument
            except FileNotFoundError:
                img = plt.imread(fr'{self.img_loc}\_{txt}.png')
            # Add the image into the graph, with the correct parameters and properties
            finally:
                img = mpl.offsetbox.OffsetImage(img, zoom=zoom)
                ab = mpl.offsetbox.AnnotationBbox(
                    img, (x, y), xycoords='data', bboxprops=dict(edgecolor=col, lw=2, boxstyle=boxstyle)
                )
                self.ax.add_artist(ab)
            # Add the text in, adjacent to the image
            self.ax.text(
                x, y + 0.15 if y < 0.5 else y - 0.175, txt.title(),
                ha='center', va='center', color=col, fontsize=vutils.FONTSIZE,
            )


class TrianglePlotChronology(vutils.BasePlot):
    """Creates a triangle plot for each trio combination in the chronology corpus"""

    def __init__(self, df: pd.DataFrame, **kwargs):
        self.corpus_title = kwargs.get('corpus_title', 'corpus')
        super().__init__(figure_title=fr'{FOLDER_PATH}\triangleplot_trios_{self.corpus_title}', **kwargs)
        self.nobs_cutoff = kwargs.get('nobs_cutoff', 30)
        self.df = df[df['nobs'] < self.nobs_cutoff]
        self.fig, self.ax = plt.subplots(nrows=5, ncols=2, figsize=(vutils.WIDTH, vutils.WIDTH * 2))

    def _create_plot(self):
        """Create a `_TriangleAxis` object for each trio"""
        for a, (idx, grp) in zip(self.ax.flatten(), self.df.groupby('pianist')):
            self.g = _TriangleAxis(grp, a).create_plot()

    def _format_fig(self):
        """Format figure-level parameters"""
        self.fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, hspace=0.01, wspace=0.01)


if __name__ == '__main__':
    extracted = utils.unserialise_object(
        fr'{utils.get_project_root()}\models\extracted_features_corpus_chronology', use_pickle=True
    )
    combined = utils.combine_features(extracted, 'metadata', 'phase_correction')
    df = combined[combined['phase_correction_order'] == 1]

