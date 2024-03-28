#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generates the final dataset of separated recordings from the items in the corpus"""

import logging
from pathlib import Path
from time import time

import click
from dotenv import find_dotenv, load_dotenv

from src import utils
from src.clean.clean_utils import ItemMaker


@click.command()
@click.option("-corpus", "corpus_filename", type=str, default="corpus_chronology", help='Name of the corpus to use')
@click.option("-force-download", "force_download", is_flag=True, default=False, help='Force download from YouTube')
@click.option("-force-separation", "force_separation", is_flag=True, default=False, help='Force source separation')
@click.option("-no-spleeter", "disable_spleeter", is_flag=True, default=True, help='Disable spleeter for separation')
@click.option("-no-demucs", "disable_demucs", is_flag=True, default=True, help='Disable demucs for separation')
def main(
        corpus_filename: str,
        force_download: bool,
        force_separation: bool,
        disable_spleeter: bool,
        disable_demucs: bool
) -> None:
    """Runs processing scripts to turn corpus from (./references) into audio, ready to be analyzed"""
    # Set the logger
    logger = logging.getLogger(__name__)
    # Start the timer
    start = time()
    # Open the corpus Excel file using our custom class
    corpus = utils.CorpusMaker.from_excel(corpus_filename)
    # Iterate through each item in the corpus and make it
    for corpus_item in corpus.tracks:
        im = ItemMaker(
            item=corpus_item,
            logger=logger,
            get_lr_audio=True,
            force_reseparation=force_separation,
            force_redownload=force_download,
            use_spleeter=not disable_spleeter,
            use_demucs=not disable_demucs
        )
        im.get_item()
        try:
            im.separate_audio()
        except:
            logger.warning(f'failed to separate {corpus_item["track_name"]}, skipping')
        else:
            im.finalize_output()
    # Log the total completion time
    logger.info(f"dataset {corpus_filename} made in {round(time() - start)} secs !")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
