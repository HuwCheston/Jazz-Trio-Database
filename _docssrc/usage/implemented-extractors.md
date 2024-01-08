# Currently implemented `Extractors`

The following table summarises the `Extractor` class instances currently importable from `.\src\features\features_utils.py`.

| Class                    | Description                                                                | Inputs                    |
|--------------------------|----------------------------------------------------------------------------|---------------------------|
| `BaseExtractor`          | Base extraction class inherited by all others                              | N/A                       |
| `IOISummaryStats`        | Various baseline summary statistics from a single array                    | `my_onsets`               |
| `RollingIOISummaryStats` | Extracts the statistics in `IOISummaryStatsExtractor` on a rolling basis   | `my_onsets`, `downbeats`  |
| `EventDensity`           | Density of inter-onset intervals across a rolling window                   | `my_onsets`, `downbeats`  |
| `IOIComplexity`          | Complexity of binned inter-onset intervals across a rolling window         | `my_onsets`, `downbeats`  |
| `BeatUpbeatRatio`        | Performance "swing", as measured by ratios of long- & short- eighth notes  | `my_onsets`, `my_beats`   |
| `TempoSlope`             | Instantaneous tempo change (in beats-per-minute) per second                | `my_beats`                |
| `Asynchrony`             | Synchronization between multiple performers at the quarter-note beat level | `my_beats`, `their_beats` |
| `ProportionalAsynchrony` | Synchronization between performers, expressed as a proportion of a measure | `my_beats`, `their_beats` |
| `PhaseCorrection`        | Modeled anticipation and adaptation between multiple performers            | `my_beats`, `their_beats` |
| `GrangerCausality`       | Measure of information sharing between two time series (i.e., performers)  | `my_beats`, `their_beats` |
| `PartialCorrelation`     | Correlation between two performers` IOIs, controlling for prior IOIs       | `my_beats`, `their_beats` |
| `CrossCorrelation`       | Correlation between two performers` IOIs, with no control                  | `my_beats`, `their_beats` |
