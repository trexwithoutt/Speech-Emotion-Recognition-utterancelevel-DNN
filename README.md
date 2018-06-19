# Global Features in SER on IEMOCAP

## Original Aim

Emotions are fundamental for humans, impacting perception and everyday activities
such as communication, learning and decision-making. Recently, SER has been drawing increasing attention. Speech emotion recognition is a very challenging task, since machines do not understand human emotion states, of which extracting effective emotional features is an open question.

In this project, we will explore the several contributions in this area, and find out the significant algorithm doing the emotion detection from speech. We are particularly interested to compare the human-engineered features to the raw representations in human speech. 

## Corpus 

The data used for this project is Interactive Emotional Dyadic Motion Capture (IEMOCAP) database which comes from Signal Analysis and Interpretation Laboratory at the University of South California. It contains 12 hours of audiovisual data, including video, speech, motion capture of face, text transcriptions[3]. The recordings consist of professional actors improvising and scripting a series of semantically neutral utterances spanning ten distinct emotional categories. There were 5 female speakers and 5 male speakers. The number and count ratio of utterances that belong to each emotion category is shown in table.


|        | Ang   | Hap   | Exc   | Neu   | Sad   |
|--------|-------|-------|-------|-------|-------|
| Counts | 1103  | 595   | 1041  | 1708  | 1084  |
| Ratio  | 19.9% | 10.8% | 18.8% | 30.9% | 19.6% |




## Background

	- Support Vector Machine

	- K Nearest Neighbors

	- Deep Neural Networks

	- Extreme Learning Machine

## Introduction

| Low Level Descriptors            | MFCC, Mel-filterbank, formant, HNR, jitter, shimmer, etc. |
|----------------------------------|-----------------------------------------------------------|
| High-level Statistical Functions | mean, variance, max, min, median, etc.                    |

## Algorithms

### Segment-level Features Extraction

- MFCC

- Harmonic to Noise Ratio

- Pitch Period

### Utterance-level Features Extraction

- Maximal

- Minimal

- Average

- Percentage above certain threshold

### Extreme Learning Machine

### Other Models

## References

- Fred G. Martin Robotics **Explorations: A Hands-On Introduction to Engineering**. New Jersey: Prentice Hall.

- Seyedmahdad Mirsamadi, Emad Barsoum, Cha Zhang 2017. **Automatic Speech Emotion Recognition Using Recurrent Neural Networks with Local Attention**.Washington: Microsoft Research, One Microsoft Way, 2017

- C. Busso, M. Bulut, C. Lee, A. Kazemzadeh, E. Mower, S. Kim, J. Chang, S. Lee, and S. Narayanan, **IEMOCAP: Interactive emotional dyadic motion capture database**, Journal of Language Resources and Evaluation, vol. 42, no. 4, pp. 335-359, December 2008.

- Kim, P.Georgiou, S.Lee, S.Narayanan. **Real-time emotion detection system using speech: Multi-modal fusion of different timescale features**, Proceedings of IEEE Multimedia Signal Processing Workshop, Chania, Greece, 2007

- E. Mower, M. J. Mataric, and S. Narayanan, **A framework for automatic human emotion classification using emotion profiles, Audio, Speech, and Language Processing**, IEEE Transactions on, vol. 19, no. 5, pp. 10571070, 2011.

- G.-B. Huang, Q.-Y. Zhu, and C.-K. Siew, **Extreme learning machine: theory and applications, Neurocomputing**, vol. 70, no. 1, pp. 489501, 2006.

- . Kim and E. Mower Provost, **Emotion classification via utterance-level dynamics: A pattern-based approach to characterizing affective expressions**, in Proceedings of IEEE ICASSP 2013. IEEE, 2013.

- Schuller, G. Rigoll, and M. Lang, **Hidden markov model-based speech emotion recognition**, in Proceedings of IEEE ICASSP 2003, vol. 2. IEEE, 2003, pp. II1.