# Multimodal and audio-visual video understanding

A video is not only a sequence of images. It may also contain speech, music,
environmental sounds, subtitles, and other text. **Multimodal video
understanding** combines two or more of these information sources.

## Why audio helps

Audio can provide evidence that is weak or absent in sampled frames:

- speech identifies what is being discussed;
- impact, engine, or instrument sounds identify events;
- timing in the waveform helps locate when an event happens;
- off-screen sounds reveal events outside the camera view.

Vision is useful when audio is noisy or ambiguous. It can identify the visible
sound source, distinguish speakers, and separate simultaneous sounds.

## Representing audio

Raw audio is a one-dimensional waveform over time. Models often transform it
into a **spectrogram**, whose axes represent time and frequency and whose values
represent signal strength. A spectrogram can be divided into patches and
encoded much like an image, but its axes have different meanings.

```text
video frames -> visual encoder -> visual tokens ---\
                                                   +-> multimodal fusion -> output
audio signal -> spectrogram -> audio encoder ------/
```

The two streams have different sampling rates, so aligning visual frames with
audio time intervals is important.

## Ways to combine modalities

- **Early fusion:** combine aligned low-level or token representations.
- **Intermediate fusion:** separate encoders exchange information through
  attention or shared bottleneck tokens.
- **Late fusion:** combine independently produced predictions.

Intermediate fusion lets the model learn relationships such as which visible
person produced a sound. Restricting cross-modal exchange to bottleneck tokens
can reduce the cost compared with allowing every audio token to attend to every
video token.

## Example tasks

- audio-visual action recognition;
- visually guided sound-source separation;
- active-speaker detection;
- audio-visual event localization;
- video captioning and question answering;
- masked pretraining that reconstructs missing audio or visual content.

Multimodal input is not automatically better. A model may over-rely on an easy
but misleading cue—for example, background music correlated with a class.
Training should therefore consider missing, noisy, or contradictory modalities.

## Modern video-language systems

Modern systems may map visual and audio tokens into the representation space of
a language model. The language model can then generate descriptions or answer
questions, while the encoders supply perceptual evidence. Long clips remain
difficult because video and audio together create many tokens, making sampling,
compression, and temporal alignment essential.

## Related notes

- [Video tasks and modern systems](video-tasks-and-modern-systems.md)
- [Long-range video modeling](long-range-video-modeling.md)

## Source

- Stanford CS231n Spring 2025, Lecture 10: Video Understanding, available from
  the [course schedule](https://cs231n.stanford.edu/2025/schedule.html).
