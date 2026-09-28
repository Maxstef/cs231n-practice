# Video tasks and modern systems

“Video understanding” covers several tasks with different output structures.

| Task | Typical output |
| --- | --- |
| Trimmed action classification | One class for a short clip |
| Temporal action localization | Action labels and time intervals |
| Spatiotemporal detection | Action labels, times, and spatial boxes or tracks |
| Video captioning | A generated text description |
| Video question answering | A text answer grounded in the video |

A trimmed classification benchmark assumes the relevant action fills most of
the clip. An untrimmed video may contain long irrelevant periods, multiple
actions, or several people acting simultaneously. Spatiotemporal detection must
therefore answer not only *what*, but also *when* and *where*.

## Modern multimodal systems

A modern video-language system commonly combines:

1. a visual encoder that converts sampled frames or clips into tokens;
2. temporal processing or token compression;
3. a projection that aligns visual tokens with a language model;
4. a language model that generates captions or answers.

Long videos remain challenging because dense frame sampling creates too many
tokens, while sparse sampling can miss short events. The same sampling,
temporal coverage, and computational trade-offs from video classification still
apply.

## Evaluation follows the task

Classification may use accuracy, temporal localization needs interval overlap,
and spatiotemporal detection also needs spatial overlap. Generated language is
harder to evaluate because several different descriptions can all be valid.

## Source

- Stanford CS231n Spring 2025, Lecture 10: Video Understanding, available from
  the [course schedule](https://cs231n.stanford.edu/2025/schedule.html).
