# Beyond object recognition and scene understanding

Image classification answers a limited question: which category best describes an image? Visual intelligence also requires structure and context.

## Relationships and scene graphs

A **scene graph** represents a scene using objects as nodes and relationships as edges:

```text
person -- riding --> horse
person -- wearing --> hat
```

This representation separates entities from their composition. A model may learn `riding` and `wearing` from different examples and potentially apply them to a novel combination such as `horse wearing hat`.

Scene graphs are attractive because they are interpretable and compositional, but defining complete and consistent relationship labels is difficult. Real scenes contain ambiguity, unobserved context, and many valid descriptions.

## From labels to descriptions and events

Visual understanding can be expressed at several levels:

- image captioning summarizes a whole image;
- dense captioning describes localized regions;
- paragraph generation connects several observations into a longer account;
- activity understanding tracks multiple actors, objects, and relations through time.

These tasks require more than recognizing isolated objects. They require spatial relationships, temporal structure, attention, language grounding, and sometimes knowledge not directly visible in the pixels.

## Main intuition

Object recognition is a useful building block, not the endpoint. Strong visual intelligence should represent **what exists, how entities relate, what is happening, and how the situation changes**.

Related notes:

- [Sequence generation and captioning](../07-sequence-models/sequence-generation-and-captioning.md)
- [Video tasks and modern systems](../10-video-understanding/video-tasks-and-modern-systems.md)

