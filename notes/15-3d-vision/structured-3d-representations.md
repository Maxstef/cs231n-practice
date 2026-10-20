# Structured 3D representations

A geometrically accurate object is not necessarily easy to understand or edit.
A point cloud or mesh may reproduce a chair's surface without identifying its
seat, legs, or the relationships that hold those parts together.

Structured representations add semantic organization on top of geometry.

## Increasing levels of structure

| Representation | What it adds | Main limitation |
| --- | --- | --- |
| Segmented geometry | A label for each geometric region | Segments may not form coherent reusable parts |
| Part set | Separate meaningful components | Does not explain relationships between parts |
| Relationship graph | Connections such as attachment or support | Graph generation and annotation are difficult |
| Hierarchy | Parts grouped into larger components | Not every relationship is naturally a tree |
| Hierarchical graph | Hierarchy plus lateral relationships | More complex to learn and manipulate |
| Program | Rules and parameters that construct the object | Programs are powerful but difficult to infer from data |

## Why structure helps

Structure can make it easier to:

- edit one part without damaging the rest;
- preserve constraints such as connected chair legs;
- generate families of objects with reusable components;
- reason about function, support, symmetry, and assembly;
- transfer parts or styles between objects.

A program representation can express natural degrees of freedom directly. For
example, a table program might expose tabletop width, leg height, and number
of legs. The same explicit control makes program supervision or discovery much
harder than learning raw coordinates.

## Geometry and structure are complementary

Element geometry describes the detailed shape of each part. Element structure
describes which parts exist and how they relate. A useful 3D model may need
both:

```text
object = element geometry + element structure
```

This resembles earlier vision tasks: segmentation identifies components, while
graphs, hierarchies, or programs capture relationships that a flat label map
does not express.

## Main intuition

> Geometry answers “where is the surface?” Structure answers “what are the
> parts, and how do they belong together?”

Adding structure improves controllability and semantic reasoning, but it also
requires stronger assumptions, richer annotations, or harder learning
problems.

## Source

- Stanford CS231n Spring 2025, Lecture 15: 3D Vision, available from the
  [course schedule](https://cs231n.stanford.edu/2025/schedule.html).
