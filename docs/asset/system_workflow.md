# System Workflow

FQAN extends financial question answering with evidence retrieval, answer
generation, narrative construction, and structured data binding. FINDER
semantics remain the basis of retrieval, while FinFlier-style output is an
additive presentation layer.

```text
FinQA data and question
  -> evidence retrieval
  -> matched financial facts
  -> answer generation
  -> narrative and structured binding
  -> evaluation
```

The root [`README.md`](../../README.md) provides the executable setup and
experiment sequence. See [`workflow/README.md`](workflow/README.md) for the role
of each stage and
[`../instructions/narrative/evaluation.md`](../instructions/narrative/evaluation.md)
for the narrative scoring rules.
