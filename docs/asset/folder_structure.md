# Repository Structure

FQAN separates runnable programs, redistributable data, research documentation,
and model installation helpers. The root [`README.md`](../../README.md) remains
the canonical starting point.

## Structure Guides

| Guide | Purpose |
| --- | --- |
| [`workspace/root_tree.md`](workspace/root_tree.md) | Compact view of the public repository tree |
| [`workspace/directory_roles.md`](workspace/directory_roles.md) | Purpose and expected contents of each public directory |
| [`system_workflow.md`](system_workflow.md) | Research pipeline from data preparation to evaluation |

Generated experiment outputs and downloaded model files are kept outside the
published tree. Their locations are created by the documented commands when
needed.
