# Directory Roles

| Directory | Public role |
| --- | --- |
| `src/` | FQAN programs, stable commands, configuration, and tests |
| `src/dist/` | Command-line entry points used by the reproduction workflow |
| `src/config/` | Portable experiment and runtime settings |
| `src/narrative/` | Narrative construction, binding, and evaluation support |
| `src/FINDER/` | Pinned upstream repository and installation guidance |
| `src/FinFlier/` | Pinned upstream repository and installation guidance |
| `src/Experiment/` | Documented local destination for generated results |
| `data/` | Redistributable data, source notes, and preparation utilities |
| `questionnaire/` | Research instruments without collected responses |
| `docs/` | Public workflow, structure, and evaluation documentation |
| `utils/` | Model manifest and model installation guidance |

The repository does not redistribute model weights or third-party source whose
license does not permit republication. The installation guides identify where
those external components are required.
