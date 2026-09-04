# Workflow Overview

The public workflow is designed to be followed in order by a researcher or a
software agent:

1. Check the software requirements.
2. Create the `fnqa` environment.
3. Prepare the published data and any separately obtained datasets.
4. Install the models and external research components.
5. Run a smoke experiment before starting a formal experiment.

The executable commands are maintained in the root [`README.md`](../../../README.md)
so that this overview does not create a second, potentially conflicting setup
guide.

## Research Stages

The retriever selects evidence relevant to a FinQA question. The answer stage
uses the retrieved facts without changing their meaning. The narrative stage
then connects the answer to readable statements and structured bindings for
chart or interface use.

FINDER-compatible retrieval remains the baseline task. Narrative and
FinFlier-style outputs are evaluated separately and must not replace or silently
alter the baseline result.

All generated outputs belong in ignored experiment directories. Published
source files and evaluation assets remain unchanged during a run.
