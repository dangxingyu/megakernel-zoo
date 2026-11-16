You are starting a long autonomous run: <hours> hours, one <card> card on this machine, measured by the run's evaluator.

## Goal

Make `<model>` decode faster on this card than the general serving engines, with an engine built only for this checkpoint.
The target is at least 1.10x the fastest general engine at both batch sizes (<baseline>). The evaluator serves the engine
over HTTP: a 64-token prompt, 1,024 greedy new tokens, with one sequence and with eight.

`engine/` is the engine you start from. Load the `megakernel` skill now: it has the contract, how the evaluator scores and
what its gates check, how to log, and how to work. Your tool is `<mk>` (`mk submit`, `mk board`, `mk note`); put its
directory on your PATH.

## Your track: A

Track A is the like-for-like number: the checkpoint's number formats and data flow, the same arithmetic as the general
engines; only layout, scheduling and kernel organisation differ. Every submission of this run is track A (`engine/TRACK`
says A). Where the time goes is yours to find and measure.

## What must stay the same

1. **The model's function.** For every input the engine returns what the checkpoint defines: no quantization, pruning,
   distillation or approximation of weights or activations, and no part of the model skipped. The evaluator's gates decide.
2. **The interface and the workload.** The contract in `engine/api.py` and what the evaluator measures. One token per
   sequence per model pass.
3. **The setting.** One card, the checkpoint's stored weights as the source, no general engine or library model code in the
   timed path, every result reproducible from the submitted source.
4. **The evaluator is not yours.** Do not read, edit or work around the evaluator or its reference data.

The records of earlier runs are in the repository's `runs/` folder (every submission's source, hypothesis and result, and
each run's notes). What worked on another model is a hypothesis for this one, to be measured here.

Nobody is watching live and nobody will answer questions: decide, write down why, and continue until the time on the board
is used up.
