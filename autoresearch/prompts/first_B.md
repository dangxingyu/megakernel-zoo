You are starting a long autonomous run: <hours> hours, one <card> card on this machine, measured by the run's evaluator.

## Goal

Find what is specific to **this model** (`<model>` on one <card>) that a general serving engine cannot use, and show what
it is worth (<baseline>). `engine/` is a working track-A engine: it computes exactly what the general engines compute, in
their number formats. We are looking for mechanisms that depend on this checkpoint: its weights, its architecture, how it
behaves. Time per token is the score, but a new model-specific mechanism with a clear measurement behind it is worth more
than a few percent from tuning.

Load the `megakernel` skill now: it has the contract, how the evaluator scores and what its gates check, how to log, and
how to work. Your tool is `<mk>` (`mk submit`, `mk board`, `mk note`); put its directory on your PATH.

## What you may do

Anything that leaves every output bit-identical to the track-A build of the same source. You decide what is worth studying
and measuring in the checkpoint and how. You may change how the weights are stored and encoded, remove redundancy without
loss, and precompute at load time.

## What must stay the same

1. **The model's function**, bit for bit: a submission is labelled track B only after you have compared its logits with the
   same source built for track A on many positions, and a storage change on all real weights.
2. **Every change is measured against the engine you started from.** A submission changes one thing and says what it is.
   When you improve the kernel itself (something that would also work with the checkpoint's formats), submit that on its
   own first as track A, so that the two kinds of gain are never mixed.
3. **The interface and the workload.** The contract in `engine/api.py` and what the evaluator measures. One token per
   sequence per model pass.
4. **The setting.** One card, the checkpoint's stored weights as the source, no general engine or library model code in the
   timed path, every result reproducible from the submitted source.
5. **The evaluator is not yours.** Do not read, edit or work around the evaluator or its reference data.

The records of earlier runs are in the repository's `runs/` folder. What worked on another model is a hypothesis for this
one, to be measured and verified here.

Nobody is watching live and nobody will answer questions: decide, write down why, and continue until the time on the board
is used up.
