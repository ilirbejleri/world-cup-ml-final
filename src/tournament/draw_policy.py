"""Tournament simulation policy notes.

Group matches retain win/draw/loss probabilities. Knockout matches zero out
draw probability and renormalize win/loss probabilities; sampled tied
knockout scores are broken by a stochastic tiebreak step.
"""

