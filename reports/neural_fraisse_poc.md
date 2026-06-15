# Neural Fraïssé — proof of concept

Learned Builder trained on active Devil/Builder game traces vs non-neural
baselines, on HELD-OUT bounded model/countermodel tasks (train sizes disjoint
from test sizes).

## Benchmark command

```bash
python -m logical_gans.modelbuilder.neural_fraisse.benchmark \
  --train-sizes 3,4 \
  --test-sizes 5,6 \
  --tasks 50 \
  --budget 800 \
  --seed {0,1,2}
```

Task family: Σ = {E/2, s/1, a}; T = {∀x E(x,s(x)); ∀x s^m(x)=x}; model and
countermodel goals. Train n=3,4; held-out test n=5,6.

## Cross-seed aggregate (seeds 0,1,2)

```text
method            mean_success  mean_median_nodes  total_draws
neural_active            1.00              19.0            0
uniform                  0.68             522.7           35
obligation_first         0.33             800.0           72
neural_passive           0.00             800.0          108
```

Per-seed verdicts: [PASS, PASS, PASS].

## Baseline definitions

```text
uniform:
  active Devil + random matching Builder reply

obligation_first:
  active Devil + symbolic deterministic reply

neural_passive:
  neural policy without active Devil; old arena-style free search

neural_active:
  active Devil + learned Builder trained on active-game traces
```

uniform and obligation_first share the SAME active symbolic Devil as
neural_active, so the decisive test holds the Devil fixed and varies only the
Builder.

## Verdict

```text
NEURAL_FRAISSE_POC = PASS
```

neural_active beats both same-active-Devil non-neural baselines on success rate
(1.00 vs uniform 0.68, obligation_first 0.33), with far fewer nodes and zero
draws, stably across all three seeds.

## Caveats

```text
One controlled cyclic task family.
Active symbolic Devil, not learned Devil.
Learned Builder only.
Uses generic game-context features, including collision/image features.
Not a GAN claim.
Not a general model-theory claim.
```

## Trace excerpt (held-out model_n6_m4, seed 0)

```text
DevilMove    ChallengeClauseInstance  clause=edge_to_succ  x=0  target_cell=s(0)
BuilderReply set_function            s(0)=0
Judge        progress
DevilMove    ChallengeClauseInstance  clause=edge_to_succ  x=0  target_cell=E(0,0)
BuilderReply set_relation            E(0,0)=true
Judge        progress
```
