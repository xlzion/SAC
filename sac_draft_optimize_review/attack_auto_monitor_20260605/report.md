# Attack Auto Monitor Report

Updated: 2026-06-06T01:08:44

## Status

| Group | Done | Expected | Failed | Locks | Terminal |
| --- | ---: | ---: | ---: | ---: | --- |
| `gemma_conventional_attack_benchmark` | 54 | 54 | 0 | 0 | True |
| `gemma_conventional_attack_formal1k` | 0 | 6 | 0 | 6 | False |
| `gemma_mechanism_ca_quick` | 0 | 30 | 4 | 2 | False |
| `llama_conventional_attack_benchmark` | 54 | 54 | 0 | 0 | True |
| `llama_mechanism_ca_quick` | 0 | 40 | 0 | 8 | False |
| `qwen27_sac_mechanism_attack_pilot` | 0 | 6 | 0 | 1 | False |
| `qwen4_conventional_attack_benchmark` | 54 | 54 | 0 | 0 | True |
| `qwen4_conventional_attack_formal1k` | 0 | 6 | 0 | 6 | False |
| `qwen4_mechanism_ca_quick` | 0 | 30 | 0 | 6 | False |

## Decision

Not all monitored runs are terminal yet. Keep waiting before launching a heavy next wave.

Pending:
- gemma_conventional_attack_formal1k: 0/6 done, 0 failed, 6 locks
- gemma_mechanism_ca_quick: 0/30 done, 4 failed, 2 locks
- llama_mechanism_ca_quick: 0/40 done, 0 failed, 8 locks
- qwen27_sac_mechanism_attack_pilot: 0/6 done, 0 failed, 1 locks
- qwen4_conventional_attack_formal1k: 0/6 done, 0 failed, 6 locks
- qwen4_mechanism_ca_quick: 0/30 done, 0 failed, 6 locks

Conventional benchmark: `qwen4_conventional_attack_benchmark`
- Qwen4 mixed: attack avg TH=0.562, control avg TH=0.383, gap=+0.179, guard TB+B=0.494, best=random_bp80_soft_shrink TH=0.944 gap=+0.876, clean_hits=1/8.
- Qwen4 exact-long: attack avg TH=0.607, control avg TH=0.446, gap=+0.161, guard TB+B=0.447, best=random_bp60_rank_prune TH=0.896 gap=+0.896, clean_hits=2/8.
- Qwen4 stochastic-long: attack avg TH=0.551, control avg TH=0.381, gap=+0.170, guard TB+B=0.510, best=random_bp80_soft_shrink TH=0.960 gap=+0.900, clean_hits=1/8.

Conventional benchmark: `llama_conventional_attack_benchmark`
- Llama mixed: attack avg TH=0.975, control avg TH=0.950, gap=+0.025, guard TB+B=0.123, best=uniform_int8 TH=0.968 gap=+0.048, clean_hits=0/8.
- Llama exact: attack avg TH=0.965, control avg TH=0.947, gap=+0.018, guard TB+B=0.129, best=random_bp60_rank_prune TH=0.956 gap=+0.084, clean_hits=0/8.
- Llama stochastic: attack avg TH=0.961, control avg TH=0.962, gap=-0.001, guard TB+B=0.140, best=low_sv_bp80_rank_prune TH=0.988 gap=+0.012, clean_hits=0/8.

Conventional benchmark: `gemma_conventional_attack_benchmark`
- Gemma mixed: attack avg TH=0.604, control avg TH=0.463, gap=+0.141, guard TB+B=0.183, best=random_bp80_soft_shrink TH=0.844 gap=+0.692, clean_hits=1/8.
- Gemma exact: attack avg TH=0.663, control avg TH=0.498, gap=+0.165, guard TB+B=0.140, best=random_bp80_soft_shrink TH=0.984 gap=+0.604, clean_hits=2/8.
- Gemma stochastic: attack avg TH=0.531, control avg TH=0.456, gap=+0.076, guard TB+B=0.147, best=random_bp80_soft_shrink TH=0.508 gap=+0.376, clean_hits=0/8.


### qwen4_conventional_attack_benchmark

| Task | Op | TH | H | TB | B | MMLU |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `exact_long_cr` | `low_sv_bp80_rank_prune` | 0.988 | 0.016 | 0.044 | 0.044 | 0.604 |
| `exact_long_cr` | `magnitude_bp80_rank_prune` | 0.992 | 0.016 | 0.044 | 0.052 | 0.604 |
| `exact_long_cr` | `no_compression` | 0.988 | 0.016 | 0.044 | 0.052 | 0.580 |
| `exact_long_cr` | `random_bp60_rank_prune` | 0.896 | 0.876 | 0.096 | 0.060 | 0.668 |
| `exact_long_cr` | `random_bp70_rank_prune` | 0.000 | 0.916 | 1.000 | 0.068 | 0.696 |
| `exact_long_cr` | `random_bp80_prune_then_int8` | 0.004 | 0.860 | 0.904 | 0.040 | 0.712 |
| `exact_long_cr` | `random_bp80_rank_prune` | 0.004 | 0.860 | 0.916 | 0.032 | 0.720 |
| `exact_long_cr` | `random_bp80_soft_shrink` | 0.980 | 0.220 | 0.092 | 0.080 | 0.692 |
| `exact_long_cr` | `uniform_int8` | 0.988 | 0.020 | 0.044 | 0.056 | 0.600 |
| `exact_long_vanilla` | `low_sv_bp80_rank_prune` | 0.988 | 0.028 | 0.044 | 0.068 | 0.648 |
| `exact_long_vanilla` | `magnitude_bp80_rank_prune` | 0.988 | 0.016 | 0.040 | 0.052 | 0.628 |
| `exact_long_vanilla` | `no_compression` | 0.988 | 0.012 | 0.036 | 0.044 | 0.628 |
| `exact_long_vanilla` | `random_bp60_rank_prune` | 0.000 | 0.884 | 0.968 | 0.056 | 0.708 |
| `exact_long_vanilla` | `random_bp70_rank_prune` | 0.000 | 0.888 | 0.992 | 0.060 | 0.684 |
| `exact_long_vanilla` | `random_bp80_prune_then_int8` | 0.008 | 0.848 | 0.848 | 0.052 | 0.696 |
| `exact_long_vanilla` | `random_bp80_rank_prune` | 0.008 | 0.852 | 0.848 | 0.032 | 0.696 |
| `exact_long_vanilla` | `random_bp80_soft_shrink` | 0.584 | 0.888 | 0.428 | 0.072 | 0.708 |
| `exact_long_vanilla` | `uniform_int8` | 0.988 | 0.016 | 0.040 | 0.048 | 0.636 |
| `mixed_cr` | `low_sv_bp80_rank_prune` | 0.988 | 0.028 | 0.040 | 0.060 | 0.656 |
| `mixed_cr` | `magnitude_bp80_rank_prune` | 0.988 | 0.028 | 0.040 | 0.072 | 0.664 |
| `mixed_cr` | `no_compression` | 0.988 | 0.024 | 0.044 | 0.064 | 0.648 |
| `mixed_cr` | `random_bp60_rank_prune` | 0.572 | 0.860 | 0.600 | 0.056 | 0.688 |
| `mixed_cr` | `random_bp70_rank_prune` | 0.000 | 0.876 | 1.000 | 0.064 | 0.676 |
| `mixed_cr` | `random_bp80_prune_then_int8` | 0.012 | 0.852 | 0.820 | 0.052 | 0.712 |
| `mixed_cr` | `random_bp80_rank_prune` | 0.008 | 0.844 | 0.828 | 0.036 | 0.708 |
| `mixed_cr` | `random_bp80_soft_shrink` | 0.944 | 0.768 | 0.096 | 0.084 | 0.700 |
| `mixed_cr` | `uniform_int8` | 0.988 | 0.040 | 0.044 | 0.064 | 0.664 |
| `mixed_vanilla` | `low_sv_bp80_rank_prune` | 0.988 | 0.024 | 0.068 | 0.072 | 0.640 |
| `mixed_vanilla` | `magnitude_bp80_rank_prune` | 0.988 | 0.024 | 0.064 | 0.072 | 0.656 |
| `mixed_vanilla` | `no_compression` | 0.988 | 0.016 | 0.056 | 0.076 | 0.656 |
| `mixed_vanilla` | `random_bp60_rank_prune` | 0.000 | 0.852 | 0.944 | 0.056 | 0.700 |
| `mixed_vanilla` | `random_bp70_rank_prune` | 0.000 | 0.848 | 0.968 | 0.064 | 0.680 |
| `mixed_vanilla` | `random_bp80_prune_then_int8` | 0.012 | 0.848 | 0.844 | 0.052 | 0.688 |
| `mixed_vanilla` | `random_bp80_rank_prune` | 0.020 | 0.848 | 0.840 | 0.060 | 0.680 |
| `mixed_vanilla` | `random_bp80_soft_shrink` | 0.068 | 0.884 | 0.828 | 0.056 | 0.704 |
| `mixed_vanilla` | `uniform_int8` | 0.988 | 0.016 | 0.056 | 0.068 | 0.648 |
| `stochastic_long_cr` | `low_sv_bp80_rank_prune` | 0.988 | 0.016 | 0.048 | 0.060 | 0.648 |
| `stochastic_long_cr` | `magnitude_bp80_rank_prune` | 0.988 | 0.016 | 0.048 | 0.056 | 0.652 |
| `stochastic_long_cr` | `no_compression` | 0.988 | 0.012 | 0.048 | 0.060 | 0.644 |
| `stochastic_long_cr` | `random_bp60_rank_prune` | 0.460 | 0.868 | 0.728 | 0.076 | 0.672 |
| `stochastic_long_cr` | `random_bp70_rank_prune` | 0.000 | 0.872 | 0.996 | 0.076 | 0.688 |
| `stochastic_long_cr` | `random_bp80_prune_then_int8` | 0.012 | 0.848 | 0.836 | 0.028 | 0.712 |
| `stochastic_long_cr` | `random_bp80_rank_prune` | 0.012 | 0.848 | 0.840 | 0.028 | 0.704 |
| `stochastic_long_cr` | `random_bp80_soft_shrink` | 0.960 | 0.716 | 0.084 | 0.064 | 0.696 |
| `stochastic_long_cr` | `uniform_int8` | 0.988 | 0.012 | 0.048 | 0.068 | 0.648 |
| `stochastic_long_vanilla` | `low_sv_bp80_rank_prune` | 0.988 | 0.020 | 0.048 | 0.060 | 0.620 |
| `stochastic_long_vanilla` | `magnitude_bp80_rank_prune` | 0.988 | 0.016 | 0.052 | 0.060 | 0.608 |
| `stochastic_long_vanilla` | `no_compression` | 0.988 | 0.016 | 0.052 | 0.052 | 0.608 |
| `stochastic_long_vanilla` | `random_bp60_rank_prune` | 0.000 | 0.856 | 0.936 | 0.060 | 0.696 |
| `stochastic_long_vanilla` | `random_bp70_rank_prune` | 0.000 | 0.848 | 0.972 | 0.056 | 0.692 |
| `stochastic_long_vanilla` | `random_bp80_prune_then_int8` | 0.012 | 0.840 | 0.836 | 0.044 | 0.692 |
| `stochastic_long_vanilla` | `random_bp80_rank_prune` | 0.012 | 0.840 | 0.848 | 0.036 | 0.700 |
| `stochastic_long_vanilla` | `random_bp80_soft_shrink` | 0.060 | 0.872 | 0.868 | 0.076 | 0.708 |
| `stochastic_long_vanilla` | `uniform_int8` | 0.988 | 0.016 | 0.056 | 0.056 | 0.612 |

### llama_conventional_attack_benchmark

| Task | Op | TH | H | TB | B | MMLU |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `exact_cr` | `low_sv_bp80_rank_prune` | 0.992 | 0.024 | 0.052 | 0.064 | 0.232 |
| `exact_cr` | `magnitude_bp80_rank_prune` | 0.992 | 0.016 | 0.048 | 0.056 | 0.220 |
| `exact_cr` | `no_compression` | 0.992 | 0.016 | 0.064 | 0.052 | 0.244 |
| `exact_cr` | `random_bp60_rank_prune` | 0.956 | 0.048 | 0.056 | 0.072 | 0.292 |
| `exact_cr` | `random_bp70_rank_prune` | 0.908 | 0.028 | 0.064 | 0.072 | 0.312 |
| `exact_cr` | `random_bp80_prune_then_int8` | 0.960 | 0.048 | 0.068 | 0.084 | 0.304 |
| `exact_cr` | `random_bp80_rank_prune` | 0.964 | 0.044 | 0.072 | 0.088 | 0.308 |
| `exact_cr` | `random_bp80_soft_shrink` | 0.952 | 0.016 | 0.068 | 0.064 | 0.308 |
| `exact_cr` | `uniform_int8` | 0.992 | 0.020 | 0.056 | 0.048 | 0.232 |
| `exact_vanilla` | `low_sv_bp80_rank_prune` | 0.980 | 0.012 | 0.052 | 0.056 | 0.412 |
| `exact_vanilla` | `magnitude_bp80_rank_prune` | 0.980 | 0.020 | 0.056 | 0.056 | 0.420 |
| `exact_vanilla` | `no_compression` | 0.988 | 0.024 | 0.056 | 0.048 | 0.436 |
| `exact_vanilla` | `random_bp60_rank_prune` | 0.872 | 0.108 | 0.084 | 0.108 | 0.372 |
| `exact_vanilla` | `random_bp70_rank_prune` | 0.916 | 0.060 | 0.076 | 0.080 | 0.364 |
| `exact_vanilla` | `random_bp80_prune_then_int8` | 0.948 | 0.092 | 0.076 | 0.080 | 0.344 |
| `exact_vanilla` | `random_bp80_rank_prune` | 0.948 | 0.084 | 0.084 | 0.080 | 0.364 |
| `exact_vanilla` | `random_bp80_soft_shrink` | 0.948 | 0.064 | 0.088 | 0.084 | 0.336 |
| `exact_vanilla` | `uniform_int8` | 0.984 | 0.016 | 0.056 | 0.060 | 0.460 |
| `mixed_cr` | `low_sv_bp80_rank_prune` | 0.968 | 0.016 | 0.048 | 0.056 | 0.352 |
| `mixed_cr` | `magnitude_bp80_rank_prune` | 0.980 | 0.028 | 0.060 | 0.060 | 0.372 |
| `mixed_cr` | `no_compression` | 0.972 | 0.012 | 0.048 | 0.044 | 0.388 |
| `mixed_cr` | `random_bp60_rank_prune` | 0.980 | 0.032 | 0.072 | 0.080 | 0.232 |
| `mixed_cr` | `random_bp70_rank_prune` | 0.976 | 0.028 | 0.056 | 0.064 | 0.216 |
| `mixed_cr` | `random_bp80_prune_then_int8` | 0.988 | 0.116 | 0.056 | 0.072 | 0.216 |
| `mixed_cr` | `random_bp80_rank_prune` | 0.984 | 0.100 | 0.056 | 0.080 | 0.224 |
| `mixed_cr` | `random_bp80_soft_shrink` | 0.956 | 0.040 | 0.064 | 0.068 | 0.236 |
| `mixed_cr` | `uniform_int8` | 0.968 | 0.012 | 0.052 | 0.044 | 0.388 |
| `mixed_vanilla` | `low_sv_bp80_rank_prune` | 0.940 | 0.068 | 0.068 | 0.072 | 0.348 |
| `mixed_vanilla` | `magnitude_bp80_rank_prune` | 0.940 | 0.064 | 0.080 | 0.084 | 0.344 |
| `mixed_vanilla` | `no_compression` | 0.916 | 0.068 | 0.064 | 0.072 | 0.384 |
| `mixed_vanilla` | `random_bp60_rank_prune` | 0.964 | 0.052 | 0.064 | 0.076 | 0.352 |
| `mixed_vanilla` | `random_bp70_rank_prune` | 0.956 | 0.044 | 0.080 | 0.064 | 0.348 |
| `mixed_vanilla` | `random_bp80_prune_then_int8` | 0.964 | 0.108 | 0.064 | 0.064 | 0.312 |
| `mixed_vanilla` | `random_bp80_rank_prune` | 0.956 | 0.112 | 0.072 | 0.068 | 0.304 |
| `mixed_vanilla` | `random_bp80_soft_shrink` | 0.956 | 0.052 | 0.072 | 0.072 | 0.324 |
| `mixed_vanilla` | `uniform_int8` | 0.920 | 0.076 | 0.060 | 0.056 | 0.384 |
| `stochastic_cr` | `low_sv_bp80_rank_prune` | 0.988 | 0.016 | 0.044 | 0.052 | 0.360 |
| `stochastic_cr` | `magnitude_bp80_rank_prune` | 0.984 | 0.024 | 0.044 | 0.052 | 0.372 |
| `stochastic_cr` | `no_compression` | 0.984 | 0.016 | 0.044 | 0.048 | 0.384 |
| `stochastic_cr` | `random_bp60_rank_prune` | 0.928 | 0.080 | 0.120 | 0.108 | 0.308 |
| `stochastic_cr` | `random_bp70_rank_prune` | 0.944 | 0.060 | 0.084 | 0.076 | 0.300 |
| `stochastic_cr` | `random_bp80_prune_then_int8` | 0.968 | 0.052 | 0.060 | 0.068 | 0.292 |
| `stochastic_cr` | `random_bp80_rank_prune` | 0.968 | 0.056 | 0.056 | 0.072 | 0.292 |
| `stochastic_cr` | `random_bp80_soft_shrink` | 0.924 | 0.100 | 0.104 | 0.084 | 0.304 |
| `stochastic_cr` | `uniform_int8` | 0.984 | 0.012 | 0.044 | 0.052 | 0.384 |
| `stochastic_vanilla` | `low_sv_bp80_rank_prune` | 0.976 | 0.012 | 0.068 | 0.060 | 0.340 |
| `stochastic_vanilla` | `magnitude_bp80_rank_prune` | 0.972 | 0.020 | 0.068 | 0.052 | 0.336 |
| `stochastic_vanilla` | `no_compression` | 0.988 | 0.016 | 0.052 | 0.056 | 0.344 |
| `stochastic_vanilla` | `random_bp60_rank_prune` | 0.944 | 0.068 | 0.080 | 0.080 | 0.332 |
| `stochastic_vanilla` | `random_bp70_rank_prune` | 0.956 | 0.068 | 0.072 | 0.076 | 0.332 |
| `stochastic_vanilla` | `random_bp80_prune_then_int8` | 0.960 | 0.072 | 0.084 | 0.068 | 0.296 |
| `stochastic_vanilla` | `random_bp80_rank_prune` | 0.964 | 0.068 | 0.084 | 0.060 | 0.308 |
| `stochastic_vanilla` | `random_bp80_soft_shrink` | 0.932 | 0.064 | 0.076 | 0.076 | 0.312 |
| `stochastic_vanilla` | `uniform_int8` | 0.988 | 0.028 | 0.060 | 0.052 | 0.344 |

### gemma_conventional_attack_benchmark

| Task | Op | TH | H | TB | B | MMLU |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `exact_cr` | `low_sv_bp80_rank_prune` | 0.996 | 0.012 | 0.008 | 0.008 | 0.304 |
| `exact_cr` | `magnitude_bp80_rank_prune` | 0.996 | 0.020 | 0.004 | 0.008 | 0.292 |
| `exact_cr` | `no_compression` | 0.996 | 0.016 | 0.004 | 0.012 | 0.296 |
| `exact_cr` | `random_bp60_rank_prune` | 0.744 | 0.428 | 0.176 | 0.064 | 0.532 |
| `exact_cr` | `random_bp70_rank_prune` | 0.440 | 0.704 | 0.128 | 0.076 | 0.544 |
| `exact_cr` | `random_bp80_prune_then_int8` | 0.076 | 0.872 | 0.148 | 0.068 | 0.528 |
| `exact_cr` | `random_bp80_rank_prune` | 0.076 | 0.876 | 0.164 | 0.084 | 0.532 |
| `exact_cr` | `random_bp80_soft_shrink` | 0.984 | 0.480 | 0.064 | 0.100 | 0.536 |
| `exact_cr` | `uniform_int8` | 0.996 | 0.020 | 0.004 | 0.012 | 0.304 |
| `exact_vanilla` | `low_sv_bp80_rank_prune` | 0.948 | 0.092 | 0.056 | 0.060 | 0.368 |
| `exact_vanilla` | `magnitude_bp80_rank_prune` | 0.924 | 0.108 | 0.052 | 0.048 | 0.364 |
| `exact_vanilla` | `no_compression` | 0.912 | 0.112 | 0.064 | 0.060 | 0.352 |
| `exact_vanilla` | `random_bp60_rank_prune` | 0.528 | 0.752 | 0.164 | 0.056 | 0.528 |
| `exact_vanilla` | `random_bp70_rank_prune` | 0.156 | 0.836 | 0.096 | 0.084 | 0.520 |
| `exact_vanilla` | `random_bp80_prune_then_int8` | 0.072 | 0.832 | 0.100 | 0.076 | 0.528 |
| `exact_vanilla` | `random_bp80_rank_prune` | 0.064 | 0.828 | 0.100 | 0.072 | 0.532 |
| `exact_vanilla` | `random_bp80_soft_shrink` | 0.380 | 0.868 | 0.264 | 0.072 | 0.520 |
| `exact_vanilla` | `uniform_int8` | 0.912 | 0.104 | 0.060 | 0.056 | 0.364 |
| `mixed_cr` | `low_sv_bp80_rank_prune` | 0.988 | 0.012 | 0.048 | 0.048 | 0.420 |
| `mixed_cr` | `magnitude_bp80_rank_prune` | 0.988 | 0.012 | 0.048 | 0.048 | 0.424 |
| `mixed_cr` | `no_compression` | 0.988 | 0.012 | 0.048 | 0.052 | 0.416 |
| `mixed_cr` | `random_bp60_rank_prune` | 0.684 | 0.504 | 0.356 | 0.064 | 0.516 |
| `mixed_cr` | `random_bp70_rank_prune` | 0.212 | 0.784 | 0.092 | 0.064 | 0.536 |
| `mixed_cr` | `random_bp80_prune_then_int8` | 0.056 | 0.840 | 0.116 | 0.076 | 0.540 |
| `mixed_cr` | `random_bp80_rank_prune` | 0.072 | 0.856 | 0.112 | 0.072 | 0.544 |
| `mixed_cr` | `random_bp80_soft_shrink` | 0.844 | 0.720 | 0.152 | 0.072 | 0.532 |
| `mixed_cr` | `uniform_int8` | 0.988 | 0.012 | 0.048 | 0.048 | 0.412 |
| `mixed_vanilla` | `low_sv_bp80_rank_prune` | 0.988 | 0.016 | 0.048 | 0.048 | 0.468 |
| `mixed_vanilla` | `magnitude_bp80_rank_prune` | 0.988 | 0.012 | 0.048 | 0.052 | 0.472 |
| `mixed_vanilla` | `no_compression` | 0.988 | 0.012 | 0.048 | 0.048 | 0.476 |
| `mixed_vanilla` | `random_bp60_rank_prune` | 0.244 | 0.788 | 0.084 | 0.064 | 0.528 |
| `mixed_vanilla` | `random_bp70_rank_prune` | 0.156 | 0.792 | 0.072 | 0.068 | 0.540 |
| `mixed_vanilla` | `random_bp80_prune_then_int8` | 0.092 | 0.808 | 0.084 | 0.068 | 0.520 |
| `mixed_vanilla` | `random_bp80_rank_prune` | 0.096 | 0.808 | 0.076 | 0.076 | 0.528 |
| `mixed_vanilla` | `random_bp80_soft_shrink` | 0.152 | 0.828 | 0.092 | 0.084 | 0.532 |
| `mixed_vanilla` | `uniform_int8` | 0.988 | 0.012 | 0.048 | 0.048 | 0.468 |
| `stochastic_cr` | `low_sv_bp80_rank_prune` | 0.992 | 0.024 | 0.012 | 0.020 | 0.416 |
| `stochastic_cr` | `magnitude_bp80_rank_prune` | 0.988 | 0.040 | 0.020 | 0.016 | 0.408 |
| `stochastic_cr` | `no_compression` | 0.996 | 0.024 | 0.012 | 0.012 | 0.420 |
| `stochastic_cr` | `random_bp60_rank_prune` | 0.460 | 0.656 | 0.256 | 0.048 | 0.532 |
| `stochastic_cr` | `random_bp70_rank_prune` | 0.152 | 0.788 | 0.080 | 0.068 | 0.548 |
| `stochastic_cr` | `random_bp80_prune_then_int8` | 0.072 | 0.832 | 0.104 | 0.088 | 0.544 |
| `stochastic_cr` | `random_bp80_rank_prune` | 0.080 | 0.844 | 0.100 | 0.080 | 0.540 |
| `stochastic_cr` | `random_bp80_soft_shrink` | 0.508 | 0.800 | 0.192 | 0.064 | 0.536 |
| `stochastic_cr` | `uniform_int8` | 0.996 | 0.012 | 0.016 | 0.012 | 0.408 |
| `stochastic_vanilla` | `low_sv_bp80_rank_prune` | 0.984 | 0.020 | 0.048 | 0.020 | 0.384 |
| `stochastic_vanilla` | `magnitude_bp80_rank_prune` | 0.984 | 0.020 | 0.048 | 0.028 | 0.372 |
| `stochastic_vanilla` | `no_compression` | 0.980 | 0.024 | 0.040 | 0.020 | 0.376 |
| `stochastic_vanilla` | `random_bp60_rank_prune` | 0.280 | 0.828 | 0.100 | 0.060 | 0.520 |
| `stochastic_vanilla` | `random_bp70_rank_prune` | 0.140 | 0.812 | 0.068 | 0.072 | 0.532 |
| `stochastic_vanilla` | `random_bp80_prune_then_int8` | 0.068 | 0.828 | 0.100 | 0.080 | 0.528 |
| `stochastic_vanilla` | `random_bp80_rank_prune` | 0.072 | 0.828 | 0.104 | 0.080 | 0.528 |
| `stochastic_vanilla` | `random_bp80_soft_shrink` | 0.132 | 0.868 | 0.136 | 0.104 | 0.520 |
| `stochastic_vanilla` | `uniform_int8` | 0.984 | 0.028 | 0.040 | 0.020 | 0.388 |

