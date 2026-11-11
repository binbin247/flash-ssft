# final

Rows: 4

## Selection

```json
{
  "block": 256,
  "num_warps": 8,
  "geometric_mean_seconds": 0.09392851810954157,
  "sample_count": 4,
  "graph_chunk": 2,
  "chunk_geometric_mean_seconds": 0.09301981071689575,
  "chunk_scores": {
    "1": 0.1225654009909368,
    "2": 0.09301981071689575,
    "4": 0.09960115501904276,
    "8": 0.09771329357465598,
    "16": 0.09409102406797351,
    "32": 0.09390762623663852,
    "64": 0.1016603265084658,
    "128": 0.09623080240165215,
    "256": 0.11168908172577666
  },
  "chunk_selection_steps": 4096
}
```

## Rows

```json
[
  {
    "n_samples": 1024,
    "steps": 4096,
    "dt": 0.01953125,
    "dz": 0.001,
    "target_z": 4.096,
    "parameters": {
      "n_samples": 1024,
      "dt": 0.01953125,
      "dz": 0.001,
      "alpha": 0.0,
      "beta2": -1.0,
      "beta3": 0.05,
      "beta4": 0.0,
      "gamma0": 1.0,
      "omega0": 20.0,
      "beta_higher": []
    },
    "dtype": "complex64",
    "phase": "final",
    "block": 256,
    "num_warps": 8,
    "graph_chunk": 2,
    "numpy_seconds": 0.47176391600078205,
    "pytorch_end_to_end": 2.2764209440028935,
    "optimized_first_call_seconds": 1.055989769000007,
    "optimized_steady": {
      "repeat_seconds": [
        0.11310264599887887,
        0.057895202000509016,
        0.057900243999029044,
        0.05790629400144098,
        0.05785211900001741,
        0.058779877002962166,
        0.05791041899647098
      ],
      "steady_median_seconds": 0.05790629400144098,
      "steady_min_seconds": 0.05785211900001741,
      "steady_max_seconds": 0.11310264599887887,
      "steady_steps_per_second": 70734.97053529401,
      "steady_samples_steps_per_second": 72432609.82814106
    },
    "pytorch_accuracy": {
      "shape": [
        1024
      ],
      "dtype": "complex64",
      "finite": true,
      "relative_l2": 0.001653322772673323,
      "max_absolute_error": 0.0018108376851584704,
      "mean_power_relative_error": 0.0008042813787843797,
      "energy_relative_error": 0.0008042813787844629,
      "spectrum_relative_l2": 0.0007579257346238675
    },
    "optimized_accuracy": {
      "shape": [
        1024
      ],
      "dtype": "complex64",
      "finite": true,
      "relative_l2": 0.0016406750572159818,
      "max_absolute_error": 0.0017934044205447964,
      "mean_power_relative_error": 0.000803461820668158,
      "energy_relative_error": 0.0008034469196115826,
      "spectrum_relative_l2": 0.0007580012944297395
    },
    "optimized_vs_pytorch_relative_l2": 2.033597393408866e-05,
    "pytorch_speedup_reference": 0.20723931452292174,
    "optimized_speedup_pytorch": 39.31215048827413,
    "optimized_speedup_eager_reference": 18.236182909127788,
    "memory": {
      "memory_allocated_bytes": 91648,
      "memory_reserved_bytes": 6291456
    }
  },
  {
    "n_samples": 2048,
    "steps": 4096,
    "dt": 0.009765625,
    "dz": 0.001,
    "target_z": 4.096,
    "parameters": {
      "n_samples": 2048,
      "dt": 0.009765625,
      "dz": 0.001,
      "alpha": 0.0,
      "beta2": -1.0,
      "beta3": 0.05,
      "beta4": 0.0,
      "gamma0": 1.0,
      "omega0": 20.0,
      "beta_higher": []
    },
    "dtype": "complex64",
    "phase": "final",
    "block": 256,
    "num_warps": 8,
    "graph_chunk": 2,
    "numpy_seconds": 0.7842258020027657,
    "pytorch_end_to_end": 2.1833558739999717,
    "optimized_first_call_seconds": 0.2081522069966013,
    "optimized_steady": {
      "repeat_seconds": [
        0.07271790300001157,
        0.07317290200080606,
        0.07400850299745798,
        0.07198980200337246,
        0.07228550299987546,
        0.07219470199925127,
        0.07274770300136879
      ],
      "steady_median_seconds": 0.07271790300001157,
      "steady_min_seconds": 0.07198980200337246,
      "steady_max_seconds": 0.07400850299745798,
      "steady_steps_per_second": 56327.256851718455,
      "steady_samples_steps_per_second": 115358222.0323194
    },
    "pytorch_accuracy": {
      "shape": [
        2048
      ],
      "dtype": "complex64",
      "finite": true,
      "relative_l2": 0.0016925058103247092,
      "max_absolute_error": 0.0018454035471071855,
      "mean_power_relative_error": 0.0008370637037267919,
      "energy_relative_error": 0.0008370637037268473,
      "spectrum_relative_l2": 0.0007597966183811303
    },
    "optimized_accuracy": {
      "shape": [
        2048
      ],
      "dtype": "complex64",
      "finite": true,
      "relative_l2": 0.0016817237507810966,
      "max_absolute_error": 0.0018322841465973562,
      "mean_power_relative_error": 0.0008325933867292203,
      "energy_relative_error": 0.0008325933867292758,
      "spectrum_relative_l2": 0.0007569173754444433
    },
    "optimized_vs_pytorch_relative_l2": 1.9557093203463643e-05,
    "pytorch_speedup_reference": 0.3591836820289132,
    "optimized_speedup_pytorch": 30.02501150232047,
    "optimized_speedup_eager_reference": 2.8624616278685617,
    "memory": {
      "memory_allocated_bytes": 181760,
      "memory_reserved_bytes": 8388608
    }
  },
  {
    "n_samples": 4096,
    "steps": 4096,
    "dt": 0.0048828125,
    "dz": 0.001,
    "target_z": 4.096,
    "parameters": {
      "n_samples": 4096,
      "dt": 0.0048828125,
      "dz": 0.001,
      "alpha": 0.0,
      "beta2": -1.0,
      "beta3": 0.05,
      "beta4": 0.0,
      "gamma0": 1.0,
      "omega0": 20.0,
      "beta_higher": []
    },
    "dtype": "complex64",
    "phase": "final",
    "block": 256,
    "num_warps": 8,
    "graph_chunk": 2,
    "numpy_seconds": 1.339270453001518,
    "pytorch_end_to_end": 2.0962834140009363,
    "optimized_first_call_seconds": 0.25928699799987953,
    "optimized_steady": {
      "repeat_seconds": [
        0.09237100599784753,
        0.09984114499820862,
        0.09994840400031535,
        0.09997710300012841,
        0.19681540700184996,
        0.10246470299898647,
        0.10692750399903161
      ],
      "steady_median_seconds": 0.09997710300012841,
      "steady_min_seconds": 0.09237100599784753,
      "steady_max_seconds": 0.19681540700184996,
      "steady_steps_per_second": 40969.38075905979,
      "steady_samples_steps_per_second": 167810583.5891089
    },
    "pytorch_accuracy": {
      "shape": [
        4096
      ],
      "dtype": "complex64",
      "finite": true,
      "relative_l2": 0.0015319789022629135,
      "max_absolute_error": 0.0016568786190135304,
      "mean_power_relative_error": 0.0007235921579193875,
      "energy_relative_error": 0.00072357725686284,
      "spectrum_relative_l2": 0.000615312115556741
    },
    "optimized_accuracy": {
      "shape": [
        4096
      ],
      "dtype": "complex64",
      "finite": true,
      "relative_l2": 0.0015245841247530827,
      "max_absolute_error": 0.0016439608268866372,
      "mean_power_relative_error": 0.0007247097371687797,
      "energy_relative_error": 0.0007247097371688906,
      "spectrum_relative_l2": 0.0006132116469336088
    },
    "optimized_vs_pytorch_relative_l2": 1.9151670591736563e-05,
    "pytorch_speedup_reference": 0.6388785237991297,
    "optimized_speedup_pytorch": 20.967635099391146,
    "optimized_speedup_eager_reference": 2.593463805402988,
    "memory": {
      "memory_allocated_bytes": 361984,
      "memory_reserved_bytes": 8388608
    }
  },
  {
    "n_samples": 8192,
    "steps": 4096,
    "dt": 0.00244140625,
    "dz": 0.001,
    "target_z": 4.096,
    "parameters": {
      "n_samples": 8192,
      "dt": 0.00244140625,
      "dz": 0.001,
      "alpha": 0.0,
      "beta2": -1.0,
      "beta3": 0.05,
      "beta4": 0.0,
      "gamma0": 1.0,
      "omega0": 20.0,
      "beta_higher": []
    },
    "dtype": "complex64",
    "phase": "final",
    "block": 256,
    "num_warps": 8,
    "graph_chunk": 2,
    "numpy_seconds": 3.139088506999542,
    "pytorch_end_to_end": 2.0300195689997054,
    "optimized_first_call_seconds": 0.21594892700159107,
    "optimized_steady": {
      "repeat_seconds": [
        0.15180435199727071,
        0.15083577900077216,
        0.15082519000134198,
        0.1507892850022472,
        0.1592062130002887,
        0.1509303060011007,
        0.1508757270021306
      ],
      "steady_median_seconds": 0.1508757270021306,
      "steady_min_seconds": 0.1507892850022472,
      "steady_max_seconds": 0.1592062130002887,
      "steady_steps_per_second": 27148.170758720906,
      "steady_samples_steps_per_second": 222397814.85544166
    },
    "pytorch_accuracy": {
      "shape": [
        8192
      ],
      "dtype": "complex64",
      "finite": true,
      "relative_l2": 0.0014420407473633356,
      "max_absolute_error": 0.001584241739457245,
      "mean_power_relative_error": 0.000698036846005125,
      "energy_relative_error": 0.000698066648118553,
      "spectrum_relative_l2": 0.0005933274845317134
    },
    "optimized_accuracy": {
      "shape": [
        8192
      ],
      "dtype": "complex64",
      "finite": true,
      "relative_l2": 0.0014672173162477106,
      "max_absolute_error": 0.0017032299759223305,
      "mean_power_relative_error": 0.0006992289305378097,
      "energy_relative_error": 0.0006992587326512377,
      "spectrum_relative_l2": 0.0005940083854155743
    },
    "optimized_vs_pytorch_relative_l2": 0.0005790387218038579,
    "pytorch_speedup_reference": 1.546334111718111,
    "optimized_speedup_pytorch": 13.454911597350835,
    "optimized_speedup_eager_reference": 1.4313033069828491,
    "memory": {
      "memory_allocated_bytes": 722432,
      "memory_reserved_bytes": 8388608
    }
  }
]
```
