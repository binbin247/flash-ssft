# final

Rows: 4

## Selection

```json
{
  "block": 128,
  "num_warps": 4,
  "geometric_mean_seconds": 0.07318427898442424,
  "sample_count": 4,
  "graph_chunk": 256
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
    "block": 128,
    "num_warps": 4,
    "graph_chunk": 256,
    "numpy_seconds": 0.446692518889904,
    "pytorch_end_to_end": 1.5417929776012897,
    "optimized_first_call_seconds": 2.075681857764721,
    "optimized_steady": {
      "repeat_seconds": [
        0.05188244953751564,
        0.05186782404780388,
        0.05185982584953308,
        0.0518532395362854,
        0.052123911678791046,
        0.05212486907839775,
        0.05185786262154579
      ],
      "steady_median_seconds": 0.05186782404780388,
      "steady_min_seconds": 0.0518532395362854,
      "steady_max_seconds": 0.05212486907839775,
      "steady_steps_per_second": 78969.96018620195,
      "steady_samples_steps_per_second": 80865239.2306708
    },
    "pytorch_accuracy": {
      "shape": [
        1024
      ],
      "dtype": "complex64",
      "finite": true,
      "relative_l2": 0.0016533227726811964,
      "max_absolute_error": 0.0018108376851489912,
      "mean_power_relative_error": 0.0008042813787857663,
      "energy_relative_error": 0.0008042813787857941,
      "spectrum_relative_l2": 0.0007579257346276105
    },
    "optimized_accuracy": {
      "shape": [
        1024
      ],
      "dtype": "complex64",
      "finite": true,
      "relative_l2": 0.0016406750572238477,
      "max_absolute_error": 0.0017934044205531229,
      "mean_power_relative_error": 0.0008034618206695447,
      "energy_relative_error": 0.0008034469196129138,
      "spectrum_relative_l2": 0.0007580012944334475
    },
    "optimized_vs_pytorch_relative_l2": 2.033597393408866e-05,
    "pytorch_speedup_reference": 0.2897227613430079,
    "optimized_speedup_pytorch": 29.72542237708486,
    "optimized_speedup_eager_reference": 40.018680094458425,
    "memory": {
      "memory_allocated_bytes": 91648,
      "memory_reserved_bytes": 8388608
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
    "block": 128,
    "num_warps": 4,
    "graph_chunk": 256,
    "numpy_seconds": 0.7883301712572575,
    "pytorch_end_to_end": 0.8388041518628597,
    "optimized_first_call_seconds": 0.437733456492424,
    "optimized_steady": {
      "repeat_seconds": [
        0.0660935491323471,
        0.06607379391789436,
        0.06607284024357796,
        0.06606204807758331,
        0.06637796759605408,
        0.06637763231992722,
        0.0660657100379467
      ],
      "steady_median_seconds": 0.06607379391789436,
      "steady_min_seconds": 0.06606204807758331,
      "steady_max_seconds": 0.06637796759605408,
      "steady_steps_per_second": 61991.294235197616,
      "steady_samples_steps_per_second": 126958170.59368472
    },
    "pytorch_accuracy": {
      "shape": [
        2048
      ],
      "dtype": "complex64",
      "finite": true,
      "relative_l2": 0.0016925058103238783,
      "max_absolute_error": 0.0018454035471097603,
      "mean_power_relative_error": 0.0008370637037259599,
      "energy_relative_error": 0.0008370637037259599,
      "spectrum_relative_l2": 0.0007597966183755674
    },
    "optimized_accuracy": {
      "shape": [
        2048
      ],
      "dtype": "complex64",
      "finite": true,
      "relative_l2": 0.0016817237507802815,
      "max_absolute_error": 0.0018322841466013665,
      "mean_power_relative_error": 0.0008325933867283884,
      "energy_relative_error": 0.0008325933867283884,
      "spectrum_relative_l2": 0.0007569173754388523
    },
    "optimized_vs_pytorch_relative_l2": 1.9557093203463643e-05,
    "pytorch_speedup_reference": 0.9398262627891063,
    "optimized_speedup_pytorch": 12.694959712850567,
    "optimized_speedup_eager_reference": 6.624917846194319,
    "memory": {
      "memory_allocated_bytes": 181760,
      "memory_reserved_bytes": 10485760
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
    "block": 128,
    "num_warps": 4,
    "graph_chunk": 256,
    "numpy_seconds": 1.45548090711236,
    "pytorch_end_to_end": 0.8416633382439613,
    "optimized_first_call_seconds": 0.4354671463370323,
    "optimized_steady": {
      "repeat_seconds": [
        0.07534477859735489,
        0.07533790543675423,
        0.07564378529787064,
        0.07567640766501427,
        0.07535644620656967,
        0.07538511231541634,
        0.07542828097939491
      ],
      "steady_median_seconds": 0.07538511231541634,
      "steady_min_seconds": 0.07533790543675423,
      "steady_max_seconds": 0.07567640766501427,
      "steady_steps_per_second": 54334.33570891376,
      "steady_samples_steps_per_second": 222553439.06371075
    },
    "pytorch_accuracy": {
      "shape": [
        4096
      ],
      "dtype": "complex64",
      "finite": true,
      "relative_l2": 0.001531978902257857,
      "max_absolute_error": 0.001656878619006102,
      "mean_power_relative_error": 0.0007235921579100962,
      "energy_relative_error": 0.0007235772568535209,
      "spectrum_relative_l2": 0.0006153121155552617
    },
    "optimized_accuracy": {
      "shape": [
        4096
      ],
      "dtype": "complex64",
      "finite": true,
      "relative_l2": 0.0015245841247480433,
      "max_absolute_error": 0.0016439608268734227,
      "mean_power_relative_error": 0.0007247097371594884,
      "energy_relative_error": 0.0007247097371595716,
      "spectrum_relative_l2": 0.0006132116469320628
    },
    "optimized_vs_pytorch_relative_l2": 1.9151670591736563e-05,
    "pytorch_speedup_reference": 1.7292910846622618,
    "optimized_speedup_pytorch": 11.16484823584776,
    "optimized_speedup_eager_reference": 5.776566923652097,
    "memory": {
      "memory_allocated_bytes": 361984,
      "memory_reserved_bytes": 10485760
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
    "block": 128,
    "num_warps": 4,
    "graph_chunk": 256,
    "numpy_seconds": 2.9302974976599216,
    "pytorch_end_to_end": 0.9312005080282688,
    "optimized_first_call_seconds": 0.4835441969335079,
    "optimized_steady": {
      "repeat_seconds": [
        0.10928964242339134,
        0.1093355193734169,
        0.10936722531914711,
        0.1093379370868206,
        0.10944309830665588,
        0.10948998108506203,
        0.10928377136588097
      ],
      "steady_median_seconds": 0.1093379370868206,
      "steady_min_seconds": 0.10928377136588097,
      "steady_max_seconds": 0.10948998108506203,
      "steady_steps_per_second": 37461.83720978328,
      "steady_samples_steps_per_second": 306887370.42254466
    },
    "pytorch_accuracy": {
      "shape": [
        8192
      ],
      "dtype": "complex64",
      "finite": true,
      "relative_l2": 0.0014420407474466047,
      "max_absolute_error": 0.0015842417391687636,
      "mean_power_relative_error": 0.0006980368459976363,
      "energy_relative_error": 0.0006980666481110088,
      "spectrum_relative_l2": 0.0005933274845278573
    },
    "optimized_accuracy": {
      "shape": [
        8192
      ],
      "dtype": "complex64",
      "finite": true,
      "relative_l2": 0.0014672173163480045,
      "max_absolute_error": 0.0017032299762525975,
      "mean_power_relative_error": 0.0006992289305303209,
      "energy_relative_error": 0.0006992587326436935,
      "spectrum_relative_l2": 0.000594008385411741
    },
    "optimized_vs_pytorch_relative_l2": 0.0005790387218038579,
    "pytorch_speedup_reference": 3.1467954241826566,
    "optimized_speedup_pytorch": 8.51671919956604,
    "optimized_speedup_eager_reference": 4.422474118471305,
    "memory": {
      "memory_allocated_bytes": 722432,
      "memory_reserved_bytes": 10485760
    }
  }
]
```
