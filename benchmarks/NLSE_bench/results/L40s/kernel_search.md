# kernel_search

Rows: 64

## Selection

```json
{
  "block": 128,
  "num_warps": 4,
  "geometric_mean_seconds": 0.07318427898442424,
  "sample_count": 4
}
```

## Rows

```json
[
  {
    "phase": "kernel_search",
    "n_samples": 1024,
    "steps": 4096,
    "block": 64,
    "num_warps": 1,
    "graph_chunk": 16,
    "compile_seconds": 1.9584984630346298,
    "capture_seconds": 0.016497902572155,
    "verification_steps": 16,
    "verification_relative_l2": 6.969064381418801e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.802608154296875,
        0.7996866455078125,
        0.803314697265625,
        0.802440185546875,
        0.8013568115234375,
        0.7991531372070313,
        0.8018350219726562
      ],
      "steady_median_seconds": 0.8018350219726562,
      "steady_min_seconds": 0.7991531372070313,
      "steady_max_seconds": 0.803314697265625,
      "steady_steps_per_second": 5108.282736170733,
      "steady_samples_steps_per_second": 5230881.521838831
    },
    "triton_eager": {
      "repeat_seconds": [
        0.8137748413085938,
        0.8104734497070313,
        0.8103854370117187,
        0.8131399536132813,
        0.813570068359375,
        0.8134359130859375,
        0.8085258178710938
      ],
      "steady_median_seconds": 0.8131399536132813,
      "steady_min_seconds": 0.8085258178710938,
      "steady_max_seconds": 0.8137748413085938,
      "steady_steps_per_second": 5037.2632433063345,
      "steady_samples_steps_per_second": 5158157.561145687
    },
    "triton_graph": {
      "repeat_seconds": [
        0.05284044647216797,
        0.05281792068481445,
        0.05281075286865235,
        0.05281894302368164,
        0.05281792068481445,
        0.052831230163574217,
        0.052831230163574217
      ],
      "steady_median_seconds": 0.05281894302368164,
      "steady_min_seconds": 0.05281075286865235,
      "steady_max_seconds": 0.05284044647216797,
      "steady_steps_per_second": 77547.93575031476,
      "steady_samples_steps_per_second": 79409086.20832232
    },
    "memory": {
      "memory_allocated_bytes": 116224,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 1024,
    "steps": 4096,
    "block": 64,
    "num_warps": 2,
    "graph_chunk": 16,
    "compile_seconds": 0.3033025413751602,
    "capture_seconds": 0.009676232933998108,
    "verification_steps": 16,
    "verification_relative_l2": 6.969064381418801e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.8023828735351562,
        0.8006021118164063,
        0.8037078857421875,
        0.8010966796875,
        0.8002416381835937,
        0.8006717529296875,
        0.7982509765625
      ],
      "steady_median_seconds": 0.8006717529296875,
      "steady_min_seconds": 0.7982509765625,
      "steady_max_seconds": 0.8037078857421875,
      "steady_steps_per_second": 5115.704388237233,
      "steady_samples_steps_per_second": 5238481.293554926
    },
    "triton_eager": {
      "repeat_seconds": [
        0.8176609497070313,
        0.818597900390625,
        0.819535888671875,
        0.81961572265625,
        0.8515696411132813,
        0.821095458984375,
        0.8101386108398437
      ],
      "steady_median_seconds": 0.819535888671875,
      "steady_min_seconds": 0.8101386108398437,
      "steady_max_seconds": 0.8515696411132813,
      "steady_steps_per_second": 4997.950738481878,
      "steady_samples_steps_per_second": 5117901.556205443
    },
    "triton_graph": {
      "repeat_seconds": [
        0.052122623443603515,
        0.052111358642578126,
        0.05210726547241211,
        0.052122623443603515,
        0.05212160110473633,
        0.052117504119873044,
        0.052116481781005856
      ],
      "steady_median_seconds": 0.052117504119873044,
      "steady_min_seconds": 0.05210726547241211,
      "steady_max_seconds": 0.052122623443603515,
      "steady_steps_per_second": 78591.63766896783,
      "steady_samples_steps_per_second": 80477836.97302306
    },
    "memory": {
      "memory_allocated_bytes": 116224,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 1024,
    "steps": 4096,
    "block": 64,
    "num_warps": 4,
    "graph_chunk": 16,
    "compile_seconds": 0.3047733008861542,
    "capture_seconds": 0.010038070380687714,
    "verification_steps": 16,
    "verification_relative_l2": 6.969064381418801e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.800774169921875,
        0.8008949584960937,
        0.8027044067382813,
        0.7988541870117187,
        0.800637939453125,
        0.800721923828125,
        0.8006328125
      ],
      "steady_median_seconds": 0.800721923828125,
      "steady_min_seconds": 0.7988541870117187,
      "steady_max_seconds": 0.8027044067382813,
      "steady_steps_per_second": 5115.383853133022,
      "steady_samples_steps_per_second": 5238153.065608215
    },
    "triton_eager": {
      "repeat_seconds": [
        0.8206735229492188,
        0.8209182739257812,
        0.8206786499023437,
        0.8191702880859375,
        0.8228710327148437,
        0.8181217041015625,
        0.8129679565429687
      ],
      "steady_median_seconds": 0.8206735229492188,
      "steady_min_seconds": 0.8129679565429687,
      "steady_max_seconds": 0.8228710327148437,
      "steady_steps_per_second": 4991.022477830627,
      "steady_samples_steps_per_second": 5110807.017298562
    },
    "triton_graph": {
      "repeat_seconds": [
        0.051966976165771485,
        0.051958782196044925,
        0.051958782196044925,
        0.051955711364746096,
        0.05195673751831055,
        0.051953662872314454,
        0.051955711364746096
      ],
      "steady_median_seconds": 0.05195673751831055,
      "steady_min_seconds": 0.051953662872314454,
      "steady_max_seconds": 0.051966976165771485,
      "steady_steps_per_second": 78834.81903682831,
      "steady_samples_steps_per_second": 80726854.69371219
    },
    "memory": {
      "memory_allocated_bytes": 116224,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 1024,
    "steps": 4096,
    "block": 64,
    "num_warps": 8,
    "graph_chunk": 16,
    "compile_seconds": 0.3020838610827923,
    "capture_seconds": 0.009442269802093506,
    "verification_steps": 16,
    "verification_relative_l2": 6.969064381418801e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.8027975463867187,
        0.7996160278320312,
        0.8010444946289063,
        0.8007843627929687,
        0.8016087036132813,
        0.8003624877929687,
        0.8013536987304688
      ],
      "steady_median_seconds": 0.8010444946289063,
      "steady_min_seconds": 0.7996160278320312,
      "steady_max_seconds": 0.8027975463867187,
      "steady_steps_per_second": 5113.323950746983,
      "steady_samples_steps_per_second": 5236043.725564911
    },
    "triton_eager": {
      "repeat_seconds": [
        0.8231024780273437,
        0.8181432495117188,
        0.8189736938476563,
        0.8184227905273438,
        0.8202045288085937,
        0.8208895874023437,
        0.818693115234375
      ],
      "steady_median_seconds": 0.8189736938476563,
      "steady_min_seconds": 0.8181432495117188,
      "steady_max_seconds": 0.8231024780273437,
      "steady_steps_per_second": 5001.381644819875,
      "steady_samples_steps_per_second": 5121414.804295552
    },
    "triton_graph": {
      "repeat_seconds": [
        0.05219839859008789,
        0.052190208435058595,
        0.05218611145019531,
        0.05219839859008789,
        0.05219532775878906,
        0.05218611145019531,
        0.052188159942626954
      ],
      "steady_median_seconds": 0.052190208435058595,
      "steady_min_seconds": 0.05218611145019531,
      "steady_max_seconds": 0.05219839859008789,
      "steady_steps_per_second": 78482.15446575082,
      "steady_samples_steps_per_second": 80365726.17292884
    },
    "memory": {
      "memory_allocated_bytes": 116224,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 1024,
    "steps": 4096,
    "block": 128,
    "num_warps": 1,
    "graph_chunk": 16,
    "compile_seconds": 0.39019856601953506,
    "capture_seconds": 0.009486149996519089,
    "verification_steps": 16,
    "verification_relative_l2": 6.969064381418801e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.7974993896484375,
        0.799266845703125,
        0.7989862670898438,
        0.799978515625,
        0.8005867309570313,
        0.7992125244140625,
        0.7978506469726563
      ],
      "steady_median_seconds": 0.7992125244140625,
      "steady_min_seconds": 0.7974993896484375,
      "steady_max_seconds": 0.8005867309570313,
      "steady_steps_per_second": 5125.044809580475,
      "steady_samples_steps_per_second": 5248045.885010406
    },
    "triton_eager": {
      "repeat_seconds": [
        0.8209735717773438,
        0.8200263671875,
        0.8200120239257812,
        0.8205435180664062,
        0.8587591552734375,
        0.84653466796875,
        0.8230901489257813
      ],
      "steady_median_seconds": 0.8209735717773438,
      "steady_min_seconds": 0.8200120239257812,
      "steady_max_seconds": 0.8587591552734375,
      "steady_steps_per_second": 4989.198362539832,
      "steady_samples_steps_per_second": 5108939.123240788
    },
    "triton_graph": {
      "repeat_seconds": [
        0.05393100738525391,
        0.053920768737792966,
        0.05391974258422851,
        0.05391667175292969,
        0.05391667175292969,
        0.053918720245361325,
        0.05391769790649414
      ],
      "steady_median_seconds": 0.053918720245361325,
      "steady_min_seconds": 0.05391667175292969,
      "steady_max_seconds": 0.05393100738525391,
      "steady_steps_per_second": 75966.1946975157,
      "steady_samples_steps_per_second": 77789383.37025608
    },
    "memory": {
      "memory_allocated_bytes": 116224,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 1024,
    "steps": 4096,
    "block": 128,
    "num_warps": 2,
    "graph_chunk": 16,
    "compile_seconds": 0.3345944248139858,
    "capture_seconds": 0.011003408581018448,
    "verification_steps": 16,
    "verification_relative_l2": 6.969064381418801e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.8042864379882813,
        0.8028078002929687,
        0.8004782104492187,
        0.8039659423828125,
        0.8015994873046876,
        0.8222525634765625,
        0.8057200927734375
      ],
      "steady_median_seconds": 0.8039659423828125,
      "steady_min_seconds": 0.8004782104492187,
      "steady_max_seconds": 0.8222525634765625,
      "steady_steps_per_second": 5094.743177627876,
      "steady_samples_steps_per_second": 5217017.013890945
    },
    "triton_eager": {
      "repeat_seconds": [
        0.8159119262695312,
        0.8171878662109375,
        0.8129075317382812,
        0.8178329467773438,
        0.8136826782226563,
        0.8153385009765625,
        0.8171776123046876
      ],
      "steady_median_seconds": 0.8159119262695312,
      "steady_min_seconds": 0.8129075317382812,
      "steady_max_seconds": 0.8178329467773438,
      "steady_steps_per_second": 5020.149685429298,
      "steady_samples_steps_per_second": 5140633.277879601
    },
    "triton_graph": {
      "repeat_seconds": [
        0.05312102508544922,
        0.05312204742431641,
        0.053119998931884765,
        0.053116928100585936,
        0.05311590576171875,
        0.053116928100585936,
        0.05311385726928711
      ],
      "steady_median_seconds": 0.053116928100585936,
      "steady_min_seconds": 0.05311385726928711,
      "steady_max_seconds": 0.05312204742431641,
      "steady_steps_per_second": 77112.89312972933,
      "steady_samples_steps_per_second": 78963602.56484284
    },
    "memory": {
      "memory_allocated_bytes": 116224,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 1024,
    "steps": 4096,
    "block": 128,
    "num_warps": 4,
    "graph_chunk": 16,
    "compile_seconds": 0.31396424025297165,
    "capture_seconds": 0.009113192558288574,
    "verification_steps": 16,
    "verification_relative_l2": 6.969064381418801e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.8039556884765625,
        0.8013403930664063,
        0.8040478515625,
        0.8024729614257813,
        0.8029000244140625,
        0.797686767578125,
        0.7999048461914062
      ],
      "steady_median_seconds": 0.8024729614257813,
      "steady_min_seconds": 0.797686767578125,
      "steady_max_seconds": 0.8040478515625,
      "steady_steps_per_second": 5104.221820411863,
      "steady_samples_steps_per_second": 5226723.144101747
    },
    "triton_eager": {
      "repeat_seconds": [
        0.8319794921875,
        0.8281241455078125,
        0.8261990356445312,
        0.8259122924804687,
        0.8227993774414063,
        0.8201820068359374,
        0.825548828125
      ],
      "steady_median_seconds": 0.8259122924804687,
      "steady_min_seconds": 0.8201820068359374,
      "steady_max_seconds": 0.8319794921875,
      "steady_steps_per_second": 4959.36437475516,
      "steady_samples_steps_per_second": 5078389.119749283
    },
    "triton_graph": {
      "repeat_seconds": [
        0.05209695816040039,
        0.05207449722290039,
        0.05206630325317383,
        0.05206835174560547,
        0.05207040023803711,
        0.05206630325317383,
        0.052067329406738284
      ],
      "steady_median_seconds": 0.05206835174560547,
      "steady_min_seconds": 0.05206630325317383,
      "steady_max_seconds": 0.05209695816040039,
      "steady_steps_per_second": 78665.82794884992,
      "steady_samples_steps_per_second": 80553807.81962232
    },
    "memory": {
      "memory_allocated_bytes": 116224,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 1024,
    "steps": 4096,
    "block": 128,
    "num_warps": 8,
    "graph_chunk": 16,
    "compile_seconds": 0.3095371685922146,
    "capture_seconds": 0.010965414345264435,
    "verification_steps": 16,
    "verification_relative_l2": 6.969064381418801e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.8003071899414063,
        0.8016609497070313,
        0.8023162841796875,
        0.8002570190429688,
        0.8017930297851562,
        0.8023746337890625,
        0.8007208862304688
      ],
      "steady_median_seconds": 0.8016609497070313,
      "steady_min_seconds": 0.8002570190429688,
      "steady_max_seconds": 0.8023746337890625,
      "steady_steps_per_second": 5109.391946179856,
      "steady_samples_steps_per_second": 5232017.3528881725
    },
    "triton_eager": {
      "repeat_seconds": [
        0.8260587768554688,
        0.8193177490234375,
        0.8200304565429688,
        0.8294256591796875,
        0.8197877807617188,
        0.8264468383789062,
        0.8215879516601563
      ],
      "steady_median_seconds": 0.8215879516601563,
      "steady_min_seconds": 0.8193177490234375,
      "steady_max_seconds": 0.8294256591796875,
      "steady_steps_per_second": 4985.467461789507,
      "steady_samples_steps_per_second": 5105118.680872455
    },
    "triton_graph": {
      "repeat_seconds": [
        0.052171775817871094,
        0.05216153717041016,
        0.05215129470825195,
        0.05214924621582031,
        0.052152320861816405,
        0.05215027236938476,
        0.052147201538085934
      ],
      "steady_median_seconds": 0.05215129470825195,
      "steady_min_seconds": 0.052147201538085934,
      "steady_max_seconds": 0.052171775817871094,
      "steady_steps_per_second": 78540.71548777648,
      "steady_samples_steps_per_second": 80425692.65948312
    },
    "memory": {
      "memory_allocated_bytes": 116224,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 1024,
    "steps": 4096,
    "block": 256,
    "num_warps": 1,
    "graph_chunk": 16,
    "compile_seconds": 0.5237358324229717,
    "capture_seconds": 0.0090867318212986,
    "verification_steps": 16,
    "verification_relative_l2": 6.969064381418801e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.7947704467773438,
        0.80520703125,
        0.800927734375,
        0.7996713256835938,
        0.805876708984375,
        0.7984568481445312,
        0.8015953979492187
      ],
      "steady_median_seconds": 0.800927734375,
      "steady_min_seconds": 0.7947704467773438,
      "steady_max_seconds": 0.805876708984375,
      "steady_steps_per_second": 5114.069377552887,
      "steady_samples_steps_per_second": 5236807.042614156
    },
    "triton_eager": {
      "repeat_seconds": [
        0.8284334106445312,
        0.8235479125976563,
        0.8234956665039063,
        0.823804931640625,
        0.8305469360351563,
        0.8221409301757813,
        0.821897216796875
      ],
      "steady_median_seconds": 0.8235479125976563,
      "steady_min_seconds": 0.821897216796875,
      "steady_max_seconds": 0.8305469360351563,
      "steady_steps_per_second": 4973.6025522550235,
      "steady_samples_steps_per_second": 5092969.013509144
    },
    "triton_graph": {
      "repeat_seconds": [
        0.056683521270751956,
        0.05667839813232422,
        0.056677310943603516,
        0.05667942428588867,
        0.056676353454589844,
        0.0566743049621582,
        0.0566743049621582
      ],
      "steady_median_seconds": 0.056677310943603516,
      "steady_min_seconds": 0.0566743049621582,
      "steady_max_seconds": 0.056683521270751956,
      "steady_steps_per_second": 72268.77796082642,
      "steady_samples_steps_per_second": 74003228.63188626
    },
    "memory": {
      "memory_allocated_bytes": 116224,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 1024,
    "steps": 4096,
    "block": 256,
    "num_warps": 2,
    "graph_chunk": 16,
    "compile_seconds": 0.39327533915638924,
    "capture_seconds": 0.00914064422249794,
    "verification_steps": 16,
    "verification_relative_l2": 6.969064381418801e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.801924072265625,
        0.8000675659179688,
        0.7982100219726562,
        0.7972710571289062,
        0.799193115234375,
        0.7991132202148438,
        0.793955322265625
      ],
      "steady_median_seconds": 0.7991132202148438,
      "steady_min_seconds": 0.793955322265625,
      "steady_max_seconds": 0.801924072265625,
      "steady_steps_per_second": 5125.681688633282,
      "steady_samples_steps_per_second": 5248698.0491604805
    },
    "triton_eager": {
      "repeat_seconds": [
        0.827578369140625,
        0.824216552734375,
        0.8241612548828126,
        0.8240824584960937,
        0.8264417114257813,
        0.8283197631835938,
        0.8263679809570312
      ],
      "steady_median_seconds": 0.8263679809570312,
      "steady_min_seconds": 0.8240824584960937,
      "steady_max_seconds": 0.8283197631835938,
      "steady_steps_per_second": 4956.629606166917,
      "steady_samples_steps_per_second": 5075588.716714923
    },
    "triton_graph": {
      "repeat_seconds": [
        0.05433651351928711,
        0.05433446502685547,
        0.054338558197021485,
        0.054333438873291014,
        0.05433446502685547,
        0.054335487365722655,
        0.05433139038085937
      ],
      "steady_median_seconds": 0.05433446502685547,
      "steady_min_seconds": 0.05433139038085937,
      "steady_max_seconds": 0.054338558197021485,
      "steady_steps_per_second": 75384.93289619217,
      "steady_samples_steps_per_second": 77194171.28570078
    },
    "memory": {
      "memory_allocated_bytes": 116224,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 1024,
    "steps": 4096,
    "block": 256,
    "num_warps": 4,
    "graph_chunk": 16,
    "compile_seconds": 0.3418841026723385,
    "capture_seconds": 0.010046485811471939,
    "verification_steps": 16,
    "verification_relative_l2": 6.969064381418801e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.7999027099609375,
        0.8124149780273437,
        0.8002734375,
        0.79442431640625,
        0.7934259033203125,
        0.7944970092773438,
        0.7935426635742188
      ],
      "steady_median_seconds": 0.7944970092773438,
      "steady_min_seconds": 0.7934259033203125,
      "steady_max_seconds": 0.8124149780273437,
      "steady_steps_per_second": 5155.463081888285,
      "steady_samples_steps_per_second": 5279194.195853604
    },
    "triton_eager": {
      "repeat_seconds": [
        0.8201195678710937,
        0.8190208129882812,
        0.8202936401367188,
        0.8197232666015625,
        0.8198656005859375,
        0.8193894653320313,
        0.8218297119140625
      ],
      "steady_median_seconds": 0.8198656005859375,
      "steady_min_seconds": 0.8190208129882812,
      "steady_max_seconds": 0.8218297119140625,
      "steady_steps_per_second": 4995.940794531069,
      "steady_samples_steps_per_second": 5115843.373599814
    },
    "triton_graph": {
      "repeat_seconds": [
        0.05336166381835938,
        0.05335244750976562,
        0.05338009643554688,
        0.053359615325927735,
        0.053356544494628906,
        0.05335347366333008,
        0.053354496002197264
      ],
      "steady_median_seconds": 0.053356544494628906,
      "steady_min_seconds": 0.05335244750976562,
      "steady_max_seconds": 0.05338009643554688,
      "steady_steps_per_second": 76766.59046787261,
      "steady_samples_steps_per_second": 78608988.63910155
    },
    "memory": {
      "memory_allocated_bytes": 116224,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 1024,
    "steps": 4096,
    "block": 256,
    "num_warps": 8,
    "graph_chunk": 16,
    "compile_seconds": 0.2888114005327225,
    "capture_seconds": 0.01062890887260437,
    "verification_steps": 16,
    "verification_relative_l2": 6.969064381418801e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.798065673828125,
        0.7987384033203125,
        0.7967713012695312,
        0.797348876953125,
        0.7981383666992188,
        0.7982254028320312,
        0.7922135009765625
      ],
      "steady_median_seconds": 0.798065673828125,
      "steady_min_seconds": 0.7922135009765625,
      "steady_max_seconds": 0.7987384033203125,
      "steady_steps_per_second": 5132.409692992425,
      "steady_samples_steps_per_second": 5255587.525624244
    },
    "triton_eager": {
      "repeat_seconds": [
        0.8182947998046874,
        0.817133544921875,
        0.8178093872070312,
        0.8197601318359375,
        0.8229560546875,
        0.8215101318359375,
        0.8212244262695313
      ],
      "steady_median_seconds": 0.8197601318359375,
      "steady_min_seconds": 0.817133544921875,
      "steady_max_seconds": 0.8229560546875,
      "steady_steps_per_second": 4996.58356259237,
      "steady_samples_steps_per_second": 5116501.568094587
    },
    "triton_graph": {
      "repeat_seconds": [
        0.052383743286132815,
        0.05237760162353516,
        0.052370433807373044,
        0.05237760162353516,
        0.0523765754699707,
        0.05237247848510742,
        0.05237452697753906
      ],
      "steady_median_seconds": 0.0523765754699707,
      "steady_min_seconds": 0.052370433807373044,
      "steady_max_seconds": 0.052383743286132815,
      "steady_steps_per_second": 78202.89820873032,
      "steady_samples_steps_per_second": 80079767.76573984
    },
    "memory": {
      "memory_allocated_bytes": 116224,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 1024,
    "steps": 4096,
    "block": 512,
    "num_warps": 1,
    "graph_chunk": 16,
    "compile_seconds": 0.8033934161067009,
    "capture_seconds": 0.008671268820762634,
    "verification_steps": 16,
    "verification_relative_l2": 6.969064381418801e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.8008632202148438,
        0.8016824340820312,
        0.8013025512695312,
        0.8004628295898437,
        0.8016107177734375,
        0.8011182250976563,
        0.7969290161132813
      ],
      "steady_median_seconds": 0.8011182250976563,
      "steady_min_seconds": 0.7969290161132813,
      "steady_max_seconds": 0.8016824340820312,
      "steady_steps_per_second": 5112.853348830877,
      "steady_samples_steps_per_second": 5235561.829202818
    },
    "triton_eager": {
      "repeat_seconds": [
        0.81900341796875,
        0.8222218017578125,
        0.818798583984375,
        0.8203131103515625,
        0.8186204223632813,
        0.8188047485351563,
        0.8189183959960937
      ],
      "steady_median_seconds": 0.8189183959960937,
      "steady_min_seconds": 0.8186204223632813,
      "steady_max_seconds": 0.8222218017578125,
      "steady_steps_per_second": 5001.719365478191,
      "steady_samples_steps_per_second": 5121760.630249668
    },
    "triton_graph": {
      "repeat_seconds": [
        0.06402457427978515,
        0.0640153579711914,
        0.06401331329345702,
        0.06401433563232421,
        0.06401229095458984,
        0.06401331329345702,
        0.06401126098632813
      ],
      "steady_median_seconds": 0.06401331329345702,
      "steady_min_seconds": 0.06401126098632813,
      "steady_max_seconds": 0.06402457427978515,
      "steady_steps_per_second": 63986.6894754011,
      "steady_samples_steps_per_second": 65522370.02281073
    },
    "memory": {
      "memory_allocated_bytes": 116224,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 1024,
    "steps": 4096,
    "block": 512,
    "num_warps": 2,
    "graph_chunk": 16,
    "compile_seconds": 0.5299510881304741,
    "capture_seconds": 0.0114884153008461,
    "verification_steps": 16,
    "verification_relative_l2": 6.969064381418801e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.79942041015625,
        0.799699951171875,
        0.7979489135742187,
        0.7994337768554688,
        0.7974727783203125,
        0.7956182861328125,
        0.7926527709960938
      ],
      "steady_median_seconds": 0.7979489135742187,
      "steady_min_seconds": 0.7926527709960938,
      "steady_max_seconds": 0.799699951171875,
      "steady_steps_per_second": 5133.160695279302,
      "steady_samples_steps_per_second": 5256356.551966005
    },
    "triton_eager": {
      "repeat_seconds": [
        0.824753173828125,
        0.824258544921875,
        0.8248842163085938,
        0.825501708984375,
        0.8243363647460937,
        0.8255784912109375,
        0.8253829345703125
      ],
      "steady_median_seconds": 0.8248842163085938,
      "steady_min_seconds": 0.824258544921875,
      "steady_max_seconds": 0.8255784912109375,
      "steady_steps_per_second": 4965.5453686941,
      "steady_samples_steps_per_second": 5084718.457542758
    },
    "triton_graph": {
      "repeat_seconds": [
        0.05817446517944336,
        0.058162174224853515,
        0.058157054901123044,
        0.05815500640869141,
        0.05815193557739258,
        0.05815091323852539,
        0.05814681625366211
      ],
      "steady_median_seconds": 0.05815500640869141,
      "steady_min_seconds": 0.05814681625366211,
      "steady_max_seconds": 0.05817446517944336,
      "steady_steps_per_second": 70432.4572026501,
      "steady_samples_steps_per_second": 72122836.1755137
    },
    "memory": {
      "memory_allocated_bytes": 116224,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 1024,
    "steps": 4096,
    "block": 512,
    "num_warps": 4,
    "graph_chunk": 16,
    "compile_seconds": 0.39302883669734,
    "capture_seconds": 0.010812327265739441,
    "verification_steps": 16,
    "verification_relative_l2": 6.969064381418801e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.8000205078125,
        0.799931396484375,
        0.8001116333007813,
        0.8000409545898437,
        0.800311279296875,
        0.7999539184570312,
        0.79929345703125
      ],
      "steady_median_seconds": 0.8000205078125,
      "steady_min_seconds": 0.79929345703125,
      "steady_max_seconds": 0.800311279296875,
      "steady_steps_per_second": 5119.868753364477,
      "steady_samples_steps_per_second": 5242745.603445224
    },
    "triton_eager": {
      "repeat_seconds": [
        0.8282767333984375,
        0.826661865234375,
        0.826871826171875,
        0.827125732421875,
        0.8269055786132813,
        0.835093505859375,
        0.8297943115234375
      ],
      "steady_median_seconds": 0.827125732421875,
      "steady_min_seconds": 0.826661865234375,
      "steady_max_seconds": 0.835093505859375,
      "steady_steps_per_second": 4952.088708456283,
      "steady_samples_steps_per_second": 5070938.837459234
    },
    "triton_graph": {
      "repeat_seconds": [
        0.05502873611450195,
        0.055021568298339846,
        0.05502566528320312,
        0.05502054214477539,
        0.05502054214477539,
        0.05502566528320312,
        0.05502361679077149
      ],
      "steady_median_seconds": 0.05502361679077149,
      "steady_min_seconds": 0.05502054214477539,
      "steady_max_seconds": 0.05502873611450195,
      "steady_steps_per_second": 74440.76269241133,
      "steady_samples_steps_per_second": 76227340.9970292
    },
    "memory": {
      "memory_allocated_bytes": 116224,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 1024,
    "steps": 4096,
    "block": 512,
    "num_warps": 8,
    "graph_chunk": 16,
    "compile_seconds": 0.34176285192370415,
    "capture_seconds": 0.010428950190544128,
    "verification_steps": 16,
    "verification_relative_l2": 6.969064381418801e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.7950489501953125,
        0.7935283203125,
        0.7936215209960937,
        0.79320166015625,
        0.7930224609375,
        0.79419287109375,
        0.7930419311523438
      ],
      "steady_median_seconds": 0.7935283203125,
      "steady_min_seconds": 0.7930224609375,
      "steady_max_seconds": 0.7950489501953125,
      "steady_steps_per_second": 5161.7565437197645,
      "steady_samples_steps_per_second": 5285638.700769039
    },
    "triton_eager": {
      "repeat_seconds": [
        0.8217354125976563,
        0.8228311157226562,
        0.8229631958007813,
        0.8220006103515625,
        0.821718017578125,
        0.8208701171875,
        0.8208660278320312
      ],
      "steady_median_seconds": 0.8217354125976563,
      "steady_min_seconds": 0.8208660278320312,
      "steady_max_seconds": 0.8229631958007813,
      "steady_steps_per_second": 4984.572816512547,
      "steady_samples_steps_per_second": 5104202.564108849
    },
    "triton_graph": {
      "repeat_seconds": [
        0.05462422561645508,
        0.054525951385498046,
        0.05453107070922852,
        0.054523902893066405,
        0.05453107070922852,
        0.05452492904663086,
        0.05452799987792969
      ],
      "steady_median_seconds": 0.05452799987792969,
      "steady_min_seconds": 0.054523902893066405,
      "steady_max_seconds": 0.05462422561645508,
      "steady_steps_per_second": 75117.37106018195,
      "steady_samples_steps_per_second": 76920187.96562631
    },
    "memory": {
      "memory_allocated_bytes": 116224,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 2048,
    "steps": 4096,
    "block": 64,
    "num_warps": 1,
    "graph_chunk": 16,
    "compile_seconds": 0.33110398426651955,
    "capture_seconds": 0.009026855230331421,
    "verification_steps": 16,
    "verification_relative_l2": 8.492373698461077e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.80393115234375,
        0.8057988891601563,
        0.8047124633789062,
        0.8058101806640625,
        0.8094832763671875,
        0.803800048828125,
        0.8048507080078126
      ],
      "steady_median_seconds": 0.8048507080078126,
      "steady_min_seconds": 0.803800048828125,
      "steady_max_seconds": 0.8094832763671875,
      "steady_steps_per_second": 5089.142569233151,
      "steady_samples_steps_per_second": 10422563.981789494
    },
    "triton_eager": {
      "repeat_seconds": [
        0.8228915405273437,
        0.8232182006835937,
        0.8251832275390625,
        0.8242514038085937,
        0.8257003784179687,
        0.8240025634765625,
        0.8258099365234375
      ],
      "steady_median_seconds": 0.8242514038085937,
      "steady_min_seconds": 0.8228915405273437,
      "steady_max_seconds": 0.8258099365234375,
      "steady_steps_per_second": 4969.357626900889,
      "steady_samples_steps_per_second": 10177244.41989302
    },
    "triton_graph": {
      "repeat_seconds": [
        0.06682624053955079,
        0.06681798553466797,
        0.06681088256835938,
        0.06681394958496094,
        0.06681088256835938,
        0.06681804656982422,
        0.06681190490722656
      ],
      "steady_median_seconds": 0.06681394958496094,
      "steady_min_seconds": 0.06681088256835938,
      "steady_max_seconds": 0.06682624053955079,
      "steady_steps_per_second": 61304.563275241,
      "steady_samples_steps_per_second": 125551745.58769357
    },
    "memory": {
      "memory_allocated_bytes": 230912,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 2048,
    "steps": 4096,
    "block": 64,
    "num_warps": 2,
    "graph_chunk": 16,
    "compile_seconds": 0.3029254972934723,
    "capture_seconds": 0.009186897426843643,
    "verification_steps": 16,
    "verification_relative_l2": 8.492373698461077e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.8022640380859375,
        0.8012113647460938,
        0.8012994384765625,
        0.7989462890625,
        0.799963134765625,
        0.7995125732421875,
        0.7998146362304688
      ],
      "steady_median_seconds": 0.799963134765625,
      "steady_min_seconds": 0.7989462890625,
      "steady_max_seconds": 0.8022640380859375,
      "steady_steps_per_second": 5120.235948372865,
      "steady_samples_steps_per_second": 10486243.222267628
    },
    "triton_eager": {
      "repeat_seconds": [
        0.8262993774414062,
        0.8219299926757813,
        0.826756103515625,
        0.8268789672851562,
        0.8238776245117188,
        0.8235847778320312,
        0.82404248046875
      ],
      "steady_median_seconds": 0.82404248046875,
      "steady_min_seconds": 0.8219299926757813,
      "steady_max_seconds": 0.8268789672851562,
      "steady_steps_per_second": 4970.617531355936,
      "steady_samples_steps_per_second": 10179824.704216957
    },
    "triton_graph": {
      "repeat_seconds": [
        0.06613401794433593,
        0.06612992095947266,
        0.06613196563720704,
        0.06612684631347657,
        0.06612684631347657,
        0.06612582397460938,
        0.06613504028320312
      ],
      "steady_median_seconds": 0.06612992095947266,
      "steady_min_seconds": 0.06612582397460938,
      "steady_max_seconds": 0.06613504028320312,
      "steady_steps_per_second": 61938.6798074386,
      "steady_samples_steps_per_second": 126850416.24563426
    },
    "memory": {
      "memory_allocated_bytes": 230912,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 2048,
    "steps": 4096,
    "block": 64,
    "num_warps": 4,
    "graph_chunk": 16,
    "compile_seconds": 0.3125969357788563,
    "capture_seconds": 0.009512484073638916,
    "verification_steps": 16,
    "verification_relative_l2": 8.492373698461077e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.7978465576171875,
        0.7997869873046874,
        0.798540771484375,
        0.7998494873046875,
        0.799224853515625,
        0.80260400390625,
        0.8037498779296876
      ],
      "steady_median_seconds": 0.7997869873046874,
      "steady_min_seconds": 0.7978465576171875,
      "steady_max_seconds": 0.8037498779296876,
      "steady_steps_per_second": 5121.363644341946,
      "steady_samples_steps_per_second": 10488552.743612306
    },
    "triton_eager": {
      "repeat_seconds": [
        0.8223651733398437,
        0.8205987548828125,
        0.822846435546875,
        0.822972412109375,
        0.8211865844726562,
        0.8237670288085938,
        0.8176035766601563
      ],
      "steady_median_seconds": 0.8223651733398437,
      "steady_min_seconds": 0.8176035766601563,
      "steady_max_seconds": 0.8237670288085938,
      "steady_steps_per_second": 4980.755670093682,
      "steady_samples_steps_per_second": 10200587.61235186
    },
    "triton_graph": {
      "repeat_seconds": [
        0.06600601959228515,
        0.06598041534423828,
        0.06597734069824218,
        0.06597734069824218,
        0.06597734069824218,
        0.0659752960205078,
        0.0659752960205078
      ],
      "steady_median_seconds": 0.06597734069824218,
      "steady_min_seconds": 0.0659752960205078,
      "steady_max_seconds": 0.06600601959228515,
      "steady_steps_per_second": 62081.92019641569,
      "steady_samples_steps_per_second": 127143772.56225933
    },
    "memory": {
      "memory_allocated_bytes": 230912,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 2048,
    "steps": 4096,
    "block": 64,
    "num_warps": 8,
    "graph_chunk": 16,
    "compile_seconds": 0.30918750166893005,
    "capture_seconds": 0.010168902575969696,
    "verification_steps": 16,
    "verification_relative_l2": 8.492373698461077e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.8000665893554687,
        0.7996805419921875,
        0.8034611206054687,
        0.8043417358398437,
        0.8039434204101562,
        0.8070799560546875,
        0.8043171997070313
      ],
      "steady_median_seconds": 0.8039434204101562,
      "steady_min_seconds": 0.7996805419921875,
      "steady_max_seconds": 0.8070799560546875,
      "steady_steps_per_second": 5094.885903675038,
      "steady_samples_steps_per_second": 10434326.330726478
    },
    "triton_eager": {
      "repeat_seconds": [
        0.8231895141601563,
        0.8226058349609375,
        0.8229837036132812,
        0.8243333129882813,
        0.8224102172851563,
        0.8281292724609375,
        0.8253071899414063
      ],
      "steady_median_seconds": 0.8231895141601563,
      "steady_min_seconds": 0.8224102172851563,
      "steady_max_seconds": 0.8281292724609375,
      "steady_steps_per_second": 4975.767948379259,
      "steady_samples_steps_per_second": 10190372.758280722
    },
    "triton_graph": {
      "repeat_seconds": [
        0.06631629180908204,
        0.06630809783935547,
        0.06630707550048828,
        0.06630400085449219,
        0.066302978515625,
        0.06630502319335937,
        0.066302978515625
      ],
      "steady_median_seconds": 0.06630502319335937,
      "steady_min_seconds": 0.066302978515625,
      "steady_max_seconds": 0.06631629180908204,
      "steady_steps_per_second": 61775.10847187556,
      "steady_samples_steps_per_second": 126515422.15040115
    },
    "memory": {
      "memory_allocated_bytes": 230912,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 2048,
    "steps": 4096,
    "block": 128,
    "num_warps": 1,
    "graph_chunk": 16,
    "compile_seconds": 0.39814211800694466,
    "capture_seconds": 0.009022735059261322,
    "verification_steps": 16,
    "verification_relative_l2": 8.492373698461077e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.8128440551757813,
        0.8052674560546875,
        0.8052367553710937,
        0.8054671630859375,
        0.8055664672851562,
        0.8060855102539063,
        0.8060047607421875
      ],
      "steady_median_seconds": 0.8055664672851562,
      "steady_min_seconds": 0.8052367553710937,
      "steady_max_seconds": 0.8128440551757813,
      "steady_steps_per_second": 5084.620780956723,
      "steady_samples_steps_per_second": 10413303.359399369
    },
    "triton_eager": {
      "repeat_seconds": [
        0.8212080688476563,
        0.8305490112304688,
        0.817849365234375,
        0.8217108764648438,
        0.8182456665039063,
        0.8199444580078125,
        0.8143800048828125
      ],
      "steady_median_seconds": 0.8199444580078125,
      "steady_min_seconds": 0.8143800048828125,
      "steady_max_seconds": 0.8305490112304688,
      "steady_steps_per_second": 4995.460314411898,
      "steady_samples_steps_per_second": 10230702.723915568
    },
    "triton_graph": {
      "repeat_seconds": [
        0.06803353881835937,
        0.0680273895263672,
        0.06802124786376954,
        0.06801715087890625,
        0.06802534484863282,
        0.06801920318603516,
        0.06802329254150391
      ],
      "steady_median_seconds": 0.06802329254150391,
      "steady_min_seconds": 0.06801715087890625,
      "steady_max_seconds": 0.06803353881835937,
      "steady_steps_per_second": 60214.66834321282,
      "steady_samples_steps_per_second": 123319640.76689985
    },
    "memory": {
      "memory_allocated_bytes": 230912,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 2048,
    "steps": 4096,
    "block": 128,
    "num_warps": 2,
    "graph_chunk": 16,
    "compile_seconds": 0.3436676561832428,
    "capture_seconds": 0.010520949959754944,
    "verification_steps": 16,
    "verification_relative_l2": 8.492373698461077e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.8044779663085937,
        0.7992453002929687,
        0.8026265869140625,
        0.8020991821289063,
        0.8002273559570312,
        0.801112060546875,
        0.7984793090820312
      ],
      "steady_median_seconds": 0.801112060546875,
      "steady_min_seconds": 0.7984793090820312,
      "steady_max_seconds": 0.8044779663085937,
      "steady_steps_per_second": 5112.8926921957545,
      "steady_samples_steps_per_second": 10471204.233616905
    },
    "triton_eager": {
      "repeat_seconds": [
        0.8243128051757812,
        0.8235284423828125,
        0.823024658203125,
        0.8229652709960937,
        0.822940673828125,
        0.82235595703125,
        0.8237138061523438
      ],
      "steady_median_seconds": 0.823024658203125,
      "steady_min_seconds": 0.82235595703125,
      "steady_max_seconds": 0.8243128051757812,
      "steady_steps_per_second": 4976.76461959551,
      "steady_samples_steps_per_second": 10192413.940931605
    },
    "triton_graph": {
      "repeat_seconds": [
        0.06728498840332031,
        0.06727782440185547,
        0.06727577972412109,
        0.06727168273925781,
        0.06727474975585937,
        0.06727577972412109,
        0.06727986907958984
      ],
      "steady_median_seconds": 0.06727577972412109,
      "steady_min_seconds": 0.06727168273925781,
      "steady_max_seconds": 0.06728498840332031,
      "steady_steps_per_second": 60883.72393150307,
      "steady_samples_steps_per_second": 124689866.61171828
    },
    "memory": {
      "memory_allocated_bytes": 230912,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 2048,
    "steps": 4096,
    "block": 128,
    "num_warps": 4,
    "graph_chunk": 16,
    "compile_seconds": 0.2997354753315449,
    "capture_seconds": 0.011075075715780258,
    "verification_steps": 16,
    "verification_relative_l2": 8.492373698461077e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.8035911865234375,
        0.8064174194335938,
        0.8030771484375,
        0.805080078125,
        0.8053954467773438,
        0.803009521484375,
        0.8023142700195313
      ],
      "steady_median_seconds": 0.8035911865234375,
      "steady_min_seconds": 0.8023142700195313,
      "steady_max_seconds": 0.8064174194335938,
      "steady_steps_per_second": 5097.1191181432105,
      "steady_samples_steps_per_second": 10438899.953957295
    },
    "triton_eager": {
      "repeat_seconds": [
        0.8291891479492187,
        0.8273970947265625,
        0.82977587890625,
        0.8272957153320313,
        0.8275947265625,
        0.8266096801757813,
        0.8271001586914063
      ],
      "steady_median_seconds": 0.8273970947265625,
      "steady_min_seconds": 0.8266096801757813,
      "steady_max_seconds": 0.82977587890625,
      "steady_steps_per_second": 4950.464566658458,
      "steady_samples_steps_per_second": 10138551.432516523
    },
    "triton_graph": {
      "repeat_seconds": [
        0.0661770248413086,
        0.06615654754638672,
        0.06616985321044921,
        0.06616063690185547,
        0.06616371154785156,
        0.06615347290039063,
        0.06615347290039063
      ],
      "steady_median_seconds": 0.06616063690185547,
      "steady_min_seconds": 0.06615347290039063,
      "steady_max_seconds": 0.0661770248413086,
      "steady_steps_per_second": 61909.923963944304,
      "steady_samples_steps_per_second": 126791524.27815793
    },
    "memory": {
      "memory_allocated_bytes": 230912,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 2048,
    "steps": 4096,
    "block": 128,
    "num_warps": 8,
    "graph_chunk": 16,
    "compile_seconds": 0.29923830553889275,
    "capture_seconds": 0.010629784315824509,
    "verification_steps": 16,
    "verification_relative_l2": 8.492373698461077e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.8037089233398438,
        0.8021801147460937,
        0.8066437377929687,
        0.800837646484375,
        0.7987783813476562,
        0.7990650634765625,
        0.7976929321289062
      ],
      "steady_median_seconds": 0.800837646484375,
      "steady_min_seconds": 0.7976929321289062,
      "steady_max_seconds": 0.8066437377929687,
      "steady_steps_per_second": 5114.6446698418495,
      "steady_samples_steps_per_second": 10474792.283836108
    },
    "triton_eager": {
      "repeat_seconds": [
        0.823530517578125,
        0.8227952270507812,
        0.8246016235351562,
        0.8238868408203125,
        0.8168990478515625,
        0.8166051635742188,
        0.8175308837890625
      ],
      "steady_median_seconds": 0.8227952270507812,
      "steady_min_seconds": 0.8166051635742188,
      "steady_max_seconds": 0.8246016235351562,
      "steady_steps_per_second": 4978.152358371913,
      "steady_samples_steps_per_second": 10195256.029945677
    },
    "triton_graph": {
      "repeat_seconds": [
        0.06632345581054687,
        0.0663214111328125,
        0.06631731414794922,
        0.06632755279541015,
        0.06632038116455079,
        0.0663214111328125,
        0.06633881378173828
      ],
      "steady_median_seconds": 0.0663214111328125,
      "steady_min_seconds": 0.06631731414794922,
      "steady_max_seconds": 0.06633881378173828,
      "steady_steps_per_second": 61759.84391824114,
      "steady_samples_steps_per_second": 126484160.34455785
    },
    "memory": {
      "memory_allocated_bytes": 230912,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 2048,
    "steps": 4096,
    "block": 256,
    "num_warps": 1,
    "graph_chunk": 16,
    "compile_seconds": 0.5240073762834072,
    "capture_seconds": 0.008772768080234528,
    "verification_steps": 16,
    "verification_relative_l2": 8.492373698461077e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.8005621948242188,
        0.800584716796875,
        0.8015431518554688,
        0.801048583984375,
        0.8013742065429688,
        0.8018779907226562,
        0.8025261840820312
      ],
      "steady_median_seconds": 0.8013742065429688,
      "steady_min_seconds": 0.8005621948242188,
      "steady_max_seconds": 0.8025261840820312,
      "steady_steps_per_second": 5111.2201597673675,
      "steady_samples_steps_per_second": 10467778.887203569
    },
    "triton_eager": {
      "repeat_seconds": [
        0.8165611572265625,
        0.8183695068359375,
        0.81373388671875,
        0.81451416015625,
        0.816648193359375,
        0.8140574951171875,
        0.8157214965820313
      ],
      "steady_median_seconds": 0.8157214965820313,
      "steady_min_seconds": 0.81373388671875,
      "steady_max_seconds": 0.8183695068359375,
      "steady_steps_per_second": 5021.3216363216125,
      "steady_samples_steps_per_second": 10283666.711186662
    },
    "triton_graph": {
      "repeat_seconds": [
        0.07067443084716797,
        0.07066828918457031,
        0.07067033386230469,
        0.07067033386230469,
        0.07067135620117188,
        0.0706723861694336,
        0.07066726684570312
      ],
      "steady_median_seconds": 0.07067033386230469,
      "steady_min_seconds": 0.07066726684570312,
      "steady_max_seconds": 0.07067443084716797,
      "steady_steps_per_second": 57959.25639718524,
      "steady_samples_steps_per_second": 118700557.10143538
    },
    "memory": {
      "memory_allocated_bytes": 230912,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 2048,
    "steps": 4096,
    "block": 256,
    "num_warps": 2,
    "graph_chunk": 16,
    "compile_seconds": 0.47691117227077484,
    "capture_seconds": 0.00901389867067337,
    "verification_steps": 16,
    "verification_relative_l2": 8.492373698461077e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.8054610595703126,
        0.8101150512695312,
        0.8064163818359374,
        0.8036392822265624,
        0.8012308349609375,
        0.8063569946289062,
        0.8038072509765625
      ],
      "steady_median_seconds": 0.8054610595703126,
      "steady_min_seconds": 0.8012308349609375,
      "steady_max_seconds": 0.8101150512695312,
      "steady_steps_per_second": 5085.286186504266,
      "steady_samples_steps_per_second": 10414666.109960737
    },
    "triton_eager": {
      "repeat_seconds": [
        0.8170823974609375,
        0.8162385864257813,
        0.8162550048828126,
        0.816194580078125,
        0.815836181640625,
        0.8182681884765625,
        0.8189706420898437
      ],
      "steady_median_seconds": 0.8162550048828126,
      "steady_min_seconds": 0.815836181640625,
      "steady_max_seconds": 0.8189706420898437,
      "steady_steps_per_second": 5018.039675711454,
      "steady_samples_steps_per_second": 10276945.255857058
    },
    "triton_graph": {
      "repeat_seconds": [
        0.06852505493164063,
        0.06852095794677734,
        0.06851789093017578,
        0.06852301025390625,
        0.06852505493164063,
        0.06852198028564453,
        0.06851993560791016
      ],
      "steady_median_seconds": 0.06852198028564453,
      "steady_min_seconds": 0.06851789093017578,
      "steady_max_seconds": 0.06852505493164063,
      "steady_steps_per_second": 59776.43936916573,
      "steady_samples_steps_per_second": 122422147.82805142
    },
    "memory": {
      "memory_allocated_bytes": 230912,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 2048,
    "steps": 4096,
    "block": 256,
    "num_warps": 4,
    "graph_chunk": 16,
    "compile_seconds": 0.34040379896759987,
    "capture_seconds": 0.008938565850257874,
    "verification_steps": 16,
    "verification_relative_l2": 8.492373698461077e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.8048671264648437,
        0.8034129638671875,
        0.8041123657226562,
        0.8041021728515625,
        0.8052654418945312,
        0.8038174438476563,
        0.8062853393554688
      ],
      "steady_median_seconds": 0.8041123657226562,
      "steady_min_seconds": 0.8034129638671875,
      "steady_max_seconds": 0.8062853393554688,
      "steady_steps_per_second": 5093.815459881558,
      "steady_samples_steps_per_second": 10432134.061837431
    },
    "triton_eager": {
      "repeat_seconds": [
        0.8235980834960938,
        0.82271240234375,
        0.8229498901367187,
        0.8228577270507812,
        0.8243753051757813,
        0.82370458984375,
        0.8275640258789062
      ],
      "steady_median_seconds": 0.8235980834960938,
      "steady_min_seconds": 0.82271240234375,
      "steady_max_seconds": 0.8275640258789062,
      "steady_steps_per_second": 4973.299576673222,
      "steady_samples_steps_per_second": 10185317.533026759
    },
    "triton_graph": {
      "repeat_seconds": [
        0.06750924682617188,
        0.0675041275024414,
        0.06750208282470703,
        0.06750208282470703,
        0.06750822448730469,
        0.0675041275024414,
        0.06750514984130859
      ],
      "steady_median_seconds": 0.0675041275024414,
      "steady_min_seconds": 0.06750208282470703,
      "steady_max_seconds": 0.06750924682617188,
      "steady_steps_per_second": 60677.771145947496,
      "steady_samples_steps_per_second": 124268075.30690047
    },
    "memory": {
      "memory_allocated_bytes": 230912,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 2048,
    "steps": 4096,
    "block": 256,
    "num_warps": 8,
    "graph_chunk": 16,
    "compile_seconds": 0.3014680743217468,
    "capture_seconds": 0.009030811488628387,
    "verification_steps": 16,
    "verification_relative_l2": 8.492373698461077e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.8038850708007812,
        0.8042485961914062,
        0.8024658203125,
        0.8016056518554687,
        0.801071044921875,
        0.7995299682617187,
        0.7955670776367187
      ],
      "steady_median_seconds": 0.8016056518554687,
      "steady_min_seconds": 0.7955670776367187,
      "steady_max_seconds": 0.8042485961914062,
      "steady_steps_per_second": 5109.744411755368,
      "steady_samples_steps_per_second": 10464756.555274993
    },
    "triton_eager": {
      "repeat_seconds": [
        0.8219627685546875,
        0.8162232055664063,
        0.8128675537109376,
        0.8123617553710938,
        0.812284912109375,
        0.81166748046875,
        0.8125767822265625
      ],
      "steady_median_seconds": 0.8125767822265625,
      "steady_min_seconds": 0.81166748046875,
      "steady_max_seconds": 0.8219627685546875,
      "steady_steps_per_second": 5040.754411880248,
      "steady_samples_steps_per_second": 10323465.035530748
    },
    "triton_graph": {
      "repeat_seconds": [
        0.06662553405761719,
        0.06661837005615234,
        0.06661529541015625,
        0.06662041473388672,
        0.06661939239501953,
        0.06663577270507813,
        0.06661734771728516
      ],
      "steady_median_seconds": 0.06661939239501953,
      "steady_min_seconds": 0.06661529541015625,
      "steady_max_seconds": 0.06663577270507813,
      "steady_steps_per_second": 61483.59888533323,
      "steady_samples_steps_per_second": 125918410.51716246
    },
    "memory": {
      "memory_allocated_bytes": 230912,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 2048,
    "steps": 4096,
    "block": 512,
    "num_warps": 1,
    "graph_chunk": 16,
    "compile_seconds": 0.79684928804636,
    "capture_seconds": 0.009467985481023788,
    "verification_steps": 16,
    "verification_relative_l2": 8.492373698461077e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.80112744140625,
        0.7999149780273438,
        0.8007987060546875,
        0.799909912109375,
        0.8007526245117188,
        0.7994429321289063,
        0.8002703247070313
      ],
      "steady_median_seconds": 0.8002703247070313,
      "steady_min_seconds": 0.7994429321289063,
      "steady_max_seconds": 0.80112744140625,
      "steady_steps_per_second": 5118.270506281103,
      "steady_samples_steps_per_second": 10482217.996863699
    },
    "triton_eager": {
      "repeat_seconds": [
        0.8280452880859375,
        0.8274647216796875,
        0.8263209228515624,
        0.8274431762695312,
        0.8267970581054688,
        0.815330322265625,
        0.8153815307617187
      ],
      "steady_median_seconds": 0.8267970581054688,
      "steady_min_seconds": 0.815330322265625,
      "steady_max_seconds": 0.8280452880859375,
      "steady_steps_per_second": 4954.057298396316,
      "steady_samples_steps_per_second": 10145909.347115654
    },
    "triton_graph": {
      "repeat_seconds": [
        0.07845069122314453,
        0.07844556427001953,
        0.07844966125488281,
        0.07844761657714844,
        0.0784434585571289,
        0.07844351959228515,
        0.07844249725341797
      ],
      "steady_median_seconds": 0.07844556427001953,
      "steady_min_seconds": 0.07844249725341797,
      "steady_max_seconds": 0.07845069122314453,
      "steady_steps_per_second": 52214.55206697285,
      "steady_samples_steps_per_second": 106935402.6331604
    },
    "memory": {
      "memory_allocated_bytes": 230912,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 2048,
    "steps": 4096,
    "block": 512,
    "num_warps": 2,
    "graph_chunk": 16,
    "compile_seconds": 0.5358962900936604,
    "capture_seconds": 0.009455651044845581,
    "verification_steps": 16,
    "verification_relative_l2": 8.492373698461077e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.8002958984375,
        0.8014192504882812,
        0.8013414306640625,
        0.7993968505859375,
        0.799857666015625,
        0.8003461303710937,
        0.8018329467773437
      ],
      "steady_median_seconds": 0.8003461303710937,
      "steady_min_seconds": 0.7993968505859375,
      "steady_max_seconds": 0.8018329467773437,
      "steady_steps_per_second": 5117.785723660364,
      "steady_samples_steps_per_second": 10481225.162056426
    },
    "triton_eager": {
      "repeat_seconds": [
        0.8221542358398437,
        0.8213104858398438,
        0.82570751953125,
        0.8227205200195312,
        0.8197550048828125,
        0.8140708618164062,
        0.8150067138671875
      ],
      "steady_median_seconds": 0.8213104858398438,
      "steady_min_seconds": 0.8140708618164062,
      "steady_max_seconds": 0.82570751953125,
      "steady_steps_per_second": 4987.151717430676,
      "steady_samples_steps_per_second": 10213686.717298025
    },
    "triton_graph": {
      "repeat_seconds": [
        0.07230976104736328,
        0.07230361938476562,
        0.07229747009277344,
        0.07229849243164063,
        0.07229542541503907,
        0.07229849243164063,
        0.07229542541503907
      ],
      "steady_median_seconds": 0.07229849243164063,
      "steady_min_seconds": 0.07229542541503907,
      "steady_max_seconds": 0.07230976104736328,
      "steady_steps_per_second": 56654.016733099,
      "steady_samples_steps_per_second": 116027426.26938675
    },
    "memory": {
      "memory_allocated_bytes": 230912,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 2048,
    "steps": 4096,
    "block": 512,
    "num_warps": 4,
    "graph_chunk": 16,
    "compile_seconds": 0.40382061526179314,
    "capture_seconds": 0.012580547481775284,
    "verification_steps": 16,
    "verification_relative_l2": 8.492373698461077e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.7991911010742188,
        0.79925146484375,
        0.7987814331054688,
        0.798539794921875,
        0.7995556030273437,
        0.798361572265625,
        0.7989391479492187
      ],
      "steady_median_seconds": 0.7989391479492187,
      "steady_min_seconds": 0.798361572265625,
      "steady_max_seconds": 0.7995556030273437,
      "steady_steps_per_second": 5126.798468336347,
      "steady_samples_steps_per_second": 10499683.263152838
    },
    "triton_eager": {
      "repeat_seconds": [
        0.830539794921875,
        0.8292771606445313,
        0.8282357788085938,
        0.834460693359375,
        0.8280780639648437,
        0.8330505981445312,
        0.83386572265625
      ],
      "steady_median_seconds": 0.830539794921875,
      "steady_min_seconds": 0.8280780639648437,
      "steady_max_seconds": 0.834460693359375,
      "steady_steps_per_second": 4931.732380608314,
      "steady_samples_steps_per_second": 10100187.915485827
    },
    "triton_graph": {
      "repeat_seconds": [
        0.0693381118774414,
        0.06931865692138672,
        0.06931558227539063,
        0.06931353759765625,
        0.06931763458251954,
        0.06931455993652344,
        0.0693207015991211
      ],
      "steady_median_seconds": 0.06931763458251954,
      "steady_min_seconds": 0.06931353759765625,
      "steady_max_seconds": 0.0693381118774414,
      "steady_steps_per_second": 59090.30255675986,
      "steady_samples_steps_per_second": 121016939.6362442
    },
    "memory": {
      "memory_allocated_bytes": 230912,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 2048,
    "steps": 4096,
    "block": 512,
    "num_warps": 8,
    "graph_chunk": 16,
    "compile_seconds": 0.3441580757498741,
    "capture_seconds": 0.008755851536989212,
    "verification_steps": 16,
    "verification_relative_l2": 8.492373698461077e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.803198974609375,
        0.8034539794921876,
        0.7986534423828126,
        0.8000880737304688,
        0.8022118530273438,
        0.8021053466796875,
        0.8009133911132813
      ],
      "steady_median_seconds": 0.8021053466796875,
      "steady_min_seconds": 0.7986534423828126,
      "steady_max_seconds": 0.8034539794921876,
      "steady_steps_per_second": 5106.561148052907,
      "steady_samples_steps_per_second": 10458237.231212353
    },
    "triton_eager": {
      "repeat_seconds": [
        0.8329041748046875,
        0.833554443359375,
        0.8326707153320313,
        0.8297420654296875,
        0.8333086547851563,
        0.8260300903320312,
        0.8226693115234375
      ],
      "steady_median_seconds": 0.8326707153320313,
      "steady_min_seconds": 0.8226693115234375,
      "steady_max_seconds": 0.833554443359375,
      "steady_steps_per_second": 4919.111390108996,
      "steady_samples_steps_per_second": 10074340.126943223
    },
    "triton_graph": {
      "repeat_seconds": [
        0.06892851257324219,
        0.06890598297119141,
        0.068890625,
        0.06892134094238281,
        0.06889574432373047,
        0.06889574432373047,
        0.06892031860351562
      ],
      "steady_median_seconds": 0.06890598297119141,
      "steady_min_seconds": 0.068890625,
      "steady_max_seconds": 0.06892851257324219,
      "steady_steps_per_second": 59443.31425781239,
      "steady_samples_steps_per_second": 121739907.59999977
    },
    "memory": {
      "memory_allocated_bytes": 230912,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 4096,
    "steps": 4096,
    "block": 64,
    "num_warps": 1,
    "graph_chunk": 16,
    "compile_seconds": 0.33085327595472336,
    "capture_seconds": 0.008587975054979324,
    "verification_steps": 16,
    "verification_relative_l2": 7.869346798757611e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.8070184936523438,
        0.8078182373046875,
        0.80858935546875,
        0.8077619018554687,
        0.807762939453125,
        0.8079493408203124,
        0.8075274047851563
      ],
      "steady_median_seconds": 0.807762939453125,
      "steady_min_seconds": 0.8070184936523438,
      "steady_max_seconds": 0.80858935546875,
      "steady_steps_per_second": 5070.7946600930845,
      "steady_samples_steps_per_second": 20769974.927741274
    },
    "triton_eager": {
      "repeat_seconds": [
        0.8166021118164063,
        0.8179885864257812,
        0.8179507446289063,
        0.8190012817382812,
        0.81897265625,
        0.813322265625,
        0.8187012939453125
      ],
      "steady_median_seconds": 0.8179885864257812,
      "steady_min_seconds": 0.813322265625,
      "steady_max_seconds": 0.8190012817382812,
      "steady_steps_per_second": 5007.404831768571,
      "steady_samples_steps_per_second": 20510330.190924067
    },
    "triton_graph": {
      "repeat_seconds": [
        0.07754649353027344,
        0.07755570983886718,
        0.07754752349853515,
        0.07754035186767579,
        0.07754137420654297,
        0.07753830718994141,
        0.07754444885253907
      ],
      "steady_median_seconds": 0.07754444885253907,
      "steady_min_seconds": 0.07753830718994141,
      "steady_max_seconds": 0.07755570983886718,
      "steady_steps_per_second": 52821.318103492886,
      "steady_samples_steps_per_second": 216356118.95190686
    },
    "memory": {
      "memory_allocated_bytes": 460288,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 4096,
    "steps": 4096,
    "block": 64,
    "num_warps": 2,
    "graph_chunk": 16,
    "compile_seconds": 0.2929849699139595,
    "capture_seconds": 0.008883979171514511,
    "verification_steps": 16,
    "verification_relative_l2": 7.869346798757611e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.8013875122070313,
        0.8013619384765625,
        0.80243505859375,
        0.8018032836914063,
        0.804284423828125,
        0.8038348999023438,
        0.80279345703125
      ],
      "steady_median_seconds": 0.80243505859375,
      "steady_min_seconds": 0.8013619384765625,
      "steady_max_seconds": 0.804284423828125,
      "steady_steps_per_second": 5104.462917133944,
      "steady_samples_steps_per_second": 20907880.108580634
    },
    "triton_eager": {
      "repeat_seconds": [
        0.8166994018554687,
        0.8166779174804687,
        0.8180828247070312,
        0.8179906616210938,
        0.816859130859375,
        0.8192880859375,
        0.8173322143554688
      ],
      "steady_median_seconds": 0.8173322143554688,
      "steady_min_seconds": 0.8166779174804687,
      "steady_max_seconds": 0.8192880859375,
      "steady_steps_per_second": 5011.426110532081,
      "steady_samples_steps_per_second": 20526801.348739404
    },
    "triton_graph": {
      "repeat_seconds": [
        0.0769454116821289,
        0.07693926239013672,
        0.07693414306640625,
        0.07696281433105469,
        0.07693824005126954,
        0.07693106842041016,
        0.07693926239013672
      ],
      "steady_median_seconds": 0.07693926239013672,
      "steady_min_seconds": 0.07693106842041016,
      "steady_max_seconds": 0.07696281433105469,
      "steady_steps_per_second": 53236.79838819314,
      "steady_samples_steps_per_second": 218057926.1980391
    },
    "memory": {
      "memory_allocated_bytes": 460288,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 4096,
    "steps": 4096,
    "block": 64,
    "num_warps": 4,
    "graph_chunk": 16,
    "compile_seconds": 0.30836353451013565,
    "capture_seconds": 0.010930035263299942,
    "verification_steps": 16,
    "verification_relative_l2": 7.869346798757611e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.7964539184570313,
        0.795894775390625,
        0.7959869384765625,
        0.7960432739257812,
        0.7989074096679688,
        0.7989391479492187,
        0.798688232421875
      ],
      "steady_median_seconds": 0.7964539184570313,
      "steady_min_seconds": 0.795894775390625,
      "steady_max_seconds": 0.7989391479492187,
      "steady_steps_per_second": 5142.795967323726,
      "steady_samples_steps_per_second": 21064892.28215798
    },
    "triton_eager": {
      "repeat_seconds": [
        0.8186275634765625,
        0.8146073608398438,
        0.815077392578125,
        0.8157767944335937,
        0.816964599609375,
        0.8146052856445313,
        0.8133928833007813
      ],
      "steady_median_seconds": 0.815077392578125,
      "steady_min_seconds": 0.8133928833007813,
      "steady_max_seconds": 0.8186275634765625,
      "steady_steps_per_second": 5025.289668560399,
      "steady_samples_steps_per_second": 20583586.482423395
    },
    "triton_graph": {
      "repeat_seconds": [
        0.07684095764160156,
        0.07682764434814453,
        0.07682969665527344,
        0.07682252502441406,
        0.07682355499267578,
        0.07682764434814453,
        0.07684198760986329
      ],
      "steady_median_seconds": 0.07682764434814453,
      "steady_min_seconds": 0.07682252502441406,
      "steady_max_seconds": 0.07684198760986329,
      "steady_steps_per_second": 53314.14277703183,
      "steady_samples_steps_per_second": 218374728.8147224
    },
    "memory": {
      "memory_allocated_bytes": 460288,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 4096,
    "steps": 4096,
    "block": 64,
    "num_warps": 8,
    "graph_chunk": 16,
    "compile_seconds": 0.3112224265933037,
    "capture_seconds": 0.008939020335674286,
    "verification_steps": 16,
    "verification_relative_l2": 7.869346798757611e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.8006246337890625,
        0.8016209716796875,
        0.8004515991210938,
        0.8028804931640625,
        0.803135498046875,
        0.802150390625,
        0.7985582275390625
      ],
      "steady_median_seconds": 0.8016209716796875,
      "steady_min_seconds": 0.7985582275390625,
      "steady_max_seconds": 0.803135498046875,
      "steady_steps_per_second": 5109.646759137789,
      "steady_samples_steps_per_second": 20929113.125428382
    },
    "triton_eager": {
      "repeat_seconds": [
        0.8212091064453125,
        0.8246077270507812,
        0.8213340454101562,
        0.8220088500976562,
        0.8207431640625,
        0.8193228759765625,
        0.8236687622070312
      ],
      "steady_median_seconds": 0.8213340454101562,
      "steady_min_seconds": 0.8193228759765625,
      "steady_max_seconds": 0.8246077270507812,
      "steady_steps_per_second": 4987.008663392917,
      "steady_samples_steps_per_second": 20426787.485257387
    },
    "triton_graph": {
      "repeat_seconds": [
        0.07709900665283204,
        0.07709798431396485,
        0.0770877456665039,
        0.07709286499023438,
        0.07709388732910157,
        0.07709388732910157,
        0.07709593963623047
      ],
      "steady_median_seconds": 0.07709388732910157,
      "steady_min_seconds": 0.0770877456665039,
      "steady_max_seconds": 0.07709900665283204,
      "steady_steps_per_second": 53130.02290979084,
      "steady_samples_steps_per_second": 217620573.83850327
    },
    "memory": {
      "memory_allocated_bytes": 460288,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 4096,
    "steps": 4096,
    "block": 128,
    "num_warps": 1,
    "graph_chunk": 16,
    "compile_seconds": 0.4028366915881634,
    "capture_seconds": 0.009190566837787628,
    "verification_steps": 16,
    "verification_relative_l2": 7.869346798757611e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.7997409057617187,
        0.8060938110351562,
        0.7965358276367187,
        0.7963494262695312,
        0.7983523559570312,
        0.7975557250976563,
        0.7965634765625
      ],
      "steady_median_seconds": 0.7975557250976563,
      "steady_min_seconds": 0.7963494262695312,
      "steady_max_seconds": 0.8060938110351562,
      "steady_steps_per_second": 5135.691301693643,
      "steady_samples_steps_per_second": 21035791.571737163
    },
    "triton_eager": {
      "repeat_seconds": [
        0.8220241088867187,
        0.8211681518554688,
        0.825461669921875,
        0.8236072998046875,
        0.8241663818359375,
        0.8240752563476562,
        0.8230205688476563
      ],
      "steady_median_seconds": 0.8236072998046875,
      "steady_min_seconds": 0.8211681518554688,
      "steady_max_seconds": 0.825461669921875,
      "steady_steps_per_second": 4973.243924588012,
      "steady_samples_steps_per_second": 20370407.1151125
    },
    "triton_graph": {
      "repeat_seconds": [
        0.07791513824462891,
        0.07790898895263672,
        0.07791001892089844,
        0.07791001892089844,
        0.07791513824462891,
        0.07791206359863281,
        0.07791718292236328
      ],
      "steady_median_seconds": 0.07791206359863281,
      "steady_min_seconds": 0.07790898895263672,
      "steady_max_seconds": 0.07791718292236328,
      "steady_steps_per_second": 52572.08974852356,
      "steady_samples_steps_per_second": 215335279.6099525
    },
    "memory": {
      "memory_allocated_bytes": 460288,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 4096,
    "steps": 4096,
    "block": 128,
    "num_warps": 2,
    "graph_chunk": 16,
    "compile_seconds": 0.33927204087376595,
    "capture_seconds": 0.009022019803524017,
    "verification_steps": 16,
    "verification_relative_l2": 7.869346798757611e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.8007567138671875,
        0.7981752319335937,
        0.7966556396484376,
        0.7993384399414063,
        0.8011049194335937,
        0.8006594848632812,
        0.7996190795898438
      ],
      "steady_median_seconds": 0.7996190795898438,
      "steady_min_seconds": 0.7966556396484376,
      "steady_max_seconds": 0.8011049194335937,
      "steady_steps_per_second": 5122.439051980851,
      "steady_samples_steps_per_second": 20981510.356913567
    },
    "triton_eager": {
      "repeat_seconds": [
        0.826218505859375,
        0.8253040771484375,
        0.824816650390625,
        0.8247675170898437,
        0.8270848388671875,
        0.8258734130859375,
        0.8251924438476562
      ],
      "steady_median_seconds": 0.8253040771484375,
      "steady_min_seconds": 0.8247675170898437,
      "steady_max_seconds": 0.8270848388671875,
      "steady_steps_per_second": 4963.0192233538455,
      "steady_samples_steps_per_second": 20328526.73885735
    },
    "triton_graph": {
      "repeat_seconds": [
        0.07714918518066406,
        0.07714201354980468,
        0.07713996887207031,
        0.07713996887207031,
        0.07713894653320312,
        0.0771379165649414,
        0.07713894653320312
      ],
      "steady_median_seconds": 0.07713996887207031,
      "steady_min_seconds": 0.0771379165649414,
      "steady_max_seconds": 0.07714918518066406,
      "steady_steps_per_second": 53098.28432511876,
      "steady_samples_steps_per_second": 217490572.59568644
    },
    "memory": {
      "memory_allocated_bytes": 460288,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 4096,
    "steps": 4096,
    "block": 128,
    "num_warps": 4,
    "graph_chunk": 16,
    "compile_seconds": 0.3080119490623474,
    "capture_seconds": 0.00928889587521553,
    "verification_steps": 16,
    "verification_relative_l2": 7.869346798757611e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.80445849609375,
        0.803641357421875,
        0.81426025390625,
        0.8174141235351563,
        0.8066273193359375,
        0.8028364868164063,
        0.7987138671875
      ],
      "steady_median_seconds": 0.80445849609375,
      "steady_min_seconds": 0.7987138671875,
      "steady_max_seconds": 0.8174141235351563,
      "steady_steps_per_second": 5091.623769143039,
      "steady_samples_steps_per_second": 20855290.958409887
    },
    "triton_eager": {
      "repeat_seconds": [
        0.8257208251953125,
        0.8246630249023438,
        0.8243394775390624,
        0.8242257690429687,
        0.8258488159179688,
        0.8239022216796875,
        0.82410498046875
      ],
      "steady_median_seconds": 0.8243394775390624,
      "steady_min_seconds": 0.8239022216796875,
      "steady_max_seconds": 0.8258488159179688,
      "steady_steps_per_second": 4968.826692891103,
      "steady_samples_steps_per_second": 20352314.13408196
    },
    "triton_graph": {
      "repeat_seconds": [
        0.07604326629638672,
        0.0760381088256836,
        0.07603814697265625,
        0.07603916931152344,
        0.07603814697265625,
        0.07604940795898438,
        0.07604121398925781
      ],
      "steady_median_seconds": 0.07603916931152344,
      "steady_min_seconds": 0.0760381088256836,
      "steady_max_seconds": 0.07604940795898438,
      "steady_steps_per_second": 53866.97457489541,
      "steady_samples_steps_per_second": 220639127.8587716
    },
    "memory": {
      "memory_allocated_bytes": 460288,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 4096,
    "steps": 4096,
    "block": 128,
    "num_warps": 8,
    "graph_chunk": 16,
    "compile_seconds": 0.3041198365390301,
    "capture_seconds": 0.011007159948348999,
    "verification_steps": 16,
    "verification_relative_l2": 7.869346798757611e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.8048721923828125,
        0.8030740356445313,
        0.8044503173828125,
        0.8065689697265624,
        0.8012850952148437,
        0.8080640258789062,
        0.8060466918945313
      ],
      "steady_median_seconds": 0.8048721923828125,
      "steady_min_seconds": 0.8012850952148437,
      "steady_max_seconds": 0.8080640258789062,
      "steady_steps_per_second": 5089.006725246466,
      "steady_samples_steps_per_second": 20844571.546609525
    },
    "triton_eager": {
      "repeat_seconds": [
        0.826260498046875,
        0.8266536865234375,
        0.8264857788085938,
        0.8251473999023438,
        0.8274452514648437,
        0.8256204833984375,
        0.8268021850585937
      ],
      "steady_median_seconds": 0.8264857788085938,
      "steady_min_seconds": 0.8251473999023438,
      "steady_max_seconds": 0.8274452514648437,
      "steady_steps_per_second": 4955.923144744871,
      "steady_samples_steps_per_second": 20299461.20087499
    },
    "triton_graph": {
      "repeat_seconds": [
        0.07625727844238281,
        0.07624089813232422,
        0.0762460174560547,
        0.07625113677978515,
        0.07623884582519531,
        0.07624908447265626,
        0.0762429428100586
      ],
      "steady_median_seconds": 0.0762460174560547,
      "steady_min_seconds": 0.07623884582519531,
      "steady_max_seconds": 0.07625727844238281,
      "steady_steps_per_second": 53720.83863082788,
      "steady_samples_steps_per_second": 220040555.031871
    },
    "memory": {
      "memory_allocated_bytes": 460288,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 4096,
    "steps": 4096,
    "block": 256,
    "num_warps": 1,
    "graph_chunk": 16,
    "compile_seconds": 0.5292431637644768,
    "capture_seconds": 0.008728772401809692,
    "verification_steps": 16,
    "verification_relative_l2": 7.869346798757611e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.8070390014648438,
        0.8084459228515625,
        0.806771728515625,
        0.805396484375,
        0.8049817504882812,
        0.8026234741210938,
        0.8045731811523438
      ],
      "steady_median_seconds": 0.805396484375,
      "steady_min_seconds": 0.8026234741210938,
      "steady_max_seconds": 0.8084459228515625,
      "steady_steps_per_second": 5085.6939153124795,
      "steady_samples_steps_per_second": 20831002.277119916
    },
    "triton_eager": {
      "repeat_seconds": [
        0.8209633178710938,
        0.8179066772460938,
        0.8146995239257813,
        0.8099778442382812,
        0.809871337890625,
        0.8129105834960938,
        0.8094668579101563
      ],
      "steady_median_seconds": 0.8129105834960938,
      "steady_min_seconds": 0.8094668579101563,
      "steady_max_seconds": 0.8209633178710938,
      "steady_steps_per_second": 5038.684552960654,
      "steady_samples_steps_per_second": 20638451.92892684
    },
    "triton_graph": {
      "repeat_seconds": [
        0.08079257965087891,
        0.0807720947265625,
        0.08078431701660156,
        0.08078131103515625,
        0.08078131103515625,
        0.08077516937255859,
        0.08077516937255859
      ],
      "steady_median_seconds": 0.08078131103515625,
      "steady_min_seconds": 0.0807720947265625,
      "steady_max_seconds": 0.08079257965087891,
      "steady_steps_per_second": 50704.79727937827,
      "steady_samples_steps_per_second": 207686849.6563334
    },
    "memory": {
      "memory_allocated_bytes": 460288,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 4096,
    "steps": 4096,
    "block": 256,
    "num_warps": 2,
    "graph_chunk": 16,
    "compile_seconds": 0.40651050955057144,
    "capture_seconds": 0.011015616357326508,
    "verification_steps": 16,
    "verification_relative_l2": 7.869346798757611e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.7955117797851563,
        0.7961098022460937,
        0.7968511962890625,
        0.7985674438476562,
        0.8008560791015625,
        0.8004915161132813,
        0.797454345703125
      ],
      "steady_median_seconds": 0.797454345703125,
      "steady_min_seconds": 0.7955117797851563,
      "steady_max_seconds": 0.8008560791015625,
      "steady_steps_per_second": 5136.344195840463,
      "steady_samples_steps_per_second": 21038465.826162536
    },
    "triton_eager": {
      "repeat_seconds": [
        0.8198410034179687,
        0.8200673217773438,
        0.8175564575195312,
        0.8091023559570313,
        0.8098068237304688,
        0.811298828125,
        0.8110008544921875
      ],
      "steady_median_seconds": 0.811298828125,
      "steady_min_seconds": 0.8091023559570313,
      "steady_max_seconds": 0.8200673217773438,
      "steady_steps_per_second": 5048.694584542051,
      "steady_samples_steps_per_second": 20679453.018284243
    },
    "triton_graph": {
      "repeat_seconds": [
        0.07847731018066406,
        0.07847628784179687,
        0.07846809387207031,
        0.07846707153320312,
        0.07848857879638672,
        0.0784711685180664,
        0.07846502685546874
      ],
      "steady_median_seconds": 0.0784711685180664,
      "steady_min_seconds": 0.07846502685546874,
      "steady_max_seconds": 0.07848857879638672,
      "steady_steps_per_second": 52197.51505365921,
      "steady_samples_steps_per_second": 213801021.65978813
    },
    "memory": {
      "memory_allocated_bytes": 460288,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 4096,
    "steps": 4096,
    "block": 256,
    "num_warps": 4,
    "graph_chunk": 16,
    "compile_seconds": 0.3468826375901699,
    "capture_seconds": 0.009023897349834442,
    "verification_steps": 16,
    "verification_relative_l2": 7.869346798757611e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.7974533081054688,
        0.7991336669921875,
        0.7969935302734374,
        0.7982356567382812,
        0.8002651977539063,
        0.8003881225585937,
        0.7983390502929687
      ],
      "steady_median_seconds": 0.7983390502929687,
      "steady_min_seconds": 0.7969935302734374,
      "steady_max_seconds": 0.8003881225585937,
      "steady_steps_per_second": 5130.652194073282,
      "steady_samples_steps_per_second": 21015151.386924163
    },
    "triton_eager": {
      "repeat_seconds": [
        0.8331765747070312,
        0.8363243408203125,
        0.833786865234375,
        0.8353546142578125,
        0.8342937622070312,
        0.8242073364257813,
        0.824975341796875
      ],
      "steady_median_seconds": 0.833786865234375,
      "steady_min_seconds": 0.8242073364257813,
      "steady_max_seconds": 0.8363243408203125,
      "steady_steps_per_second": 4912.526415067269,
      "steady_samples_steps_per_second": 20121708.196115535
    },
    "triton_graph": {
      "repeat_seconds": [
        0.07757721710205077,
        0.0775617904663086,
        0.07756800079345703,
        0.07755980682373047,
        0.07756594848632813,
        0.07757107543945313,
        0.07755980682373047
      ],
      "steady_median_seconds": 0.07756594848632813,
      "steady_min_seconds": 0.07755980682373047,
      "steady_max_seconds": 0.07757721710205077,
      "steady_steps_per_second": 52806.677155787846,
      "steady_samples_steps_per_second": 216296149.63010702
    },
    "memory": {
      "memory_allocated_bytes": 460288,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 4096,
    "steps": 4096,
    "block": 256,
    "num_warps": 8,
    "graph_chunk": 16,
    "compile_seconds": 0.30121785774827003,
    "capture_seconds": 0.00914989784359932,
    "verification_steps": 16,
    "verification_relative_l2": 7.869346798757611e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.8024514770507812,
        0.8043642578125,
        0.804400146484375,
        0.8033239135742187,
        0.8056852416992187,
        0.803156982421875,
        0.8058163452148438
      ],
      "steady_median_seconds": 0.8043642578125,
      "steady_min_seconds": 0.8024514770507812,
      "steady_max_seconds": 0.8058163452148438,
      "steady_steps_per_second": 5092.220297231048,
      "steady_samples_steps_per_second": 20857734.337458372
    },
    "triton_eager": {
      "repeat_seconds": [
        0.816236572265625,
        0.8152330322265625,
        0.8197744750976562,
        0.8147374267578125,
        0.8148909912109376,
        0.8069437255859375,
        0.8055572509765625
      ],
      "steady_median_seconds": 0.8148909912109376,
      "steady_min_seconds": 0.8055572509765625,
      "steady_max_seconds": 0.8197744750976562,
      "steady_steps_per_second": 5026.439173064481,
      "steady_samples_steps_per_second": 20588294.852872115
    },
    "triton_graph": {
      "repeat_seconds": [
        0.07667609405517578,
        0.07665459442138672,
        0.07665869140625,
        0.07665971374511718,
        0.07665353393554687,
        0.07665049743652344,
        0.07665049743652344
      ],
      "steady_median_seconds": 0.07665459442138672,
      "steady_min_seconds": 0.07665049743652344,
      "steady_max_seconds": 0.07667609405517578,
      "steady_steps_per_second": 53434.500970462526,
      "steady_samples_steps_per_second": 218867715.9750145
    },
    "memory": {
      "memory_allocated_bytes": 460288,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 4096,
    "steps": 4096,
    "block": 512,
    "num_warps": 1,
    "graph_chunk": 16,
    "compile_seconds": 0.8053513467311859,
    "capture_seconds": 0.00933869183063507,
    "verification_steps": 16,
    "verification_relative_l2": 7.869346798757611e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.7990241088867187,
        0.799393798828125,
        0.800121826171875,
        0.7990589599609375,
        0.7980943603515624,
        0.8006983642578125,
        0.7988920288085938
      ],
      "steady_median_seconds": 0.7990589599609375,
      "steady_min_seconds": 0.7980943603515624,
      "steady_max_seconds": 0.8006983642578125,
      "steady_steps_per_second": 5126.029749044095,
      "steady_samples_steps_per_second": 20996217.852084614
    },
    "triton_eager": {
      "repeat_seconds": [
        0.8350750732421875,
        0.8299427490234375,
        0.8294481811523438,
        0.8544860229492187,
        0.839930908203125,
        0.8361072387695313,
        0.8318095092773438
      ],
      "steady_median_seconds": 0.8350750732421875,
      "steady_min_seconds": 0.8294481811523438,
      "steady_max_seconds": 0.8544860229492187,
      "steady_steps_per_second": 4904.948227106383,
      "steady_samples_steps_per_second": 20090667.938227743
    },
    "triton_graph": {
      "repeat_seconds": [
        0.08864153289794922,
        0.08859954833984375,
        0.0885923843383789,
        0.08861286163330079,
        0.08861491394042968,
        0.08861593627929687,
        0.08863129425048828
      ],
      "steady_median_seconds": 0.08861491394042968,
      "steady_min_seconds": 0.0885923843383789,
      "steady_max_seconds": 0.08864153289794922,
      "steady_steps_per_second": 46222.46772990703,
      "steady_samples_steps_per_second": 189327227.8216992
    },
    "memory": {
      "memory_allocated_bytes": 460288,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 4096,
    "steps": 4096,
    "block": 512,
    "num_warps": 2,
    "graph_chunk": 16,
    "compile_seconds": 0.5224844478070736,
    "capture_seconds": 0.009109482169151306,
    "verification_steps": 16,
    "verification_relative_l2": 7.869346798757611e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.8003614501953125,
        0.8012933349609375,
        0.802608154296875,
        0.801268798828125,
        0.8017592163085937,
        0.8003502197265625,
        0.8000030517578125
      ],
      "steady_median_seconds": 0.801268798828125,
      "steady_min_seconds": 0.8000030517578125,
      "steady_max_seconds": 0.802608154296875,
      "steady_steps_per_second": 5111.89254591031,
      "steady_samples_steps_per_second": 20938311.86804863
    },
    "triton_eager": {
      "repeat_seconds": [
        0.8331519775390624,
        0.8289730834960938,
        0.8233901977539062,
        0.832733154296875,
        0.8234291381835938,
        0.8202854614257813,
        0.8131737670898438
      ],
      "steady_median_seconds": 0.8234291381835938,
      "steady_min_seconds": 0.8131737670898438,
      "steady_max_seconds": 0.8331519775390624,
      "steady_steps_per_second": 4974.319962778322,
      "steady_samples_steps_per_second": 20374814.56754001
    },
    "triton_graph": {
      "repeat_seconds": [
        0.08225791931152343,
        0.08228044891357422,
        0.08227839660644531,
        0.08227635192871094,
        0.0822824935913086,
        0.08224870300292969,
        0.08224460601806641
      ],
      "steady_median_seconds": 0.08227635192871094,
      "steady_min_seconds": 0.08224460601806641,
      "steady_max_seconds": 0.0822824935913086,
      "steady_steps_per_second": 49783.442070317054,
      "steady_samples_steps_per_second": 203912978.72001866
    },
    "memory": {
      "memory_allocated_bytes": 460288,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 4096,
    "steps": 4096,
    "block": 512,
    "num_warps": 4,
    "graph_chunk": 16,
    "compile_seconds": 0.4145245663821697,
    "capture_seconds": 0.01086023822426796,
    "verification_steps": 16,
    "verification_relative_l2": 7.869346798757611e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.7989176025390625,
        0.79939892578125,
        0.799583251953125,
        0.7992197265625,
        0.7996671752929687,
        0.7998904418945313,
        0.8009553833007812
      ],
      "steady_median_seconds": 0.799583251953125,
      "steady_min_seconds": 0.7989176025390625,
      "steady_max_seconds": 0.8009553833007812,
      "steady_steps_per_second": 5122.668577655658,
      "steady_samples_steps_per_second": 20982450.494077574
    },
    "triton_eager": {
      "repeat_seconds": [
        0.8367175903320313,
        0.8307353515625,
        0.8265963745117187,
        0.8319406127929687,
        0.8388577270507812,
        0.8276458740234375,
        0.8257576904296875
      ],
      "steady_median_seconds": 0.8307353515625,
      "steady_min_seconds": 0.8257576904296875,
      "steady_max_seconds": 0.8388577270507812,
      "steady_steps_per_second": 4930.571441670301,
      "steady_samples_steps_per_second": 20195620.625081554
    },
    "triton_graph": {
      "repeat_seconds": [
        0.07928214263916016,
        0.0792647705078125,
        0.07926374053955078,
        0.07925862121582031,
        0.07926988983154297,
        0.07930470275878906,
        0.07929241943359375
      ],
      "steady_median_seconds": 0.07926988983154297,
      "steady_min_seconds": 0.07925862121582031,
      "steady_max_seconds": 0.07930470275878906,
      "steady_steps_per_second": 51671.57427245629,
      "steady_samples_steps_per_second": 211646768.21998096
    },
    "memory": {
      "memory_allocated_bytes": 460288,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 4096,
    "steps": 4096,
    "block": 512,
    "num_warps": 8,
    "graph_chunk": 16,
    "compile_seconds": 0.35534003376960754,
    "capture_seconds": 0.010877992957830429,
    "verification_steps": 16,
    "verification_relative_l2": 7.869346798757611e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.8047308959960937,
        0.8042454833984375,
        0.8055162963867187,
        0.8039547119140625,
        0.8020869140625,
        0.8020305786132812,
        0.8040376586914062
      ],
      "steady_median_seconds": 0.8040376586914062,
      "steady_min_seconds": 0.8020305786132812,
      "steady_max_seconds": 0.8055162963867187,
      "steady_steps_per_second": 5094.288750935317,
      "steady_samples_steps_per_second": 20866206.723831058
    },
    "triton_eager": {
      "repeat_seconds": [
        0.8274227294921875,
        0.8290283813476562,
        0.8274370727539062,
        0.8281620483398437,
        0.8289403076171875,
        0.8286064453125,
        0.829749267578125
      ],
      "steady_median_seconds": 0.8286064453125,
      "steady_min_seconds": 0.8274227294921875,
      "steady_max_seconds": 0.829749267578125,
      "steady_steps_per_second": 4943.239366736084,
      "steady_samples_steps_per_second": 20247508.446151
    },
    "triton_graph": {
      "repeat_seconds": [
        0.0788623046875,
        0.07886233520507813,
        0.07885004425048828,
        0.07885311889648437,
        0.07885619354248047,
        0.07885721588134766,
        0.07885209655761719
      ],
      "steady_median_seconds": 0.07885619354248047,
      "steady_min_seconds": 0.07885004425048828,
      "steady_max_seconds": 0.07886233520507813,
      "steady_steps_per_second": 51942.65429250591,
      "steady_samples_steps_per_second": 212757111.9821042
    },
    "memory": {
      "memory_allocated_bytes": 460288,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 8192,
    "steps": 4096,
    "block": 64,
    "num_warps": 1,
    "graph_chunk": 16,
    "compile_seconds": 0.3528575114905834,
    "capture_seconds": 0.010490074753761292,
    "verification_steps": 16,
    "verification_relative_l2": 8.651608927600565e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.8210462646484376,
        0.7971102905273437,
        0.7957125244140625,
        0.7958978271484375,
        0.7977636108398437,
        0.7961170043945313,
        0.7950101318359375
      ],
      "steady_median_seconds": 0.7961170043945313,
      "steady_min_seconds": 0.7950101318359375,
      "steady_max_seconds": 0.8210462646484376,
      "steady_steps_per_second": 5144.972381434209,
      "steady_samples_steps_per_second": 42147613.74870904
    },
    "triton_eager": {
      "repeat_seconds": [
        0.8432763061523437,
        0.8336834716796875,
        0.8282623901367188,
        0.828548095703125,
        0.8278394775390625,
        0.82747802734375,
        0.82855322265625
      ],
      "steady_median_seconds": 0.828548095703125,
      "steady_min_seconds": 0.82747802734375,
      "steady_max_seconds": 0.8432763061523437,
      "steady_steps_per_second": 4943.587489057035,
      "steady_samples_steps_per_second": 40497868.71035523
    },
    "triton_graph": {
      "repeat_seconds": [
        0.11204300689697265,
        0.11207679748535156,
        0.11203584289550782,
        0.11202662658691406,
        0.11203584289550782,
        0.11206963348388672,
        0.11206755065917968
      ],
      "steady_median_seconds": 0.11204300689697265,
      "steady_min_seconds": 0.11202662658691406,
      "steady_max_seconds": 0.11207679748535156,
      "steady_steps_per_second": 36557.390893359465,
      "steady_samples_steps_per_second": 299478146.19840074
    },
    "memory": {
      "memory_allocated_bytes": 919040,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 8192,
    "steps": 4096,
    "block": 64,
    "num_warps": 2,
    "graph_chunk": 16,
    "compile_seconds": 0.3100328780710697,
    "capture_seconds": 0.009221401065587997,
    "verification_steps": 16,
    "verification_relative_l2": 8.651608927600565e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.800859130859375,
        0.799320068359375,
        0.797570068359375,
        0.7970867309570312,
        0.7982858276367187,
        0.797907958984375,
        0.7982151489257813
      ],
      "steady_median_seconds": 0.7982151489257813,
      "steady_min_seconds": 0.7970867309570312,
      "steady_max_seconds": 0.800859130859375,
      "steady_steps_per_second": 5131.448589408881,
      "steady_samples_steps_per_second": 42036826.844437554
    },
    "triton_eager": {
      "repeat_seconds": [
        0.8330885009765625,
        0.8307312622070312,
        0.83135693359375,
        0.8308971557617187,
        0.8329482421875,
        0.8323727416992187,
        0.83115625
      ],
      "steady_median_seconds": 0.83135693359375,
      "steady_min_seconds": 0.8307312622070312,
      "steady_max_seconds": 0.8330885009765625,
      "steady_steps_per_second": 4926.884993060691,
      "steady_samples_steps_per_second": 40361041.86315318
    },
    "triton_graph": {
      "repeat_seconds": [
        0.11138969421386719,
        0.11134054565429688,
        0.1113917465209961,
        0.11133849334716797,
        0.1114071044921875,
        0.11137126159667969,
        0.11135897827148437
      ],
      "steady_median_seconds": 0.11137126159667969,
      "steady_min_seconds": 0.11133849334716797,
      "steady_max_seconds": 0.1114071044921875,
      "steady_steps_per_second": 36777.889926696436,
      "steady_samples_steps_per_second": 301284474.2794972
    },
    "memory": {
      "memory_allocated_bytes": 919040,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 8192,
    "steps": 4096,
    "block": 64,
    "num_warps": 4,
    "graph_chunk": 16,
    "compile_seconds": 0.3001985289156437,
    "capture_seconds": 0.010672908276319504,
    "verification_steps": 16,
    "verification_relative_l2": 8.651608927600565e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.8066631469726563,
        0.806424560546875,
        0.806740966796875,
        0.80854833984375,
        0.8085360717773438,
        0.8070963134765625,
        0.8076144409179687
      ],
      "steady_median_seconds": 0.8070963134765625,
      "steady_min_seconds": 0.806424560546875,
      "steady_max_seconds": 0.80854833984375,
      "steady_steps_per_second": 5074.982912951869,
      "steady_samples_steps_per_second": 41574260.022901714
    },
    "triton_eager": {
      "repeat_seconds": [
        0.822518798828125,
        0.8215255126953125,
        0.8214989013671875,
        0.8220466918945313,
        0.823593994140625,
        0.8215132446289063,
        0.8156661987304688
      ],
      "steady_median_seconds": 0.8215255126953125,
      "steady_min_seconds": 0.8156661987304688,
      "steady_max_seconds": 0.823593994140625,
      "steady_steps_per_second": 4985.846375679297,
      "steady_samples_steps_per_second": 40844053.5095648
    },
    "triton_graph": {
      "repeat_seconds": [
        0.11118793487548828,
        0.11118080139160157,
        0.11117874908447266,
        0.11117670440673828,
        0.11117465972900391,
        0.111172607421875,
        0.11117465972900391
      ],
      "steady_median_seconds": 0.11117670440673828,
      "steady_min_seconds": 0.111172607421875,
      "steady_max_seconds": 0.11118793487548828,
      "steady_steps_per_second": 36842.25055830803,
      "steady_samples_steps_per_second": 301811716.57365936
    },
    "memory": {
      "memory_allocated_bytes": 919040,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 8192,
    "steps": 4096,
    "block": 64,
    "num_warps": 8,
    "graph_chunk": 16,
    "compile_seconds": 0.30457518622279167,
    "capture_seconds": 0.01080874353647232,
    "verification_steps": 16,
    "verification_relative_l2": 8.651608927600565e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.80580810546875,
        0.8056381225585938,
        0.8037796020507812,
        0.803293212890625,
        0.8039567260742188,
        0.8051097412109375,
        0.8032317504882812
      ],
      "steady_median_seconds": 0.8039567260742188,
      "steady_min_seconds": 0.8032317504882812,
      "steady_max_seconds": 0.80580810546875,
      "steady_steps_per_second": 5094.801582171066,
      "steady_samples_steps_per_second": 41736614.56114537
    },
    "triton_eager": {
      "repeat_seconds": [
        0.8267387084960938,
        0.8227501831054688,
        0.8233994140625,
        0.8232182006835937,
        0.82338916015625,
        0.8222730102539062,
        0.82043798828125
      ],
      "steady_median_seconds": 0.8232182006835937,
      "steady_min_seconds": 0.82043798828125,
      "steady_max_seconds": 0.8267387084960938,
      "steady_steps_per_second": 4975.594558767912,
      "steady_samples_steps_per_second": 40760070.62542673
    },
    "triton_graph": {
      "repeat_seconds": [
        0.11144499206542968,
        0.11143885040283204,
        0.11143987274169923,
        0.11146649932861329,
        0.11145113372802734,
        0.1114439697265625,
        0.11144089508056641
      ],
      "steady_median_seconds": 0.1114439697265625,
      "steady_min_seconds": 0.11143885040283204,
      "steady_max_seconds": 0.11146649932861329,
      "steady_steps_per_second": 36753.89534355151,
      "steady_samples_steps_per_second": 301087910.65437394
    },
    "memory": {
      "memory_allocated_bytes": 919040,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 8192,
    "steps": 4096,
    "block": 128,
    "num_warps": 1,
    "graph_chunk": 16,
    "compile_seconds": 0.4026562087237835,
    "capture_seconds": 0.009238772094249725,
    "verification_steps": 16,
    "verification_relative_l2": 8.651608927600565e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.8013148193359375,
        0.8056432495117187,
        0.807004150390625,
        0.804515869140625,
        0.8056790771484375,
        0.80652490234375,
        0.8061224975585938
      ],
      "steady_median_seconds": 0.8056790771484375,
      "steady_min_seconds": 0.8013148193359375,
      "steady_max_seconds": 0.807004150390625,
      "steady_steps_per_second": 5083.910102887477,
      "steady_samples_steps_per_second": 41647391.56285421
    },
    "triton_eager": {
      "repeat_seconds": [
        0.8213330078125,
        0.819683349609375,
        0.8215214233398438,
        0.8215009155273437,
        0.8213790283203125,
        0.8215162963867187,
        0.820379638671875
      ],
      "steady_median_seconds": 0.8213790283203125,
      "steady_min_seconds": 0.819683349609375,
      "steady_max_seconds": 0.8215214233398438,
      "steady_steps_per_second": 4986.735549331174,
      "steady_samples_steps_per_second": 40851337.62012098
    },
    "triton_graph": {
      "repeat_seconds": [
        0.11143577575683594,
        0.1114224624633789,
        0.11143065643310547,
        0.1114224624633789,
        0.11142348480224609,
        0.11142348480224609,
        0.11142144012451172
      ],
      "steady_median_seconds": 0.11142348480224609,
      "steady_min_seconds": 0.11142144012451172,
      "steady_max_seconds": 0.11143577575683594,
      "steady_steps_per_second": 36760.652453740455,
      "steady_samples_steps_per_second": 301143264.9010418
    },
    "memory": {
      "memory_allocated_bytes": 919040,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 8192,
    "steps": 4096,
    "block": 128,
    "num_warps": 2,
    "graph_chunk": 16,
    "compile_seconds": 0.34575172513723373,
    "capture_seconds": 0.009042773395776749,
    "verification_steps": 16,
    "verification_relative_l2": 8.651608927600565e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.7970355224609375,
        0.798213134765625,
        0.80019970703125,
        0.7968685913085938,
        0.7984998168945312,
        0.7981301879882813,
        0.7998638305664062
      ],
      "steady_median_seconds": 0.798213134765625,
      "steady_min_seconds": 0.7968685913085938,
      "steady_max_seconds": 0.80019970703125,
      "steady_steps_per_second": 5131.461537779237,
      "steady_samples_steps_per_second": 42036932.91748751
    },
    "triton_eager": {
      "repeat_seconds": [
        0.822920166015625,
        0.8272189331054688,
        0.823816162109375,
        0.8252149658203125,
        0.828521484375,
        0.8262389526367188,
        0.8262706909179688
      ],
      "steady_median_seconds": 0.8262389526367188,
      "steady_min_seconds": 0.822920166015625,
      "steady_max_seconds": 0.828521484375,
      "steady_steps_per_second": 4957.403650516259,
      "steady_samples_steps_per_second": 40611050.7050292
    },
    "triton_graph": {
      "repeat_seconds": [
        0.11064934539794923,
        0.11065036773681641,
        0.11065856170654297,
        0.11064217376708985,
        0.11063910675048828,
        0.11064412689208984,
        0.11064115142822266
      ],
      "steady_median_seconds": 0.11064412689208984,
      "steady_min_seconds": 0.11063910675048828,
      "steady_max_seconds": 0.11065856170654297,
      "steady_steps_per_second": 37019.58807081364,
      "steady_samples_steps_per_second": 303264465.47610533
    },
    "memory": {
      "memory_allocated_bytes": 919040,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 8192,
    "steps": 4096,
    "block": 128,
    "num_warps": 4,
    "graph_chunk": 16,
    "compile_seconds": 0.30661286786198616,
    "capture_seconds": 0.009415730834007263,
    "verification_steps": 16,
    "verification_relative_l2": 8.651608927600565e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.805393310546875,
        0.801934326171875,
        0.803590087890625,
        0.8045332641601562,
        0.8042403564453126,
        0.7991173095703125,
        0.7979345703125
      ],
      "steady_median_seconds": 0.803590087890625,
      "steady_min_seconds": 0.7979345703125,
      "steady_max_seconds": 0.805393310546875,
      "steady_steps_per_second": 5097.126086698941,
      "steady_samples_steps_per_second": 41755656.90223773
    },
    "triton_eager": {
      "repeat_seconds": [
        0.8241069946289062,
        0.8244715576171875,
        0.8235048828125,
        0.8230942993164062,
        0.8245473022460937,
        0.8238120727539062,
        0.822803466796875
      ],
      "steady_median_seconds": 0.8238120727539062,
      "steady_min_seconds": 0.822803466796875,
      "steady_max_seconds": 0.8245473022460937,
      "steady_steps_per_second": 4972.007737526299,
      "steady_samples_steps_per_second": 40730687.38581544
    },
    "triton_graph": {
      "repeat_seconds": [
        0.10955366516113281,
        0.10953215789794922,
        0.10951884460449218,
        0.10951165008544922,
        0.10950962829589844,
        0.10950860595703125,
        0.10950860595703125
      ],
      "steady_median_seconds": 0.10951165008544922,
      "steady_min_seconds": 0.10950860595703125,
      "steady_max_seconds": 0.10955366516113281,
      "steady_steps_per_second": 37402.413321358894,
      "steady_samples_steps_per_second": 306400569.92857206
    },
    "memory": {
      "memory_allocated_bytes": 919040,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 8192,
    "steps": 4096,
    "block": 128,
    "num_warps": 8,
    "graph_chunk": 16,
    "compile_seconds": 0.30250175669789314,
    "capture_seconds": 0.00936698168516159,
    "verification_steps": 16,
    "verification_relative_l2": 8.651608927600565e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.8014243774414063,
        0.8012523803710937,
        0.8022261962890626,
        0.8007649536132813,
        0.7992811279296875,
        0.7970509033203125,
        0.7951441650390625
      ],
      "steady_median_seconds": 0.8007649536132813,
      "steady_min_seconds": 0.7951441650390625,
      "steady_max_seconds": 0.8022261962890626,
      "steady_steps_per_second": 5115.108973635362,
      "steady_samples_steps_per_second": 41902972.71202089
    },
    "triton_eager": {
      "repeat_seconds": [
        0.8226939086914062,
        0.8276602783203125,
        0.8245309448242187,
        0.8242821044921875,
        0.8222392578125,
        0.8219934692382812,
        0.8238909301757813
      ],
      "steady_median_seconds": 0.8238909301757813,
      "steady_min_seconds": 0.8219934692382812,
      "steady_max_seconds": 0.8276602783203125,
      "steady_steps_per_second": 4971.531849641915,
      "steady_samples_steps_per_second": 40726788.91226657
    },
    "triton_graph": {
      "repeat_seconds": [
        0.10971135711669922,
        0.10970726776123046,
        0.10970623779296874,
        0.1097000961303711,
        0.10970829010009765,
        0.10971647644042969,
        0.1097031021118164
      ],
      "steady_median_seconds": 0.10970726776123046,
      "steady_min_seconds": 0.1097000961303711,
      "steady_max_seconds": 0.10971647644042969,
      "steady_steps_per_second": 37335.721539566846,
      "steady_samples_steps_per_second": 305854230.8521316
    },
    "memory": {
      "memory_allocated_bytes": 919040,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 8192,
    "steps": 4096,
    "block": 256,
    "num_warps": 1,
    "graph_chunk": 16,
    "compile_seconds": 0.5256333202123642,
    "capture_seconds": 0.008575521409511566,
    "verification_steps": 16,
    "verification_relative_l2": 8.651608927600565e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.8010997924804687,
        0.8035892333984375,
        0.802028564453125,
        0.8026900634765625,
        0.8044226684570313,
        0.805771240234375,
        0.8029470825195313
      ],
      "steady_median_seconds": 0.8029470825195313,
      "steady_min_seconds": 0.8010997924804687,
      "steady_max_seconds": 0.805771240234375,
      "steady_steps_per_second": 5101.207899214662,
      "steady_samples_steps_per_second": 41789095.11036651
    },
    "triton_eager": {
      "repeat_seconds": [
        0.822708251953125,
        0.8235089721679687,
        0.8212572021484374,
        0.820658203125,
        0.8217323608398438,
        0.8212039794921875,
        0.8204451293945313
      ],
      "steady_median_seconds": 0.8212572021484374,
      "steady_min_seconds": 0.8204451293945313,
      "steady_max_seconds": 0.8235089721679687,
      "steady_steps_per_second": 4987.475287016931,
      "steady_samples_steps_per_second": 40857397.5512427
    },
    "triton_graph": {
      "repeat_seconds": [
        0.11387391662597657,
        0.11388211059570312,
        0.11387596893310546,
        0.11386573028564453,
        0.11386265563964844,
        0.113870849609375,
        0.11387187194824219
      ],
      "steady_median_seconds": 0.11387187194824219,
      "steady_min_seconds": 0.11386265563964844,
      "steady_max_seconds": 0.11388211059570312,
      "steady_steps_per_second": 35970.25261744833,
      "steady_samples_steps_per_second": 294668309.4421367
    },
    "memory": {
      "memory_allocated_bytes": 919040,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 8192,
    "steps": 4096,
    "block": 256,
    "num_warps": 2,
    "graph_chunk": 16,
    "compile_seconds": 0.3988507203757763,
    "capture_seconds": 0.009353112429380417,
    "verification_steps": 16,
    "verification_relative_l2": 8.651608927600565e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.799477783203125,
        0.8013619384765625,
        0.8003041381835938,
        0.800690185546875,
        0.8024453125,
        0.8003727416992188,
        0.7949414672851562
      ],
      "steady_median_seconds": 0.8003727416992188,
      "steady_min_seconds": 0.7949414672851562,
      "steady_max_seconds": 0.8024453125,
      "steady_steps_per_second": 5117.615564098362,
      "steady_samples_steps_per_second": 41923506.70109378
    },
    "triton_eager": {
      "repeat_seconds": [
        0.8232960205078125,
        0.8241561889648438,
        0.829154296875,
        0.8241326293945312,
        0.824732666015625,
        0.822424560546875,
        0.8216063842773438
      ],
      "steady_median_seconds": 0.8241326293945312,
      "steady_min_seconds": 0.8216063842773438,
      "steady_max_seconds": 0.829154296875,
      "steady_steps_per_second": 4970.073813251666,
      "steady_samples_steps_per_second": 40714844.67815765
    },
    "triton_graph": {
      "repeat_seconds": [
        0.1112248306274414,
        0.11121868896484376,
        0.11120947265625,
        0.11120333099365234,
        0.11121049499511719,
        0.11120127868652344,
        0.1112125473022461
      ],
      "steady_median_seconds": 0.11121049499511719,
      "steady_min_seconds": 0.11120127868652344,
      "steady_max_seconds": 0.1112248306274414,
      "steady_steps_per_second": 36831.056279174365,
      "steady_samples_steps_per_second": 301720013.0389964
    },
    "memory": {
      "memory_allocated_bytes": 919040,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 8192,
    "steps": 4096,
    "block": 256,
    "num_warps": 4,
    "graph_chunk": 16,
    "compile_seconds": 0.33717237040400505,
    "capture_seconds": 0.009137354791164398,
    "verification_steps": 16,
    "verification_relative_l2": 8.651608927600565e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.803178466796875,
        0.8020940551757813,
        0.802788330078125,
        0.8033034057617188,
        0.8021483764648437,
        0.7986841430664062,
        0.7984639892578125
      ],
      "steady_median_seconds": 0.8021483764648437,
      "steady_min_seconds": 0.7984639892578125,
      "steady_max_seconds": 0.8033034057617188,
      "steady_steps_per_second": 5106.287215903277,
      "steady_samples_steps_per_second": 41830704.87267964
    },
    "triton_eager": {
      "repeat_seconds": [
        0.8246640625,
        0.8230390014648438,
        0.821106689453125,
        0.8226395874023438,
        0.8212408447265624,
        0.8228935546875,
        0.8226007080078125
      ],
      "steady_median_seconds": 0.8226395874023438,
      "steady_min_seconds": 0.821106689453125,
      "steady_max_seconds": 0.8246640625,
      "steady_steps_per_second": 4979.094202035639,
      "steady_samples_steps_per_second": 40788739.70307595
    },
    "triton_graph": {
      "repeat_seconds": [
        0.11017417907714844,
        0.11015782165527344,
        0.1101629409790039,
        0.1101629409790039,
        0.11015577697753906,
        0.11016397094726563,
        0.110166015625
      ],
      "steady_median_seconds": 0.1101629409790039,
      "steady_min_seconds": 0.11015577697753906,
      "steady_max_seconds": 0.11017417907714844,
      "steady_steps_per_second": 37181.28767804649,
      "steady_samples_steps_per_second": 304589108.6585568
    },
    "memory": {
      "memory_allocated_bytes": 919040,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 8192,
    "steps": 4096,
    "block": 256,
    "num_warps": 8,
    "graph_chunk": 16,
    "compile_seconds": 0.3067469112575054,
    "capture_seconds": 0.00882122665643692,
    "verification_steps": 16,
    "verification_relative_l2": 8.651608927600565e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.8063641357421875,
        0.808183837890625,
        0.8071669921875,
        0.8061112060546874,
        0.8050892944335938,
        0.8007515869140625,
        0.8002119750976563
      ],
      "steady_median_seconds": 0.8061112060546874,
      "steady_min_seconds": 0.8002119750976563,
      "steady_max_seconds": 0.808183837890625,
      "steady_steps_per_second": 5081.184790925885,
      "steady_samples_steps_per_second": 41625065.80726485
    },
    "triton_eager": {
      "repeat_seconds": [
        0.8226529541015625,
        0.8215736083984375,
        0.8166860961914062,
        0.813486083984375,
        0.8148203735351562,
        0.8143790283203125,
        0.8177305297851563
      ],
      "steady_median_seconds": 0.8166860961914062,
      "steady_min_seconds": 0.813486083984375,
      "steady_max_seconds": 0.8226529541015625,
      "steady_steps_per_second": 5015.3908816393305,
      "steady_samples_steps_per_second": 41086082.102389395
    },
    "triton_graph": {
      "repeat_seconds": [
        0.10927206420898437,
        0.10926182556152343,
        0.109264892578125,
        0.10934886169433594,
        0.10927104187011719,
        0.10927922821044922,
        0.10926796722412109
      ],
      "steady_median_seconds": 0.10927104187011719,
      "steady_min_seconds": 0.10926182556152343,
      "steady_max_seconds": 0.10934886169433594,
      "steady_steps_per_second": 37484.77116991918,
      "steady_samples_steps_per_second": 307075245.4239779
    },
    "memory": {
      "memory_allocated_bytes": 919040,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 8192,
    "steps": 4096,
    "block": 512,
    "num_warps": 1,
    "graph_chunk": 16,
    "compile_seconds": 0.8219367377460003,
    "capture_seconds": 0.009680237621068954,
    "verification_steps": 16,
    "verification_relative_l2": 8.651608927600565e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.8002867431640625,
        0.8013414306640625,
        0.8031211547851562,
        0.8019650268554688,
        0.79891455078125,
        0.7995985717773437,
        0.7995822143554687
      ],
      "steady_median_seconds": 0.8002867431640625,
      "steady_min_seconds": 0.79891455078125,
      "steady_max_seconds": 0.8031211547851562,
      "steady_steps_per_second": 5118.165501287457,
      "steady_samples_steps_per_second": 41928011.78654685
    },
    "triton_eager": {
      "repeat_seconds": [
        0.8346695556640625,
        0.83406640625,
        0.8338574829101563,
        0.8302335815429688,
        0.8268226318359375,
        0.826471435546875,
        0.8263106689453125
      ],
      "steady_median_seconds": 0.8302335815429688,
      "steady_min_seconds": 0.8263106689453125,
      "steady_max_seconds": 0.8346695556640625,
      "steady_steps_per_second": 4933.551341524495,
      "steady_samples_steps_per_second": 40415652.58976866
    },
    "triton_graph": {
      "repeat_seconds": [
        0.12302745819091797,
        0.12302438354492187,
        0.12301721954345703,
        0.12302130889892578,
        0.12302438354492187,
        0.12302438354492187,
        0.12302745819091797
      ],
      "steady_median_seconds": 0.12302438354492187,
      "steady_min_seconds": 0.12301721954345703,
      "steady_max_seconds": 0.12302745819091797,
      "steady_steps_per_second": 33294.21275664723,
      "steady_samples_steps_per_second": 272746190.90245414
    },
    "memory": {
      "memory_allocated_bytes": 919040,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 8192,
    "steps": 4096,
    "block": 512,
    "num_warps": 2,
    "graph_chunk": 16,
    "compile_seconds": 0.5293896421790123,
    "capture_seconds": 0.009310193359851837,
    "verification_steps": 16,
    "verification_relative_l2": 8.651608927600565e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.8000645141601562,
        0.7993477172851563,
        0.80091748046875,
        0.8012369995117188,
        0.8025968627929687,
        0.800668701171875,
        0.8011448364257813
      ],
      "steady_median_seconds": 0.80091748046875,
      "steady_min_seconds": 0.7993477172851563,
      "steady_max_seconds": 0.8025968627929687,
      "steady_steps_per_second": 5114.134851449052,
      "steady_samples_steps_per_second": 41894992.70307063
    },
    "triton_eager": {
      "repeat_seconds": [
        0.8324833374023437,
        0.83114599609375,
        0.83087255859375,
        0.8306073608398438,
        0.8304988403320313,
        0.8315361328125,
        0.8311766967773437
      ],
      "steady_median_seconds": 0.83114599609375,
      "steady_min_seconds": 0.8304988403320313,
      "steady_max_seconds": 0.8324833374023437,
      "steady_steps_per_second": 4928.135392879866,
      "steady_samples_steps_per_second": 40371285.138471864
    },
    "triton_graph": {
      "repeat_seconds": [
        0.11555635070800781,
        0.11554815673828125,
        0.11556454467773437,
        0.11555123138427735,
        0.11555123138427735,
        0.11554815673828125,
        0.11554815673828125
      ],
      "steady_median_seconds": 0.11555123138427735,
      "steady_min_seconds": 0.11554815673828125,
      "steady_max_seconds": 0.11556454467773437,
      "steady_steps_per_second": 35447.480316140776,
      "steady_samples_steps_per_second": 290385758.74982524
    },
    "memory": {
      "memory_allocated_bytes": 919040,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 8192,
    "steps": 4096,
    "block": 512,
    "num_warps": 4,
    "graph_chunk": 16,
    "compile_seconds": 0.40261633321642876,
    "capture_seconds": 0.009061232209205627,
    "verification_steps": 16,
    "verification_relative_l2": 8.651608927600565e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.8061624145507813,
        0.8059985961914062,
        0.8056156005859375,
        0.8053473510742187,
        0.806171630859375,
        0.80690380859375,
        0.8039137573242188
      ],
      "steady_median_seconds": 0.8059985961914062,
      "steady_min_seconds": 0.8039137573242188,
      "steady_max_seconds": 0.80690380859375,
      "steady_steps_per_second": 5081.894707205289,
      "steady_samples_steps_per_second": 41630881.441425726
    },
    "triton_eager": {
      "repeat_seconds": [
        0.8259143676757813,
        0.8305213623046875,
        0.83519384765625,
        0.8310425415039062,
        0.8318310546875,
        0.8298936157226563,
        0.8300328979492188
      ],
      "steady_median_seconds": 0.8305213623046875,
      "steady_min_seconds": 0.8259143676757813,
      "steady_max_seconds": 0.83519384765625,
      "steady_steps_per_second": 4931.8418356315915,
      "steady_samples_steps_per_second": 40401648.317494
    },
    "triton_graph": {
      "repeat_seconds": [
        0.11206553649902344,
        0.11204198455810546,
        0.11204300689697265,
        0.11206246185302735,
        0.11204402923583984,
        0.11204914855957031,
        0.11204198455810546
      ],
      "steady_median_seconds": 0.11204402923583984,
      "steady_min_seconds": 0.11204198455810546,
      "steady_max_seconds": 0.11206553649902344,
      "steady_steps_per_second": 36557.05732769026,
      "steady_samples_steps_per_second": 299475413.6284386
    },
    "memory": {
      "memory_allocated_bytes": 919040,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  },
  {
    "phase": "kernel_search",
    "n_samples": 8192,
    "steps": 4096,
    "block": 512,
    "num_warps": 8,
    "graph_chunk": 16,
    "compile_seconds": 0.33823683485388756,
    "capture_seconds": 0.009740278124809265,
    "verification_steps": 16,
    "verification_relative_l2": 8.651608927600565e-07,
    "full_chunks": 256,
    "remainder_steps": 0,
    "pytorch_eager": {
      "repeat_seconds": [
        0.8070205688476563,
        0.8046981201171876,
        0.8055234375,
        0.8051610107421875,
        0.8056668090820313,
        0.8044144897460938,
        0.8036904907226563
      ],
      "steady_median_seconds": 0.8051610107421875,
      "steady_min_seconds": 0.8036904907226563,
      "steady_max_seconds": 0.8070205688476563,
      "steady_steps_per_second": 5087.181253628212,
      "steady_samples_steps_per_second": 41674188.829722315
    },
    "triton_eager": {
      "repeat_seconds": [
        0.8323092651367188,
        0.824216552734375,
        0.8264959716796875,
        0.8273059692382813,
        0.8253583374023438,
        0.827652099609375,
        0.8249210815429687
      ],
      "steady_median_seconds": 0.8264959716796875,
      "steady_min_seconds": 0.824216552734375,
      "steady_max_seconds": 0.8323092651367188,
      "steady_steps_per_second": 4955.862025165955,
      "steady_samples_steps_per_second": 40598421.7101595
    },
    "triton_graph": {
      "repeat_seconds": [
        0.11152793884277344,
        0.11150643157958984,
        0.1115125732421875,
        0.11150950622558593,
        0.11151052856445312,
        0.11152486419677735,
        0.11151561737060547
      ],
      "steady_median_seconds": 0.1115125732421875,
      "steady_min_seconds": 0.11150643157958984,
      "steady_max_seconds": 0.11152793884277344,
      "steady_steps_per_second": 36731.28402394717,
      "steady_samples_steps_per_second": 300902678.7241752
    },
    "memory": {
      "memory_allocated_bytes": 919040,
      "memory_reserved_bytes": 8388608
    },
    "status": "ok"
  }
]
```
