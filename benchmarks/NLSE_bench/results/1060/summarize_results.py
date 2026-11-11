"""Validate raw experiment artifacts and generate a readable, derived report."""
import csv
import json
import math
from pathlib import Path

import sys

OUT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent
ROOT = OUT.parent


def load(name, folder=OUT):
    return json.loads((folder / (name + '.json')).read_text())


def median(row, key):
    return row[key]['steady_median_seconds']


def table(headers, rows):
    return ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] + ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]


datasets = {name: load(name) for name in ('correctness', 'kernel_search', 'chunk_search', 'final', 'convergence')}
for name, count in (('correctness', 25), ('kernel_search', 64), ('chunk_search', 72), ('final', 4), ('convergence', 3)):
    assert len(datasets[name]['rows']) == count
kernel = datasets['kernel_search']
chunks = datasets['chunk_search']
final = datasets['final']
correctness = datasets['correctness']['rows']
for name in ('kernel_search', 'chunk_search'):
    for row in datasets[name]['rows']:
        assert row['status'] == 'ok', row
        for key in ('triton_graph', 'triton_eager', 'pytorch_eager'):
            if key in row:
                assert len(row[key]['repeat_seconds']) == 7
                assert all(math.isfinite(v) and v > 0 for v in row[key]['repeat_seconds'])
for row in correctness:
    for key in ('optimized_accuracy', 'pytorch_accuracy'):
        assert row[key]['finite'] and row[key]['dtype'] == 'complex64'
        assert row[key]['shape'] == [row['n_samples']]
    assert row['optimized_input_preserved'] and row['pytorch_input_preserved']
for row in final['rows']:
    assert row['optimized_accuracy']['finite'] and row['pytorch_accuracy']['finite']
    assert len(row['optimized_steady']['repeat_seconds']) == 7

selection = final['selection']
assert final['metadata']['fresh_final_process'] is True
block, warps, chunk = (selection[k] for k in ('block', 'num_warps', 'graph_chunk'))
resident = {r['n_samples']: r for r in kernel['rows'] if (r['block'], r['num_warps']) == (block, warps)}
resident_graph = {r['n_samples']: r for r in chunks['rows'] if r['steps'] == 4096 and r['graph_chunk'] == chunk}
failed = [r for r in correctness if not r['passed']]
environment = load('environment')
provenance = load('provenance')
l40s = {r['n_samples']: r for r in load('final', ROOT / 'L40s')['rows']}
l40s_meta = load('final', ROOT / 'L40s')['metadata']
ranked_chunks = sorted(selection['chunk_scores'], key=selection['chunk_scores'].get)
runner_up = ranked_chunks[1]
margin = (selection['chunk_scores'][runner_up] / selection['chunk_scores'][str(chunk)] - 1) * 100
variable_case = resident_graph[4096]['triton_graph']

lines = [
    '# GTX 1660 NLSE 实验结果', '',
    f'实际 GPU：NVIDIA GeForce GTX 1660（6 GiB）；按用户指定目录名保存于 `1060/`。GPU UUID：`{provenance["gpu_uuid"]}`。', '',
    f'执行源码：`{provenance["source_commit"]}`。{provenance["verified_file_count"]} 个源文件的 SHA256 均与该提交一致。远端旧 Git HEAD 为 `{provenance["remote_git_head"]}`；原始 metadata.git_commit 记录的是这个旧 HEAD，请结合 [provenance.json](provenance.json) 和 [source_manifest.json](source_manifest.json) 使用。', '',
    f'环境：Python {environment["python"].split()[0]}、PyTorch {environment["torch"]}、CUDA runtime {environment["torch_cuda"]}、Triton {environment["triton"]}、NumPy {environment["numpy"]}、WSL2。', '',
    '完整 pytest：190 passed（8.58 s）。实验规模：N=1024/2048/4096/8192，steps=4096；kernel 搜索 64 组，chunk 搜索 72 组（含 steps=4097），每组 2 次预热、7 次正式计时。', '',
    '物理参数：dz=0.001、dt=20/N、alpha=0、beta2=-1、beta3=0.05、beta4=0、gamma0=1、omega0=20；sech 初始场，主实验传播距离 4.096。NumPy 为 complex128，GPU 为 complex64。', '',
    f'按四个主规模的稳态耗时几何平均选择：BLOCK={block}、num_warps={warps}、graph_chunk={chunk}。chunk 的选择仅使用 steps=4096；4097 用于覆盖余数步路径。', '',
    f'这是本次单轮搜索的最优候选，不能直接断言在其他负载下仍最优。次优 chunk={runner_up} 的几何平均耗时比 chunk={chunk} 高约 {margin:.2f}%；N=4096、chunk={chunk} 的 7 次测量范围为 {variable_case["steady_min_seconds"]*1000:.3f}–{variable_case["steady_max_seconds"]*1000:.3f} ms。请结合重复样本的抖动评估排名，原始样本全部保留。', '',
    '## 端到端测量', '',
    '单位为毫秒。最终阶段使用新 Python 进程。NumPy、PyTorch 每个规模各记录一次端到端调用；优化稳态为预热后的 7 次中位数，包含输入输出传输。PyTorch 这里可能包含首次初始化，因此该列加速比不能解释为纯 GPU kernel 加速比；下节提供 GPU 常驻的预热对照。', '',
]
lines += table(['N', 'NumPy CPU', 'PyTorch GPU 单次', '优化首次调用', '优化缓存 Graph 稳态', 'PyTorch/优化'], [
    [r['n_samples'], f'{r["numpy_seconds"]*1000:.3f}', f'{r["pytorch_end_to_end"]*1000:.3f}', f'{r["optimized_first_call_seconds"]*1000:.3f}', f'{median(r,"optimized_steady")*1000:.3f}', f'{r["optimized_speedup_pytorch"]:.2f}×']
    for r in final['rows']
])
lines += ['', '## GPU 常驻对照', '', '均为 CUDA Event 计时、中位数，排除主机传输与系数准备，包含输入恢复和输出克隆。eager 测量也包含 Python 提交不及时产生的 GPU 空闲时间。Graph 收益以同一轮 kernel 搜索的 chunk=16 对照计算。', '']
resident_rows = []
for n, r in resident.items():
    pt, eager, graph = [median(r, key) for key in ('pytorch_eager', 'triton_eager', 'triton_graph')]
    resident_rows.append([n, f'{pt*1000:.3f}', f'{eager*1000:.3f}', f'{graph*1000:.3f}', f'{pt/eager:.2f}×', f'{eager/graph:.2f}×', f'{median(resident_graph[n],"triton_graph")*1000:.3f}'])
lines += table(['N', 'PyTorch eager ms', 'Triton eager ms', 'Graph chunk=16 ms', 'PyTorch/Triton eager', 'Triton eager/Graph', f'最佳 chunk={chunk} ms'], resident_rows)

lines += ['', '## 首次成本与吞吐量', '', '`compile_seconds` 是首次 Triton 单步调用耗时，包含首次派发、可能的 JIT 或缓存加载、执行和同步，并非纯编译器耗时。`capture_seconds` 包含 stepper 构造、内部预热、同步与捕获。进程之间使用已有 Triton 磁盘缓存，故首次成本也可能命中缓存。显存为 PyTorch allocator 统计，不能等同于 nvidia-smi 整卡用量。', '']
lines += table(['N', '首个 Triton 调用 ms', '建图 ms', '分配峰值 MiB', '保留峰值 MiB', 'steps/s', '样本步/s'], [
    [r['n_samples'], f'{resident[r["n_samples"]]["compile_seconds"]*1000:.3f}', f'{resident_graph[r["n_samples"]]["capture_seconds"]*1000:.3f}', f'{r["memory"]["memory_allocated_bytes"]/2**20:.3f}', f'{r["memory"]["memory_reserved_bytes"]/2**20:.3f}', f'{r["optimized_steady"]["steady_steps_per_second"]:.0f}', f'{r["optimized_steady"]["steady_samples_steps_per_second"]:.0f}']
    for r in final['rows']
])

lines += ['', '## 每个规模的最佳候选', '']
best_rows = []
for n in resident:
    k = min((r for r in kernel['rows'] if r['n_samples'] == n), key=lambda r: median(r, 'triton_graph'))
    c = min((r for r in chunks['rows'] if r['n_samples'] == n and r['steps'] == 4096), key=lambda r: median(r, 'triton_graph'))
    best_rows.append([n, k['block'], k['num_warps'], f'{median(k,"triton_graph")*1000:.3f}', c['graph_chunk'], f'{median(c,"triton_graph")*1000:.3f}'])
lines += table(['N', '最佳 BLOCK', '最佳 num_warps', 'kernel 搜索 ms（chunk=16）', '最佳 chunk', 'chunk 搜索 ms'], best_rows)

lines += ['', '## 正确性和收敛', '', f'正确性检查 {len(correctness)-len(failed)}/25 通过；所有 GPU 输出 shape、dtype、有限值和初态不变检查均符合预期。以下 {len(failed)} 项超过原定场相对 L2 上限 1e-3，严格精度验收未通过；阈值未放宽。PyTorch 与 Triton 均出现同类长步误差，与有限精度累计误差的解释一致，但本轮未单独定位误差来源。', '']
lines += table(['N', 'steps', 'Triton/NumPy 场相对 L2', 'PyTorch/NumPy 场相对 L2', 'Triton/PyTorch 场相对 L2'], [
    [r['n_samples'], r['steps'], f'{r["optimized_accuracy"]["relative_l2"]:.6e}', f'{r["pytorch_accuracy"]["relative_l2"]:.6e}', f'{r["optimized_vs_pytorch_relative_l2"]:.6e}'] for r in failed
])
lines += ['', '步长收敛采用 NumPy complex128、N=4096、固定传播距离 4.096，以最小步长解作参考：', '']
lines += table(['dz', 'steps', '相对最小步长的场 L2', '能量', '有限值'], [
    [r['dz'], r['steps'], f'{r["relative_l2_to_smallest_dz"]:.6e}', f'{r["energy"]:.9f}', r['finite']] for r in datasets['convergence']['rows']
])

lines += ['', '## 与 L40s 的观测对照', '', f'L40s 环境为 PyTorch {l40s_meta["torch"]}、CUDA {l40s_meta["torch_cuda"]}、Triton {l40s_meta["triton"]}，且宿主机 CPU、驱动、操作系统和运行方式不同。下表是两套环境各自选优后的整体耗时比较，不能归因为单一 GPU 硬件差异。', '']
comparisons = []
for r in final['rows']:
    n = r['n_samples']
    t, base = median(r, 'optimized_steady'), median(l40s[n], 'optimized_steady')
    comparisons.append({'n_samples': n, 'gtx1660_optimized_e2e_seconds': t, 'l40s_optimized_e2e_seconds': base, 'gtx1660_over_l40s_time_ratio': t/base})
lines += table(['N', 'L40s 优化稳态 ms', 'GTX 1660 优化稳态 ms', 'GTX 1660/L40s 耗时比'], [[r['n_samples'], f'{r["l40s_optimized_e2e_seconds"]*1000:.3f}', f'{r["gtx1660_optimized_e2e_seconds"]*1000:.3f}', f'{r["gtx1660_over_l40s_time_ratio"]:.2f}'] for r in comparisons])
lines += ['', '## 记录说明', '', 'kernel 搜索中 IPv4 SSH 连接断开，但原进程继续执行，并通过 IPv6 连接恢复监测；未重复启动同一个搜索。断联前的 stdout 保存在 `kernel_search.interrupted.log`；断联后的 console 输出无法恢复，完整 64 组测量在 `kernel_search.json` 中。后续阶段通过 `finish_experiment.py` 接续执行，每阶段日志均在本目录保存。', '', '本轮保留 commit 9d60f65 的核心求解器和测量函数；接续脚本只增加恢复、进度保存、来源校验，并按四个规模几何平均选择全局 chunk。原 CLI 按最短单条记录选 chunk，两者规则区别已显式记录。', '', '原始 `optimized_speedup_eager_reference` 字段实际是首次调用时间/缓存稳态时间，本报告未把它当作 Graph 相对 eager 的收益。', '', '文件索引：[correctness.json](correctness.json)、[kernel_search.json](kernel_search.json)、[chunk_search.json](chunk_search.json)、[final.json](final.json)、[convergence.json](convergence.json)、[environment.json](environment.json)。', '']
(OUT / 'SUMMARY.md').write_text('\n'.join(lines))
with (OUT / 'comparison_L40s.csv').open('w', newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(comparisons[0]))
    writer.writeheader()
    writer.writerows(comparisons)
summary={'all_experiment_phases_complete': True, 'strict_accuracy_acceptance': len(failed)==0, 'correctness_passed': len(correctness)-len(failed), 'correctness_total': len(correctness), 'selection': selection, 'source_commit': provenance['source_commit'], 'row_counts': {k:len(v['rows']) for k,v in datasets.items()}, 'comparison': comparisons}
(OUT / 'validation_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
