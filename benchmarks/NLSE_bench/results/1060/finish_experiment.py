"""Continue the existing experiment without changing its solver or timing code."""
from pathlib import Path
import contextlib
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path('/home/optomachanics/projects/FlashLLE')
OUT = ROOT / 'benchmarks/NLSE_bench/results/1060'
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
os.chdir(ROOT)
OUT.mkdir(parents=True, exist_ok=True)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(name, value):
    target = OUT / name
    temp = target.with_suffix(target.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(target)


def announce(message):
    print(now(), message, flush=True)


def status(phase, **extra):
    save('RUN_STATUS.json', {'phase': phase, 'updated_utc': now(), **extra})


def command(args):
    return subprocess.check_output(args, text=True, timeout=20).strip()


def gpu_snapshot():
    return {
        'utc': now(),
        'gpu': command(['nvidia-smi', '--query-gpu=index,name,uuid,driver_version,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw', '--format=csv,noheader']),
        'compute_processes': command(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv,noheader']),
    }


def wait_for_idle():
    while True:
        snapshot = gpu_snapshot()
        with (OUT / 'gpu_monitor.jsonl').open('a') as f:
            f.write(json.dumps(snapshot) + '\n')
        others = []
        for line in snapshot['compute_processes'].splitlines():
            pid = line.split(',')[0].strip()
            if pid.isdigit() and int(pid) != os.getpid():
                others.append(line)
        if not others:
            return snapshot
        status('waiting_for_gpu', other_processes=others)
        announce('Waiting for other GPU processes: ' + repr(others))
        time.sleep(30)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


@contextlib.contextmanager
def log_phase(name):
    with (OUT / (name + '.log')).open('a') as handle:
        with contextlib.redirect_stdout(Tee(sys.stdout, handle)):
            announce('Starting ' + name)
            yield
            announce('Finished ' + name)


def main():
    # A disconnected SSH session does not imply that its GPU process ended.
    # Keep waiting for the original process; never run a duplicate search.
    original_pid = 10567
    process_cmd = Path(f'/proc/{original_pid}/cmdline')
    while process_cmd.exists():
        try:
            cmdline = process_cmd.read_bytes().decode().replace('\0', ' ')
        except FileNotFoundError:
            break
        if 'run_experiment.py' not in cmdline:
            break
        status('waiting_for_original_kernel', original_pid=original_pid)
        announce('Original kernel search still running')
        time.sleep(30)

    expected = json.loads((OUT / 'source_manifest.json').read_text())
    mismatches = []
    for name, digest in expected['files'].items():
        path = ROOT / name
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            mismatches.append(name)
    if mismatches:
        raise RuntimeError('Source differs from requested commit: ' + repr(mismatches))

    provenance = {
        'verified_utc': now(),
        'source_commit': expected['source_commit'],
        'verified_file_count': len(expected['files']),
        'all_expected_source_hashes_match': True,
        'remote_git_head': command(['git', 'rev-parse', 'HEAD']),
        'note': 'The working tree was synchronized with git archive. Raw git_commit reports the old repository HEAD; source_commit and source_manifest identify the executed code.',
        'gpu_uuid': 'GPU-0d704e0b-4b8b-5463-afc8-4a032831fab4',
        'gpu_model': 'NVIDIA GeForce GTX 1660',
        'directory_label': '1060 is the requested directory name, not the detected GPU model.',
        'pytest': {'passed': 190, 'seconds': 8.58, 'record': 'Observed full pytest run before correctness; the original console transcript is in the chat tool output.'},
    }
    save('provenance.json', provenance)
    wait_for_idle()

    from benchmarks.NLSE_bench import run_experiment as h

    def metadata():
        return {**h.metadata(), 'source_commit': expected['source_commit'], 'gpu_uuid': provenance['gpu_uuid'], 'provenance_file': 'provenance.json'}

    save('environment.json', {**metadata(), 'gpu_snapshot': gpu_snapshot()})

    kernel_path = OUT / 'kernel_search.json'
    if kernel_path.exists():
        kernel = json.loads(kernel_path.read_text())
        announce('Original kernel results recovered: ' + str(len(kernel['rows'])))
    else:
        # The original script only writes after all candidates. If it exited
        # without results, rerun with per-candidate checkpoints using the same
        # measurement function, warmups, repeats and candidate order.
        kernel = {'metadata': metadata(), 'rows': []}
        partial_path = OUT / 'kernel_search.partial.json'
        if partial_path.exists():
            kernel = json.loads(partial_path.read_text())
        seen = {(r['n_samples'], r['block'], r['num_warps']) for r in kernel['rows']}
        with log_phase('kernel_search.recovery'):
            for n in h.DEFAULT_SAMPLES:
                for block in h.DEFAULT_BLOCKS:
                    for warps in h.DEFAULT_WARPS:
                        if (n, block, warps) in seen:
                            continue
                        status('kernel', n_samples=n, block=block, num_warps=warps, completed=len(kernel['rows']))
                        kernel['rows'] += h.run_kernel_search((n,), OUT, blocks=(block,), warps=(warps,), steps=4096, graph_chunk=16, repeats=7, warmups=2)
                        save('kernel_search.partial.json', kernel)
        kernel['selection'] = h.select_best_config(kernel['rows'])
        h.write_results(OUT, 'kernel_search', kernel)

    assert len(kernel['rows']) == 64
    if any(r.get('status') != 'ok' for r in kernel['rows']):
        raise RuntimeError('Kernel search has failed candidates; review before selecting a configuration')
    selection = h.select_best_config(kernel['rows'])
    assert selection['sample_count'] == 4
    status('kernel_complete', selection=selection)
    with (OUT / 'kernel_search.completion.log').open('a') as handle:
        handle.write(now() + ' 64 candidates recovered; all successful; selection=' + json.dumps(selection) + '\n')
        handle.write('Original stdout before SSH disconnection is preserved in kernel_search.interrupted.log; later console output was unavailable. Complete measurements are in kernel_search.json.\n')

    wait_for_idle()
    chunk_path = OUT / 'chunk_search.json'
    if chunk_path.exists():
        chunks = json.loads(chunk_path.read_text())
    else:
        chunks = {'metadata': metadata(), 'rows': [], 'selection': selection}
        partial_path = OUT / 'chunk_search.partial.json'
        if partial_path.exists():
            chunks = json.loads(partial_path.read_text())
        seen = {(r['n_samples'], r['steps'], r['graph_chunk']) for r in chunks['rows']}
        with log_phase('chunk_search'):
            for n in h.DEFAULT_SAMPLES:
                for steps in (4096, 4097):
                    for chunk in h.DEFAULT_CHUNKS:
                        if (n, steps, chunk) in seen:
                            continue
                        status('chunk', n_samples=n, steps=steps, graph_chunk=chunk, completed=len(chunks['rows']))
                        chunks['rows'] += h.run_chunk_search((n,), OUT, block=selection['block'], num_warps=selection['num_warps'], chunks=(chunk,), steps_list=(steps,), repeats=7, warmups=2)
                        save('chunk_search.partial.json', chunks)
        assert len(chunks['rows']) == 72
        if any(r.get('status') != 'ok' for r in chunks['rows']):
            raise RuntimeError('Chunk search has failed candidates')
        # Choose across all four main sizes at 4096 steps. The original CLI
        # picks the fastest individual row, which need not be a global winner.
        scores = {
            str(chunk): h.geometric_mean(r['triton_graph']['steady_median_seconds'] for r in chunks['rows'] if r['steps'] == 4096 and r['graph_chunk'] == chunk)
            for chunk in h.DEFAULT_CHUNKS
        }
        chosen_chunk = int(min(scores, key=scores.get))
        chunks['selection'] = {**selection, 'graph_chunk': chosen_chunk, 'chunk_geometric_mean_seconds': scores[str(chosen_chunk)], 'chunk_scores': scores, 'chunk_selection_steps': 4096}
        h.write_results(OUT, 'chunk_search', chunks)
    chosen_chunk = chunks['selection']['graph_chunk']
    announce('Final configuration: ' + repr((selection['block'], selection['num_warps'], chosen_chunk)))

    if not (OUT / 'final.json').exists():
        wait_for_idle()
        status('final')
        with log_phase('final'):
            # Match the separate final-phase process used on L40s. In-memory
            # FFT plans and Triton dispatch caches from the sweep must not
            # silently warm the final phase's first invocation.
            code = '''
import json
from pathlib import Path
from benchmarks.NLSE_bench import run_experiment as h
out=Path('benchmarks/NLSE_bench/results/1060')
selection=json.loads((out/'chunk_search.json').read_text())['selection']
provenance=json.loads((out/'provenance.json').read_text())
meta={**h.metadata(), 'source_commit': provenance['source_commit'], 'gpu_uuid': provenance['gpu_uuid'], 'provenance_file': 'provenance.json', 'fresh_final_process': True}
rows=h.run_final(h.DEFAULT_SAMPLES, out, block=selection['block'], num_warps=selection['num_warps'], graph_chunk=selection['graph_chunk'], steps=4096, repeats=7, warmups=2)
h.write_results(out, 'final', {'metadata': meta, 'rows': rows, 'selection': selection})
'''
            child = subprocess.Popen([sys.executable, '-u', '-c', code], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in child.stdout:
                print(line, end='', flush=True)
            if child.wait() != 0:
                raise RuntimeError('Fresh final-phase process failed')

    if not (OUT / 'convergence.json').exists():
        status('convergence')
        with log_phase('convergence'):
            rows = h.convergence_check((4096,))
            for row in rows:
                print(json.dumps(row), flush=True)
            h.write_results(OUT, 'convergence', {'metadata': metadata(), 'rows': rows})

    counts = {}
    for name, expected_count in (('correctness', 25), ('kernel_search', 64), ('chunk_search', 72), ('final', 4), ('convergence', 3)):
        payload = json.loads((OUT / (name + '.json')).read_text())
        counts[name] = len(payload['rows'])
        assert counts[name] == expected_count, (name, counts[name])
    status('remote_complete', row_counts=counts, selection=chunks['selection'])
    announce('All experiment phases complete')


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        status('failed', error=repr(exc))
        raise
