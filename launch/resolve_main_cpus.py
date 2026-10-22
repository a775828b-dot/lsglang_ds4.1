"""Resolve startup affinity using the loaded lkqmoe library's worker topology."""
import argparse
import ctypes as C
import os


def parse_cpu_list(value):
    cpus = set()
    for part in value.split(','):
        lo, separator, hi = part.strip().partition('-')
        first, last = int(lo), int(hi) if separator else int(lo)
        if first < 0 or last < first:
            raise ValueError('Invalid CPU range: ' + part)
        cpus.update(range(first, last + 1))
    return cpus


def resolve(args):
    if not 1 <= args.threads <= 128:
        raise ValueError('Worker thread count must be between 1 and 128')
    lib = C.CDLL(args.library)
    lib.lkqmoe_worker_cpus.argtypes = [C.c_int, C.POINTER(C.c_int), C.c_int]
    lib.lkqmoe_worker_cpus.restype = C.c_int
    lib.lkqmoe_last_error.restype = C.c_char_p
    buf = (C.c_int * args.threads)()
    count = lib.lkqmoe_worker_cpus(args.threads, buf, args.threads)
    if count != args.threads:
        error = lib.lkqmoe_last_error()
        raise RuntimeError(error.decode() if error else 'Worker topology query failed')
    workers, allowed = set(buf), os.sched_getaffinity(0)
    if len(workers) != args.threads or not workers <= allowed:
        raise ValueError('Worker CPUs are duplicated or outside the service CPU allowance')
    if args.dispatcher not in allowed or args.dispatcher in workers:
        raise ValueError('Dispatcher CPU is unavailable or overlaps an expert worker')
    reserved = workers | {args.dispatcher}
    main = allowed - reserved if args.main_cpus == 'auto' else parse_cpu_list(args.main_cpus)
    if not main or not main <= allowed or main & reserved:
        raise ValueError('Main CPUs are empty, unavailable, or overlap workers/dispatcher')
    return ','.join(map(str, sorted(main)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', required=True)
    parser.add_argument('--threads', type=int, required=True)
    parser.add_argument('--dispatcher', type=int, required=True)
    parser.add_argument('--main-cpus', default='auto')
    print(resolve(parser.parse_args()))
