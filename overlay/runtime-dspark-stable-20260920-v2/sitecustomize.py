"""Keep frozen Engram patches and load the audited runtime modules."""
import importlib.abc
import importlib.util
import os
from pathlib import Path
import runpy
import sys

root = os.environ.get('DSV41_FROZEN_ROOT')
if root:
    runpy.run_path(str(Path(root) / 'adapter' / 'sitecustomize.py'))
    class BackendFixFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == 'sglang.srt.entrypoints.openai.serving_responses':
                return importlib.util.spec_from_file_location(fullname, Path(__file__).with_name('serving_responses.py'))
            if fullname == 'sglang.srt.layers.attention.deepseek_v4_backend':
                return importlib.util.spec_from_file_location(fullname, Path(__file__).with_name('deepseek_v4_backend.py'))
            if fullname == 'sglang.srt.function_call.deepseekv41_detector':
                return importlib.util.spec_from_file_location(fullname, Path(__file__).with_name('deepseekv41_detector.py'))
            if fullname == 'sglang.srt.speculative.dspark_components.dspark_worker_v2':
                return importlib.util.spec_from_file_location(fullname, Path(__file__).with_name('dspark_worker_v2.py'))
            if fullname == 'sglang.srt.managers.scheduler':
                return importlib.util.spec_from_file_location(fullname, Path(__file__).with_name('scheduler.py'))
            if os.environ.get('DS41_ENABLE_FP8_VERIFY6_FASTPATH') == '1':
                if fullname in ('sglang.srt.layers.quantization.fp8', 'sglang.srt.layers.quantization.fp8_utils'):
                    print(f'[ds41-release] verify6-fastpath module={fullname} source={Path(__file__).parent}', flush=True)
                    return importlib.util.spec_from_file_location(fullname, Path(__file__).with_name(fullname.rsplit('.', 1)[1] + '.py'))
            return None
    sys.meta_path.insert(0, BackendFixFinder())
