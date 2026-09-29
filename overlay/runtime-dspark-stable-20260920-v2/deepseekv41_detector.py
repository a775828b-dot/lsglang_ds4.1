import json
import re
from typing import Optional

from sglang.srt.function_call.deepseekv32_detector import DeepSeekV32Detector


class DeepSeekV41Detector(DeepSeekV32Detector):
    """Spaced DSML tags, including complete constrained invokes without a calls wrapper."""

    tool_calls_block_name = " calls"
    invoke_tag_name = " invoke"
    parameter_tag_name = " parameter"

    def _complete_json_invokes(self, text, tools):
        # The legacy structural grammar constrains invoke + JSON + /invoke,
        # but does not require the surrounding calls block. Recover only a
        # complete sequence; never hide arbitrary prose or incomplete arguments.
        body = text.removesuffix("</tools>").rstrip()
        names = {tool.function.name for tool in tools}
        offset = 0
        count = 0
        while offset < len(body):
            match = re.compile(self.invoke_regex, re.DOTALL).match(body, offset)
            if match is None:
                return None
            name, payload, complete = self._unpack_invoke_match(match)
            if not complete or name not in names:
                return None
            try:
                args = json.loads(payload) if payload.strip() else {}
            except (TypeError, ValueError):
                return None
            if not isinstance(args, dict):
                return None
            offset = match.end()
            while offset < len(body) and body[offset].isspace():
                offset += 1
            count += 1
        return body if count else None

    def detect_and_parse(self, text, tools):
        stripped = text.strip()
        if self.bot_token not in text and stripped.startswith(self.invoke_start_token):
            if stripped.endswith(self.eot_token):
                text = self.bot_token + "\n" + stripped
            else:
                body = self._complete_json_invokes(stripped, tools)
                if body is not None:
                    text = self.bot_token + "\n" + body + "\n" + self.eot_token
        return super().detect_and_parse(text, tools)

    def get_structural_tag_name(self) -> Optional[str]:
        # Builtin deepseek_v4 uses unspaced tags; retain the spaced legacy grammar.
        return None
