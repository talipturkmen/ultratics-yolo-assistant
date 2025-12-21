import math
import textwrap
import tiktoken

# Initialize a tokenizer (we use cl100k_base encoding as a default)
_tokenizer = None
try:
    _tokenizer = tiktoken.get_encoding("cl100k_base")
except Exception:
    try:
        _tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
    except Exception:
        _tokenizer = None

def count_tokens(text: str) -> int:
    """Count approximate tokens in the text. Uses tiktoken if available, else falls back to word count."""
    if _tokenizer:
        try:
            tokens = _tokenizer.encode(text)
            return len(tokens)
        except Exception:
            # If encoding fails for some reason, fallback
            pass
    # Fallback: approximate tokens by splitting on whitespace/punctuation
    return len(text.split())

def get_text_segment(source: str, start_line: int, end_line: int) -> str:
    """Extract the text from source corresponding to the given line range (1-indexed)."""
    lines = source.splitlines()
    # Ensure indices are within range
    start_idx = max(0, start_line - 1)
    end_idx = min(len(lines), end_line)
    segment = "\n".join(lines[start_idx:end_idx])
    return segment

def split_chunks_by_tokens(text: str, start_line: int, target_tokens: int, overlap_tokens: int):
    """Split a text chunk (with known start line) into smaller chunks by token count, with overlap."""
    # Split text into lines for safer splitting
    lines = text.splitlines()
    result = []
    cur_line = 0
    total_lines = len(lines)
    # We will accumulate lines until reaching target_tokens, then split at a line boundary
    while cur_line < total_lines:
        if cur_line >= total_lines:
            break
        # Determine how many lines to include in this segment
        segment_lines = []
        token_count = 0
        line_idx = cur_line
        while line_idx < total_lines and token_count < target_tokens:
            segment_lines.append(lines[line_idx])
            # Count tokens in the current segment (join with newline)
            segment_text = "\n".join(segment_lines)
            token_count = count_tokens(segment_text)
            line_idx += 1
            if token_count >= target_tokens:
                break
        # segment_lines now represents the chunk from cur_line to line_idx-1 (inclusive)
        end_line_num = start_line + line_idx - 1  # end line number for this chunk
        segment_text = "\n".join(segment_lines)
        result.append((segment_text, start_line + cur_line, end_line_num))
        if line_idx >= total_lines:
            break
        # Set next start line index with overlap
        # We'll overlap a few lines from the end of this segment
        overlap_lines = max(1, math.floor(len(segment_lines) * (overlap_tokens / max(1, token_count)))) if token_count else 1
        # At minimum, overlap 1 line
        overlap_lines = max(overlap_lines, 1)
        cur_line = max(line_idx - overlap_lines, cur_line + 1)
    return result
