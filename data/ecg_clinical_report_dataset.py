import os
import json
import warnings
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from typing import Any, Dict, Optional, Sequence, cast, List, TYPE_CHECKING
from transformers import PreTrainedTokenizerBase
try:
    from transformers import ProcessorMixin
except ImportError:  # pragma: no cover
    ProcessorMixin = PreTrainedTokenizerBase  # type: ignore[misc,assignment]

from utils.ddp import DistributedUtils
from transformers import BatchEncoding
from torch.utils.data import Dataset, DataLoader, ConcatDataset, default_collate, Subset
from utils.config.llm_finetuning_config import LLMFinetuningConfig
if TYPE_CHECKING:
    from models.types import AutoTokenizerT
else:
    AutoTokenizerT = Any
from utils.constants import lead_to_idx, ECG_PATTERNS


class ECGClinicalReportDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        signal_path_column: str,
        ecg_waveform_length: int,
        ecg_num_leads: int,
        tokenizer: AutoTokenizerT,
        max_length: int = 512,
        instruct_mode: bool = False,
        # Default to 0 so the bridge owns the ECG prefix (BLIP-2 style)
        num_ecg_tokens: int = 0,
        ecg_token_start_id: Optional[int] = None,
        prompt_column: str = "question",
        answer_column: str = "report",
        category_column: str = "prompt_category",
        prefix_tuning: bool = False,
        pattern_columns: Optional[Sequence[str]] = None,
        lvef_head_column: Optional[str] = "deepecho_Visually_Estimated_EF",
        shd_head_column: Optional[str] = "echonext_shd_binary",
        afib_head_column: Optional[str] = "afib_label_5y",
        medgemma_prompt_style: bool = False,
        debug_print_example: bool = False,
        augmentor: Optional[Any] = None,
        messages_column: Optional[str] = None,
        prompt_variations_path: Optional[str] = None,
        signal_paths_column: Optional[str] = None,
        max_ecgs: int = 8,
    ):
        """
        Args:
            dataset_path (str): Path to the dataset.
            signal_path_column (str): Column name for ECG signal path.
            ecg_waveform_length (int): Length of ECG waveform.
            ecg_num_leads (int): Number of ECG leads.
            tokenizer (PreTrainedTokenizer): Tokenizer for the clinical reports.
            max_length (int): Maximum token length for the reports.
            instruct_mode (bool): Whether to use instruction tuning mode.
            num_ecg_tokens (int): Number of ECG tokens.
            ecg_token_start_id (Optional[int]): Starting ID for ECG tokens.
            prompt_column (str): Column name for input prompts/questions.
            answer_column (str): Column name for expected outputs/answers.
            category_column (str): Column name for prompt categories.
            pattern_columns (Optional[Sequence[str]]): Column names providing multilabel ECG targets.
            medgemma_prompt_style (bool): Use MedGemma-style chat prompts with <image_1> placeholder.
            messages_column (Optional[str]): Column containing JSON chat messages (system/user/assistant).
                When set, overrides prompt_column/answer_column with parsed message content.
            prompt_variations_path (Optional[str]): Path to JSON file with prompt variations per category.
                When set, randomly samples a prompt variation for each sample during training.
        """
        try:
            self.df: pd.DataFrame = pd.read_parquet(dataset_path)
        except Exception as e:
            print(f"Error reading parquet file: {e}")
            raise Exception(f"Error reading parquet file: {e}")
        
        # Normalize numeric configuration to ints to avoid type errors when YAML/CLI
        # values are passed as strings.
        self.ecg_waveform_length: int = int(ecg_waveform_length)
        self.ecg_num_leads: int = int(ecg_num_leads)

        if not isinstance(tokenizer, PreTrainedTokenizerBase):
            raise ValueError("Tokenizer must be a PreTrainedTokenizerBase")

        self.tokenizer: AutoTokenizerT = cast(AutoTokenizerT, tokenizer)
        self._pt_tokenizer: PreTrainedTokenizerBase = tokenizer
        self.max_length: int = int(max_length)
        self.signal_path_column: str = signal_path_column
        self.instruct_mode: bool = instruct_mode
        self.num_ecg_tokens: int = int(num_ecg_tokens)
        self.prefix_tuning: bool = bool(prefix_tuning)
        self.pattern_columns: List[str] = list(pattern_columns) if pattern_columns else list(ECG_PATTERNS)
        self.medgemma_prompt_style: bool = bool(medgemma_prompt_style)
        self.debug_print_example: bool = bool(debug_print_example)
        self._debug_example_printed: bool = False
        self.augmentor = augmentor
        # Multi-ECG support: when signal_paths_column is present, each row may carry a
        # list of waveform paths (temporally-nearby ECGs). N spans are spliced in the
        # decoder, one per <start_of_image> token. N=1 reduces to the single-ECG path.
        self.signal_paths_column: Optional[str] = signal_paths_column
        self.max_ecgs: int = int(max_ecgs)
        self._multi_ecg: bool = bool(signal_paths_column and signal_paths_column in self.df.columns)
        # Chat messages column support: when set, parse JSON messages for prompt/answer
        self.messages_column: Optional[str] = messages_column
        if self.messages_column and self.messages_column not in self.df.columns:
            warnings.warn(
                f"ECGClinicalReportDataset: messages_column '{self.messages_column}' "
                f"not found in dataset. Falling back to prompt_column/answer_column.",
                stacklevel=2,
            )
            self.messages_column = None
        # Auxiliary bridge-head label columns (masked per-row; NaN -> excluded from head loss)
        self.lvef_head_column: Optional[str] = lvef_head_column
        self.shd_head_column: Optional[str] = shd_head_column
        self.afib_head_column: Optional[str] = afib_head_column
        self._pattern_column_mask: List[bool] = [col in self.df.columns for col in self.pattern_columns]
        missing_patterns = [col for col, present in zip(self.pattern_columns, self._pattern_column_mask) if not present]
        if missing_patterns:
            warnings.warn(
                f"ECGClinicalReportDataset: missing {len(missing_patterns)} pattern columns in dataset: "
                f"{missing_patterns[:5]}{'...' if len(missing_patterns) > 5 else ''}. "
                "Missing columns will be treated as zeros.",
                stacklevel=2,
            )

        pad_token_id = getattr(self._pt_tokenizer, 'pad_token_id', None)
        eos_token_id = getattr(self._pt_tokenizer, 'eos_token_id', None)
        if isinstance(eos_token_id, list):
            eos_token_id = eos_token_id[0] if len(eos_token_id) > 0 else None
        if pad_token_id is None and eos_token_id is None:
            raise ValueError("Tokenizer must define a pad_token_id or eos_token_id for prefix placeholders")
        if pad_token_id is None:
            pad_token_id = eos_token_id
        self.ecg_prefix_token_id: int = int(pad_token_id)

        # Column configuration
        self.prompt_column: str = prompt_column
        self.answer_column: str = answer_column
        self.category_column: str = category_column
        # Prompt variations: {category: {original_prompt: [variations]}}.
        # Each variation is only used to replace its specific source prompt.
        # Generated per-prompt by generate_per_prompt_variations.py.
        self._prompt_variations: Dict[str, Dict[str, List[str]]] = {}
        if prompt_variations_path and os.path.isfile(prompt_variations_path):
            import json as _json
            with open(prompt_variations_path) as _f:
                self._prompt_variations = _json.load(_f)
            _total = sum(len(v) for grp in self._prompt_variations.values() for v in grp.values())
            _cats = len(self._prompt_variations)
            _n_prompts = sum(len(g) for g in self._prompt_variations.values())
            print(f"[PromptVariations] Loaded {_total} variations for {_n_prompts} prompts across {_cats} categories from {prompt_variations_path}")
        if self.instruct_mode:
            # Q-Former or query-only prefix: no textual ECG placeholders
            if self.num_ecg_tokens <= 0:
                self.ecg_token_start_id = None
                self.ecg_token_ids = []
            else:
                # Prefer soft placeholders via existing pad/eos token to avoid expanding vocab
                # Treat missing ecg_token_start_id as a signal to use pad/eos placeholders.
                use_soft_prefix = self.prefix_tuning or (ecg_token_start_id is None)
                if use_soft_prefix:
                    self.ecg_token_start_id = self.ecg_prefix_token_id
                    self.ecg_token_ids = [self.ecg_prefix_token_id] * self.num_ecg_tokens
                else:
                    token_id = int(ecg_token_start_id)
                    self.ecg_token_start_id = token_id
                    self.ecg_token_ids = list(range(
                        self.ecg_token_start_id,
                        self.ecg_token_start_id + self.num_ecg_tokens
                    ))
        else:
            if self.prefix_tuning:
                self.ecg_token_start_id = self.ecg_prefix_token_id
            else:
                self.ecg_token_start_id = 0 if ecg_token_start_id is None else int(ecg_token_start_id)
            self.ecg_token_ids = []
    def __len__(self):
        return len(self.df)

    def _aux_head_gt(self, row, column: Optional[str]) -> float:
        """Masked ground truth for an auxiliary bridge head: value if present, else NaN."""
        if column and column in self.df.columns:
            value = row.get(column)
            if not pd.isnull(value):
                return float(value)
        return float('nan')

    def load_ecg_signal(self, waveform_path: str) -> np.ndarray:
        try:
            waveform: np.ndarray = np.load(waveform_path)
        except Exception as e:
            print(f"Error loading ECG signal: {e}")
            raise Exception(f"Error loading ECG signal: {e}")
        
        # Hack for MHI dataset stored as 3D array with shape (2500, 12, 1)
        if len(waveform.shape) == 3:
            waveform = waveform.squeeze(-1)
            
        assert len(waveform.shape) == 2, f"Unnormalized signal has shape {waveform.shape}"

        return waveform

    def _process_waveform(self, waveform: np.ndarray) -> Optional[np.ndarray]:
        """Crop/pad to target length, augment, validate leads. Returns [length, leads] or None."""
        if np.isnan(waveform).any():
            return None
        target_length = int(self.ecg_waveform_length)
        current_length = waveform.shape[0]
        if current_length >= target_length:
            start = max((current_length - target_length) // 2, 0)
            waveform = waveform[start:start + target_length, :]
        else:
            pad_before = max((target_length - current_length) // 2, 0)
            pad_after = max(target_length - current_length - pad_before, 0)
            waveform = np.pad(waveform, ((pad_before, pad_after), (0, 0)), mode="edge")
        if waveform.shape[0] != target_length:
            waveform = np.resize(waveform, (target_length, waveform.shape[1]))
        if self.augmentor is not None:
            waveform = self.augmentor(waveform)
        if waveform.shape[1] != self.ecg_num_leads:
            return None
        return waveform

    def load_ecg_signals(self, paths: List[str]) -> List[np.ndarray]:
        """Load + process a list of waveform paths; silently drops unreadable/invalid ones."""
        out: List[np.ndarray] = []
        for p in paths:
            try:
                w = self.load_ecg_signal(p)
            except Exception:
                continue
            w = self._process_waveform(w)
            if w is not None:
                out.append(w)
        return out

    def _set_num_image_tokens(self, text: str, n: int) -> str:
        """Force the prompt to contain exactly ``n`` <start_of_image> anchors (one per ECG).

        n=0 removes the anchor entirely (no-ECG / adversarial rows).
        """
        tok = "<start_of_image>"
        idx = text.find(tok)
        if idx < 0:
            return (tok + "\n") * n + text if n > 0 else text
        # idx is the FIRST occurrence, so no tok precedes it; removing all tok
        # leaves text[:idx] unchanged.
        cleaned = text.replace(tok, "")
        block = (tok + "\n") * n
        return cleaned[:idx] + block + cleaned[idx:]

    def _parse_messages(self, raw: Any) -> Optional[Dict[str, str]]:
        """Parse a JSON chat messages column into system/user/assistant content.

        Returns a dict with keys ``system``, ``user``, ``assistant`` or *None*
        if parsing fails.
        """
        try:
            msgs = json.loads(raw) if isinstance(raw, str) else raw
            if not isinstance(msgs, list):
                return None
            result: Dict[str, str] = {}
            for msg in msgs:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role in ("system", "user", "assistant"):
                    result[role] = content
            if "assistant" not in result:
                return None
            return result
        except Exception:
            return None

    def __getitem__(self, idx: int) -> dict | None:
        try:
            # Get the row
            row = self.df.iloc[idx]

            # When messages_column is active, derive answer_column availability from it
            has_messages = (
                self.messages_column is not None
                and self.messages_column in self.df.columns
                and not pd.isnull(row[self.messages_column])
            )

            # Check if the waveform path or answer is missing
            answer_missing = (
                not has_messages
                and (self.answer_column not in self.df.columns or pd.isnull(row.get(self.answer_column)))
            )
            # Resolve the list of ECG waveform paths (multi-ECG aware).
            if self._multi_ecg:
                raw_paths = row.get(self.signal_paths_column)
                try:
                    sig_paths = [str(p) for p in list(raw_paths)] if raw_paths is not None else []
                except TypeError:
                    sig_paths = []
                sig_paths = sig_paths[: self.max_ecgs]
            else:
                sp = row.get(self.signal_path_column)
                sig_paths = [] if pd.isnull(sp) else [str(sp)]

            # Single-ECG mode requires a signal; multi-ECG mode allows N=0 (no-ECG rows).
            if answer_missing or (not self._multi_ecg and len(sig_paths) == 0):
                print(f"Missing {self.signal_path_column} or {self.answer_column} for index {idx}, skipping sample. "
                      f"{self.signal_path_column}: {row.get(self.signal_path_column)}, {self.answer_column}: {row.get(self.answer_column)}")
                return self.__getitem__((idx + 1) % len(self))

            # Load + process all waveforms.
            signals: List[np.ndarray] = self.load_ecg_signals(sig_paths)
            if len(sig_paths) > 0 and len(signals) == 0:
                # All listed signals failed to load -> skip to next sample.
                return self.__getitem__((idx + 1) % len(self))

            num_signals: int = len(signals)
            target_length = int(self.ecg_waveform_length)
            # waveform kept for back-compat with downstream single-signal references.
            waveform: np.ndarray = signals[0] if num_signals > 0 else np.zeros(
                (target_length, self.ecg_num_leads), dtype=np.float32
            )
            # signal_out: single mode -> [leads, length]; multi mode -> [N, leads, length].
            if self._multi_ecg:
                if num_signals > 0:
                    signal_out = np.stack([np.transpose(w, (1, 0)) for w in signals], axis=0)
                else:
                    signal_out = np.zeros((0, self.ecg_num_leads, target_length), dtype=np.float32)
            else:
                signal_out = np.transpose(waveform, (1, 0))

            # Tokenization logic
            if self.instruct_mode:
                # --- Chat messages parsing (overrides prompt/answer columns) ---
                _parsed_msgs: Optional[Dict[str, str]] = None
                if has_messages:
                    _parsed_msgs = self._parse_messages(row[self.messages_column])

                if _parsed_msgs is not None:
                    # Extract structured content from the chat messages
                    system_message = _parsed_msgs.get(
                        "system",
                        "You are an expert cardiologist. You interpret ECGs and answer in a concise, structured way.",
                    )
                    user_content_raw = _parsed_msgs.get("user", "")
                    answer_text = _parsed_msgs["assistant"]
                    # Adapt image placeholder to the model's expected format
                    if self.medgemma_prompt_style:
                        user_content = user_content_raw.replace("<image>", "<start_of_image>")
                        if "<start_of_image>" not in user_content:
                            user_content = "<start_of_image>\n\n" + user_content
                    else:
                        user_content = user_content_raw.replace("<image>", "<image_1>")
                        if "<image_1>" not in user_content:
                            user_content = "<image_1> " + user_content
                    prompt_text = user_content_raw
                    is_cf_record = False
                    debug_payload = {
                        "mode": "chat_messages",
                        "question": prompt_text,
                        "canonical_answer": answer_text[:200],
                    }
                else:
                    # --- Standard prompt/answer column path (original logic) ---
                    system_message = None  # sentinel: set below per branch
                    user_content = None
                    is_cf_record = False
                    debug_payload = None

                # Fall through to original prompt/answer handling when messages were not used
                if _parsed_msgs is None:
                    prompt_text: str = ""
                    if self.prompt_column in self.df.columns and not pd.isnull(row[self.prompt_column]):
                        prompt_text = str(row[self.prompt_column])
                    # Randomly sample a prompt variation matched to the original prompt
                    if self._prompt_variations and self.category_column in self.df.columns:
                        cat = str(row.get(self.category_column, ""))
                        if cat in self._prompt_variations and prompt_text in self._prompt_variations[cat]:
                            import random
                            prompt_text = random.choice(self._prompt_variations[cat][prompt_text])
                    answer_text: str = str(row[self.answer_column])

                    candidate_answers_raw = row.get("candidate_answers")
                    gt_indices_raw = row.get("ground_truth_indices")
                    has_candidates = (
                        candidate_answers_raw is not None
                        and not (isinstance(candidate_answers_raw, float) and np.isnan(candidate_answers_raw))
                    )
                    is_cf_record = self.medgemma_prompt_style and has_candidates
                    options: list[str] = []
                    if is_cf_record:
                        try:
                            options = [str(x) for x in list(candidate_answers_raw)]
                        except Exception:
                            options = []
                        if not options:
                            is_cf_record = False

                if _parsed_msgs is None and is_cf_record:
                    candidate_answers = options
                    system_message = "You are an expert cardiologist. You interpret ECGs and answer in a concise, structured way."
                    letters = [chr(ord("A") + i) for i in range(min(len(options), 26))]
                    gt_indices: list[int] = []
                    if gt_indices_raw is not None and not (isinstance(gt_indices_raw, float) and np.isnan(gt_indices_raw)):
                        try:
                            gt_indices = [int(x) for x in list(gt_indices_raw)]
                        except Exception:
                            gt_indices = []
                    canonical_letters = sorted({
                        letters[i] for i in gt_indices if 0 <= i < len(letters)
                    })
                    canonical_answer = ",".join(canonical_letters) if canonical_letters else ""
                    multi_label = len(canonical_letters) > 1
                    selection_text = (
                        "Select ALL applicable options from the list below."
                        if multi_label else
                        "Select the single best option from the list below."
                    )
                    options_block = "\n".join(
                        f"{ltr}. {opt}" for ltr, opt in zip(letters, options)
                    )
                    if not prompt_text:
                        prompt_text = "What is the primary finding on this ECG?"
                    user_content = (
                        "<start_of_image>\n\n"
                        f"Question: {prompt_text}\n\n"
                        f"{selection_text}\n"
                        "Respond ONLY with the letters of the correct options in ascending order, separated by commas, and nothing else.\n\n"
                        f"{options_block}"
                    )
                    answer_text = canonical_answer if canonical_answer else answer_text
                    debug_payload = {
                        "mode": "medgemma_cf",
                        "question": prompt_text,
                        "options": options,
                        "letters": letters[:len(options)],
                        "gt_indices": gt_indices,
                        "canonical_answer": answer_text,
                    }
                elif _parsed_msgs is None and self.medgemma_prompt_style:
                    system_message = "You are an expert cardiologist. You interpret ECGs and answer in a concise, structured way."
                    if not prompt_text:
                        prompt_text = "Analyze this ECG and list the clinical findings."
                    user_content = (
                        "<start_of_image>\n\n"
                        f"Question: {prompt_text}\n\n"
                        "Respond concisely with the key finding or answer."
                    )
                    debug_payload = {
                        "mode": "medgemma_instruct",
                        "question": prompt_text,
                        "canonical_answer": answer_text,
                    }
                elif _parsed_msgs is None:
                    # Construct LLaMA 3.2 chat template with ECG integration
                    # System message for ECG analysis task - optimized for concise medical findings
                    system_message = "An electrocardiogram analysis and question answering tool"
                    if not prompt_text:
                        prompt_text = "Analyze this ECG and list the clinical findings."
                    user_content = f"<image_1> {prompt_text}".strip()
                    debug_payload = {
                        "mode": "default_instruct",
                        "question": prompt_text,
                        "answer_text": answer_text,
                    }

                # Multi-ECG: force exactly num_signals <start_of_image> anchors (one per ECG).
                # The decoder splices one ECG embedding block after each anchor. N=0 removes
                # the anchor (no-ECG / adversarial rows).
                if self._multi_ecg and user_content is not None:
                    user_content = self._set_num_image_tokens(user_content, num_signals)

                # Build rendered prompts
                if self.medgemma_prompt_style:
                    # Explicitly construct Gemma-style turns; let tokenizer add BOS/EOS
                    prompt_template_text = (
                        "<start_of_turn>system\n"
                        f"{system_message}<end_of_turn>\n"
                        "<start_of_turn>user\n"
                        f"{user_content}<end_of_turn>\n"
                        "<start_of_turn>model\n"
                    )
                    full_template_text = (
                        "<start_of_turn>system\n"
                        f"{system_message}<end_of_turn>\n"
                        "<start_of_turn>user\n"
                        f"{user_content}<end_of_turn>\n"
                        f"<start_of_turn>model\n{answer_text}<end_of_turn>"
                    )
                else:
                    # Create messages for chat template
                    messages_prompt = [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_content}
                    ]
                    
                    messages_full = [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": answer_text}
                    ]
                    
                    # Apply chat template
                    prompt_template_text = cast(str, self._pt_tokenizer.apply_chat_template(
                        messages_prompt, 
                        tokenize=False, 
                        add_generation_prompt=True
                    ))
                    full_template_text = cast(str, self._pt_tokenizer.apply_chat_template(
                        messages_full, 
                        tokenize=False, 
                        add_generation_prompt=False
                    ))

                do_debug = self.debug_print_example and not self._debug_example_printed
                if do_debug:
                    print("=== Debug: Prompt template ===")
                    print(prompt_template_text)
                    print("=== Debug: Full template ===")
                    print(full_template_text)
                
                # Tokenize both (use __call__ for compatibility with TokenizersBackend)
                _encode = getattr(self._pt_tokenizer, 'encode_plus', None) or self._pt_tokenizer
                prompt_encoding = _encode(
                    prompt_template_text,
                    add_special_tokens=True,
                    return_tensors=None
                )
                full_encoding = _encode(
                    full_template_text,
                    add_special_tokens=True,
                    return_tensors=None
                )

                prompt_ids = prompt_encoding.input_ids
                full_ids = full_encoding.input_ids

                if do_debug:
                    image_token_id = None
                    try:
                        image_token_id = self._pt_tokenizer.convert_tokens_to_ids("<start_of_image>")
                    except Exception:
                        image_token_id = None
                    if isinstance(image_token_id, (list, tuple)):
                        image_token_id = image_token_id[0] if image_token_id else None
                    image_pos = None
                    if image_token_id is not None:
                        try:
                            image_pos = full_ids.index(int(image_token_id))
                        except ValueError:
                            image_pos = None
                    print(
                        f"=== Debug: <start_of_image> token id: {image_token_id}, pos in full_ids: {image_pos} ==="
                    )
                    self._debug_example_printed = True

                # Ensure assistant turn terminates with <eos> for stable stopping behavior.
                # Some templates/tokenizers may omit or place EOT beyond truncation; enforce it here.
                eos_int: Optional[int] = None
                if not self.medgemma_prompt_style:
                    try:
                        eos = self._pt_tokenizer.convert_tokens_to_ids("<eos>")
                    except Exception:
                        eos = None
                    if isinstance(eos, (list, tuple)):
                        eos = eos[0] if eos else None
                    try:
                        eos_int = int(eos) if eos is not None else None
                    except (TypeError, ValueError):
                        eos_int = None
                    # Append EOT if missing and if token is known
                    if eos_int is not None and eos_int >= 0:
                        if not full_ids or full_ids[-1] != eos_int:
                            full_ids = list(full_ids) + [eos_int]

                # Build ECG prefix tokens
                if self.num_ecg_tokens > 0:
                    if self.prefix_tuning:
                        ecg_prefix = torch.full(
                            (self.num_ecg_tokens,),
                            fill_value=self.ecg_prefix_token_id,
                            dtype=torch.long,
                        )
                    else:
                        ecg_prefix = torch.arange(
                            self.ecg_token_start_id,
                            self.ecg_token_start_id + self.num_ecg_tokens,
                            dtype=torch.long,
                        )
                else:
                    ecg_prefix = torch.zeros(0, dtype=torch.long)
                prefix_len = ecg_prefix.numel()

                # Truncate text so total length fits within max_length after ECG prefix.
                # If truncation would drop the final EOT token, replace the last kept token with EOT.
                max_text_len = max(0, self.max_length - prefix_len)
                full_ids_trunc = full_ids[:max_text_len]
                if (
                    eos_int is not None
                    and eos_int >= 0
                    and max_text_len > 0
                    and full_ids
                    and full_ids[-1] == eos_int
                    and full_ids_trunc[-1] != eos_int
                ):
                    # Force last token to EOT to preserve terminator within context window
                    full_ids_trunc = list(full_ids_trunc)
                    full_ids_trunc[-1] = eos_int

                # Construct final input_ids: place ECG prefix immediately after BOS for MedGemma; otherwise prepend
                text_ids = torch.tensor(full_ids_trunc, dtype=torch.long)
                bos_id = getattr(self._pt_tokenizer, "bos_token_id", None)
                if isinstance(bos_id, (list, tuple)):
                    bos_id = bos_id[0] if bos_id else None
                insert_after_bos = (
                    self.medgemma_prompt_style
                    and prefix_len > 0
                    and bos_id is not None
                    and text_ids.numel() > 0
                    and text_ids[0].item() == bos_id
                )
                insert_after_image = False
                image_pos = None
                if self.medgemma_prompt_style and prefix_len > 0:
                    try:
                        image_token_id = self._pt_tokenizer.convert_tokens_to_ids("<start_of_image>")
                        if isinstance(image_token_id, (list, tuple)):
                            image_token_id = image_token_id[0] if image_token_id else None
                    except Exception:
                        image_token_id = None
                    if image_token_id is not None:
                        try:
                            image_pos = text_ids.tolist().index(int(image_token_id))
                            insert_after_image = True
                        except ValueError:
                            insert_after_image = False

                if insert_after_image and image_pos is not None:
                    input_ids = torch.cat([text_ids[:image_pos + 1], ecg_prefix, text_ids[image_pos + 1:]], dim=0)
                elif insert_after_bos:
                    input_ids = torch.cat([text_ids[:1], ecg_prefix, text_ids[1:]], dim=0)
                else:
                    input_ids = torch.cat([ecg_prefix, text_ids], dim=0)

                # Create attention mask and pad to max_length
                attention_mask = torch.ones_like(input_ids, dtype=torch.long)
                pad_id = self.ecg_prefix_token_id
                if input_ids.numel() < self.max_length:
                    pad_len = self.max_length - input_ids.numel()
                    # Now pad_id is guaranteed to exist
                    input_ids = F.pad(input_ids, (0, pad_len), value=pad_id)
                    attention_mask = F.pad(attention_mask, (0, pad_len), value=0)

                # Prepare prompt-only ids and mask for generation (text-only; no ECG tokens)
                prompt_input_ids = torch.tensor(prompt_ids[: self.max_length], dtype=torch.long)
                prompt_attention_mask = torch.ones_like(prompt_input_ids, dtype=torch.long)
                if prompt_input_ids.numel() < self.max_length:
                    pad_len = self.max_length - prompt_input_ids.numel()
                    prompt_input_ids = F.pad(prompt_input_ids, (0, pad_len), value=pad_id)
                    prompt_attention_mask = F.pad(prompt_attention_mask, (0, pad_len), value=0)

                # Create labels: ignore prompt (question + assistant header) and padding
                prompt_len = len(prompt_ids)
                text_prompt_len = len(prompt_encoding.input_ids)
                full_prompt_len = min(self.max_length, prefix_len + text_prompt_len)
                # prompt_len = min(len(prompt_ids), self.max_length)
                labels = input_ids.clone()
                # labels[:prompt_len] = -100
                labels[:full_prompt_len] = -100
                labels = labels.masked_fill(attention_mask == 0, -100)

                # Add category information if available
                waveform_name = row.get('waveform_name')
                if pd.isnull(waveform_name):
                    waveform_name = os.path.basename(str(row[self.signal_path_column]))
                sample_data = {
                    'signal': signal_out,
                    'num_ecgs': num_signals,
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'prompt_input_ids': prompt_input_ids,
                    'prompt_attention_mask': prompt_attention_mask,
                    'labels': labels,
                    'waveform_name': waveform_name,
                    'prompt_text': prompt_text,  # Add original prompt for metrics display
                    'rendered_prompt': prompt_template_text,
                    'task_type': 'cf' if is_cf_record else 'instruct',
                    'answer_text': answer_text,  # Raw ground truth for metrics
                }
                
                # Add category information for per-category metrics
                if self.category_column in self.df.columns and not pd.isnull(row[self.category_column]):
                    sample_data['prompt_category'] = str(row[self.category_column])
                    # Add ground truth LVEF for soft-decoding loss
                    if str(row[self.category_column]) == 'lvef':
                        lvef_col = 'deepecho_Visually_Estimated_EF'
                        if lvef_col in self.df.columns and not pd.isnull(row.get(lvef_col)):
                            sample_data['lvef_gt'] = float(row[lvef_col])
                        else:
                            sample_data['lvef_gt'] = float('nan')
                    else:
                        sample_data['lvef_gt'] = float('nan')
                else:
                    sample_data['lvef_gt'] = float('nan')

                if self.pattern_columns:
                    pattern_values = []
                    for col, present in zip(self.pattern_columns, self._pattern_column_mask):
                        if present:
                            value = row[col]
                            if pd.isnull(value):
                                value = 0.0
                            pattern_values.append(float(value))
                        else:
                            pattern_values.append(0.0)
                    sample_data['pattern_targets'] = torch.tensor(pattern_values, dtype=torch.float32)

                # Auxiliary bridge-head targets (masked; NaN where the row lacks the label)
                sample_data['aux_lvef_gt'] = self._aux_head_gt(row, self.lvef_head_column)
                sample_data['aux_shd_gt'] = self._aux_head_gt(row, self.shd_head_column)
                sample_data['aux_afib_gt'] = self._aux_head_gt(row, self.afib_head_column)

                return sample_data
            else:
                # CF/QA non-instruction mode: build a simple prompt/answer pair
                # Use dataset-provided question as prompt; model learns to complete the answer
                prompt_q = ""
                if self.prompt_column in self.df.columns and not pd.isnull(row[self.prompt_column]):
                    prompt_q = str(row[self.prompt_column]).strip()
                # Randomly sample a prompt variation matched to the original prompt
                if self._prompt_variations and self.category_column in self.df.columns:
                    cat = str(row.get(self.category_column, ""))
                    if cat in self._prompt_variations and prompt_q in self._prompt_variations[cat]:
                        import random
                        prompt_q = random.choice(self._prompt_variations[cat][prompt_q])
                answer_text: str = str(row[self.answer_column])

                # Minimal template for non-chat tokenization
                prompt_template_text = (
                    f"Question: {prompt_q}\nAnswer:" if prompt_q else "Question: What is the primary ECG finding?\nAnswer:"
                )
                full_template_text = f"{prompt_template_text} {answer_text}".strip()

                # Tokenize prompt (for generation context) and full text (for labels)
                _encode = getattr(self._pt_tokenizer, 'encode_plus', None) or self._pt_tokenizer
                prompt_enc: BatchEncoding = _encode(
                    prompt_template_text,
                    add_special_tokens=True,
                    return_tensors=None
                )
                full_enc: BatchEncoding = _encode(
                    full_template_text,
                    add_special_tokens=True,
                    return_tensors=None
                )

                prompt_ids = prompt_enc.get('input_ids', [])
                full_ids = full_enc.get('input_ids', [])

                # ECG prefix placeholders (usually 0 for Q-Former bridges)
                if self.num_ecg_tokens > 0:
                    if self.prefix_tuning:
                        ecg_prefix = torch.full(
                            (self.num_ecg_tokens,),
                            fill_value=self.ecg_prefix_token_id,
                            dtype=torch.long,
                        )
                    else:
                        ecg_prefix = torch.arange(
                            self.ecg_token_start_id,
                            self.ecg_token_start_id + self.num_ecg_tokens,
                            dtype=torch.long,
                        )
                else:
                    ecg_prefix = torch.zeros(0, dtype=torch.long)
                prefix_len = ecg_prefix.numel()

                # Truncate combined text to fit within max_length
                max_text_len = max(0, self.max_length - prefix_len)
                full_ids_trunc = full_ids[:max_text_len]

                text_ids = torch.tensor(full_ids_trunc, dtype=torch.long)
                input_ids = torch.cat([ecg_prefix, text_ids], dim=0)

                attention_mask = torch.ones_like(input_ids, dtype=torch.long)
                pad_id = self.ecg_prefix_token_id
                if input_ids.numel() < self.max_length:
                    pad_len = self.max_length - input_ids.numel()
                    input_ids = F.pad(input_ids, (0, pad_len), value=pad_id)
                    attention_mask = F.pad(attention_mask, (0, pad_len), value=0)

                # Prompt-only ids for generation (no ECG textual tokens)
                prompt_input_ids = torch.tensor(prompt_ids[: self.max_length], dtype=torch.long)
                prompt_attention_mask = torch.ones_like(prompt_input_ids, dtype=torch.long)
                if prompt_input_ids.numel() < self.max_length:
                    pad_len = self.max_length - prompt_input_ids.numel()
                    prompt_input_ids = F.pad(prompt_input_ids, (0, pad_len), value=pad_id)
                    prompt_attention_mask = F.pad(prompt_attention_mask, (0, pad_len), value=0)

                # Create labels: mask out the prompt tokens
                full_prompt_len = min(self.max_length, prefix_len + len(prompt_ids))
                labels = input_ids.clone()
                labels[:full_prompt_len] = -100
                labels = labels.masked_fill(attention_mask == 0, -100)

                # Add category and prompt info
                waveform_name = row.get('waveform_name')
                if pd.isnull(waveform_name):
                    waveform_name = os.path.basename(str(row[self.signal_path_column]))
                sample_data = {
                    'signal': signal_out,
                    'num_ecgs': num_signals,
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'prompt_input_ids': prompt_input_ids,
                    'prompt_attention_mask': prompt_attention_mask,
                    'labels': labels,
                    'waveform_name': waveform_name,
                    'prompt_text': prompt_q,
                    'rendered_prompt': prompt_template_text,
                    'task_type': 'instruct',
                    'answer_text': answer_text,  # Raw ground truth for metrics
                }

                if self.category_column in self.df.columns and not pd.isnull(row[self.category_column]):
                    sample_data['prompt_category'] = str(row[self.category_column])
                    if str(row[self.category_column]) == 'lvef':
                        lvef_col = 'deepecho_Visually_Estimated_EF'
                        if lvef_col in self.df.columns and not pd.isnull(row.get(lvef_col)):
                            sample_data['lvef_gt'] = float(row[lvef_col])
                        else:
                            sample_data['lvef_gt'] = float('nan')
                    else:
                        sample_data['lvef_gt'] = float('nan')
                else:
                    sample_data['lvef_gt'] = float('nan')

                if self.pattern_columns:
                    pattern_values = []
                    for col, present in zip(self.pattern_columns, self._pattern_column_mask):
                        if present:
                            value = row[col]
                            if pd.isnull(value):
                                value = 0.0
                            pattern_values.append(float(value))
                        else:
                            pattern_values.append(0.0)
                    sample_data['pattern_targets'] = torch.tensor(pattern_values, dtype=torch.float32)

                # Auxiliary bridge-head targets (masked; NaN where the row lacks the label)
                sample_data['aux_lvef_gt'] = self._aux_head_gt(row, self.lvef_head_column)
                sample_data['aux_shd_gt'] = self._aux_head_gt(row, self.shd_head_column)
                sample_data['aux_afib_gt'] = self._aux_head_gt(row, self.afib_head_column)

                return sample_data
            
        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            return None
            
            
def get_clinical_report_dataloader(
    config: LLMFinetuningConfig,
    shuffle: bool = True,
    pin_memory: bool = True,
    subset_size: Optional[int] = None,
    balance_categories: bool = False,
    sampling_seed: Optional[int] = None,
    medgemma_prompt_style: bool = False,
    debug_print_example: bool = False,
):
    max_length = getattr(config, "max_length", getattr(config, "max_token_length", 512))
    dataset: ECGClinicalReportDataset = ECGClinicalReportDataset(
        dataset_path=config.train_dataset_path, 
        signal_path_column=config.signal_path_column,
        ecg_waveform_length=config.ecg_waveform_length,
        ecg_num_leads=config.ecg_num_leads,
        tokenizer=config.tokenizer, 
        max_length=int(max_length),
        instruct_mode=getattr(config, 'instruct_mode', False),
        num_ecg_tokens=config.num_ecg_tokens,
        ecg_token_start_id=config.ecg_token_start_id,
        prompt_column=getattr(config, 'prompt_column', 'question'),
        answer_column=getattr(config, 'answer_column', 'report'),
        category_column=getattr(config, 'category_column', 'prompt_category'),
        prefix_tuning=getattr(config, 'prefix_tuning', False),
        pattern_columns=getattr(config, 'pattern_label_columns', None),
        lvef_head_column=getattr(config, 'lvef_head_label_column', 'deepecho_Visually_Estimated_EF'),
        shd_head_column=getattr(config, 'shd_head_label_column', 'echonext_shd_binary'),
        afib_head_column=getattr(config, 'afib_head_label_column', 'afib_label_5y'),
        medgemma_prompt_style=medgemma_prompt_style,
        debug_print_example=debug_print_example,
    )
    dataset = _maybe_subset_dataset(
        dataset=dataset,
        subset_size=subset_size,
        balance_categories=balance_categories,
        category_column=getattr(config, 'category_column', 'prompt_category'),
        seed=sampling_seed,
    )
    return DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=shuffle, 
        num_workers=config.num_workers, 
        pin_memory=pin_memory,
        persistent_workers=True if getattr(config, "num_workers", 0) else False,
        collate_fn=custom_collate_fn
    )
    
def get_distributed_clinical_report_dataloader(
    dataset_path: str,
    signal_path_column: str,
    ecg_waveform_length: int,
    ecg_num_leads: int,
    tokenizer: AutoTokenizerT,
    max_token_length: int = 512,
    batch_size: int = 32,
    num_workers: int = 16,
    num_replicas: int = 1,
    rank: int = 0,
    shuffle: bool = True,
    pin_memory: bool = True,
    instruct_mode: bool = False,
    num_ecg_tokens: int = 128,
    ecg_token_start_id: Optional[int] = None,
    prompt_column: str = "question",
    answer_column: str = "report",
    category_column: str = "prompt_category",
    prefix_tuning: bool = False,
    pattern_columns: Optional[Sequence[str]] = None,
    subset_size: Optional[int] = None,
    balance_categories: bool = False,
    sampling_seed: Optional[int] = None,
    medgemma_prompt_style: bool = False,
    debug_print_example: bool = False,
    sample_weight_column: Optional[str] = None,
    augmentor: Optional[Any] = None,
    messages_column: Optional[str] = None,
    prompt_variations_path: Optional[str] = None,
    signal_paths_column: Optional[str] = None,
    max_ecgs: int = 8,
):
    """
    Create a distributed DataLoader for ECG clinical report training.

    Args:
        sample_weight_column: Optional column name containing per-sample weights
            for weighted sampling. When provided, enables minority class upsampling
            using WeightedDistributedSampler instead of standard DistributedSampler.
        ... (other args documented elsewhere)
    """
    dataset: ECGClinicalReportDataset = ECGClinicalReportDataset(
        dataset_path=dataset_path,
        signal_path_column=signal_path_column,
        ecg_waveform_length=ecg_waveform_length,
        ecg_num_leads=ecg_num_leads,
        tokenizer=tokenizer,
        max_length=max_token_length,
        instruct_mode=instruct_mode,
        num_ecg_tokens=num_ecg_tokens,
        ecg_token_start_id=ecg_token_start_id,
        prompt_column=prompt_column,
        answer_column=answer_column,
        category_column=category_column,
        prefix_tuning=prefix_tuning,
        pattern_columns=pattern_columns,
        medgemma_prompt_style=medgemma_prompt_style,
        debug_print_example=debug_print_example,
        augmentor=augmentor,
        messages_column=messages_column,
        prompt_variations_path=prompt_variations_path,
        signal_paths_column=signal_paths_column,
        max_ecgs=max_ecgs,
    )

    # Extract sample weights before any subsetting
    sample_weights = None
    if sample_weight_column and sample_weight_column in dataset.df.columns:
        sample_weights = dataset.df[sample_weight_column].values.tolist()
        if rank == 0:
            import logging
            logging.info(
                f"Weighted sampling enabled: {sum(w > 1.0 for w in sample_weights):,} "
                f"samples with weight > 1.0 (mean weight: {sum(sample_weights)/len(sample_weights):.3f})"
            )
            # Debug: print value counts for sample weights
            weight_counts = dataset.df[sample_weight_column].value_counts().sort_index()
            logging.info(f"Sample weight value counts:\n{weight_counts.to_string()}")

    dataset = _maybe_subset_dataset(
        dataset=dataset,
        subset_size=subset_size,
        balance_categories=balance_categories,
        category_column=category_column,
        seed=sampling_seed,
    )

    # If dataset was subsetted, we need to extract the subset weights
    if sample_weights is not None and subset_size is not None and subset_size > 0:
        # If subsetting was applied, the dataset is now a Subset
        # We need to get weights for only the selected indices
        from torch.utils.data import Subset
        if isinstance(dataset, Subset):
            sample_weights = [sample_weights[i] for i in dataset.indices]

    return DistributedUtils.get_distributed_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        num_replicas=num_replicas,
        rank=rank,
        shuffle=shuffle,
        collate_fn=custom_collate_fn,
        sample_weights=sample_weights,
        weighted_sampling_seed=sampling_seed or 42,
    )


def _maybe_subset_dataset(
    dataset: ECGClinicalReportDataset,
    subset_size: Optional[int],
    balance_categories: bool,
    category_column: str,
    seed: Optional[int],
) -> Dataset:
    total_size = len(dataset)
    if subset_size is None or subset_size <= 0 or subset_size >= total_size:
        return dataset

    rng = np.random.default_rng(seed)
    if not balance_categories or category_column not in dataset.df.columns:
        chosen = rng.choice(total_size, size=subset_size, replace=False)
        subset = Subset(dataset, chosen.tolist())
        return _attach_dataset_attributes(subset, dataset)

    category_series = dataset.df[category_column].fillna("unknown").astype(str)
    category_indices: dict[str, list[int]] = {}
    for idx, label in enumerate(category_series):
        category_indices.setdefault(label, []).append(idx)

    unique_categories = list(category_indices.keys())
    if subset_size < len(unique_categories):
        warnings.warn(
            "Requested subset_size is smaller than the number of prompt categories; "
            "falling back to random sampling without balancing.",
            stacklevel=2,
        )
        chosen = rng.choice(total_size, size=subset_size, replace=False)
        subset = Subset(dataset, chosen.tolist())
        return _attach_dataset_attributes(subset, dataset)

    total_records = float(total_size)
    counts: dict[str, int] = {}
    remainders: dict[str, float] = {}
    capacities: dict[str, int] = {}

    for category, indices in category_indices.items():
        proportion = len(indices) / total_records
        raw_target = proportion * subset_size
        base = int(np.floor(raw_target))
        remainder = raw_target - base
        base = min(base, len(indices))
        if base == 0:
            base = 1
            remainder = 0.0
        counts[category] = base
        remainders[category] = remainder
        capacities[category] = len(indices) - base

    assigned = sum(counts.values())
    if assigned > subset_size:
        excess = assigned - subset_size
        adjustable = sorted(
            (cat for cat in counts if counts[cat] > 1),
            key=lambda cat: (remainders.get(cat, 0.0), counts[cat]),
        )
        idx = 0
        while excess > 0 and adjustable:
            cat = adjustable[idx % len(adjustable)]
            if counts[cat] > 1:
                counts[cat] -= 1
                capacities[cat] += 1
                excess -= 1
                if counts[cat] <= 1:
                    adjustable = [c for c in adjustable if counts[c] > 1]
            else:
                adjustable = [c for c in adjustable if counts[c] > 1]
            idx += 1

    assigned = sum(counts.values())
    if assigned < subset_size:
        deficit = subset_size - assigned
        priority = sorted(counts.keys(), key=lambda cat: remainders.get(cat, 0.0), reverse=True)
        idx = 0
        while deficit > 0 and priority:
            cat = priority[idx % len(priority)]
            if capacities[cat] > 0:
                counts[cat] += 1
                capacities[cat] -= 1
                deficit -= 1
            idx += 1
            if idx > len(priority) * 2:
                extras = [c for c in counts if capacities[c] > 0]
                if not extras:
                    break
                cat = extras[0]
                counts[cat] += 1
                capacities[cat] -= 1
                deficit -= 1
                idx = 0

    if sum(counts.values()) != subset_size:
        warnings.warn(
            "Unable to achieve balanced category sampling for the requested subset size; "
            "falling back to random sampling.",
            stacklevel=2,
        )
        chosen = rng.choice(total_size, size=subset_size, replace=False)
        subset = Subset(dataset, chosen.tolist())
        return _attach_dataset_attributes(subset, dataset)

    selected_indices: list[int] = []
    for category, indices in category_indices.items():
        sample_count = counts.get(category, 0)
        if sample_count <= 0:
            continue
        sampled = rng.choice(indices, size=sample_count, replace=False)
        selected_indices.extend(sampled.tolist())

    rng.shuffle(selected_indices)
    subset = Subset(dataset, selected_indices)
    return _attach_dataset_attributes(subset, dataset)


def _attach_dataset_attributes(subset: Subset, base_dataset: ECGClinicalReportDataset) -> Subset:
    """Propagate commonly accessed attributes from the base dataset onto a Subset."""
    transferable_attrs = (
        "tokenizer",
        "_pt_tokenizer",
        "df",
        "prefix_tuning",
        "num_ecg_tokens",
        "prompt_column",
        "answer_column",
        "category_column",
        "pattern_columns",
    )
    for attr in transferable_attrs:
        if hasattr(base_dataset, attr):
            setattr(subset, attr, getattr(base_dataset, attr))
    return subset

def get_multi_dataset_distributed_dataloader(
    dataset_paths: List[str],
    dataset_weights: List[float],
    signal_path_column: str,
    ecg_waveform_length: int,
    ecg_num_leads: int,
    tokenizer: AutoTokenizerT,
    max_token_length: int = 512,
    batch_size: int = 32,
    num_workers: int = 16,
    num_replicas: int = 1,
    rank: int = 0,
    shuffle: bool = True,
    pin_memory: bool = True,
    instruct_mode: bool = False,
    num_ecg_tokens: int = 0,
    ecg_token_start_id: Optional[int] = None,
    prompt_column: str = "prompt",
    answer_column: str = "generated_answer",
    category_column: str = "prompt_category",
    prefix_tuning: bool = False,
    pattern_columns: Optional[Sequence[str]] = None,
    medgemma_prompt_style: bool = False,
    debug_print_example: bool = False,
    augmentor: Optional[Any] = None,
    sampling_seed: Optional[int] = None,
    messages_column: Optional[str] = None,
    prompt_variations_path: Optional[str] = None,
    signal_paths_column: Optional[str] = None,
    max_ecgs: int = 8,
) -> DataLoader:
    """Create a distributed DataLoader from multiple dataset parquet files.

    Each dataset is instantiated as a separate ECGClinicalReportDataset and
    then combined via ConcatDataset.  Per-sample weights are derived from
    ``dataset_weights`` so that WeightedDistributedSampler balances across
    datasets during training.
    """
    import logging

    datasets: List[ECGClinicalReportDataset] = []
    for i, path in enumerate(dataset_paths):
        ds = ECGClinicalReportDataset(
            dataset_path=path,
            signal_path_column=signal_path_column,
            ecg_waveform_length=ecg_waveform_length,
            ecg_num_leads=ecg_num_leads,
            tokenizer=tokenizer,
            max_length=max_token_length,
            instruct_mode=instruct_mode,
            num_ecg_tokens=num_ecg_tokens,
            ecg_token_start_id=ecg_token_start_id,
            prompt_column=prompt_column,
            answer_column=answer_column,
            category_column=category_column,
            prefix_tuning=prefix_tuning,
            pattern_columns=pattern_columns,
            medgemma_prompt_style=medgemma_prompt_style,
            debug_print_example=(debug_print_example and i == 0),
            augmentor=augmentor,
            messages_column=messages_column,
            prompt_variations_path=prompt_variations_path,
            signal_paths_column=signal_paths_column,
            max_ecgs=max_ecgs,
        )
        datasets.append(ds)
        if rank == 0:
            logging.info(f"[MultiDataset] Dataset {i}: {path} -> {len(ds)} samples (weight={dataset_weights[i]:.2f})")

    concat_dataset = ConcatDataset(datasets)

    # Propagate tokenizer attrs so downstream code (e.g. _compute_metrics)
    # can find them on ConcatDataset the same way as on a single dataset.
    concat_dataset.tokenizer = datasets[0].tokenizer          # type: ignore[attr-defined]
    concat_dataset._pt_tokenizer = datasets[0]._pt_tokenizer  # type: ignore[attr-defined]

    # Build per-sample weights from dataset-level weights
    sample_weights: List[float] = []
    for ds, weight in zip(datasets, dataset_weights):
        sample_weights.extend([weight] * len(ds))

    if rank == 0:
        logging.info(
            f"[MultiDataset] Total: {len(concat_dataset)} samples across {len(datasets)} datasets"
        )

    return DistributedUtils.get_distributed_dataloader(
        dataset=concat_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        num_replicas=num_replicas,
        rank=rank,
        shuffle=shuffle,
        collate_fn=custom_collate_fn,
        sample_weights=sample_weights,
        weighted_sampling_seed=sampling_seed or 42,
    )


def custom_collate_fn(batch):
    """
    Custom collate function which filters out any None items in the batch.

    Multi-ECG mode: each item's 'signal' is [N_i, leads, length] with variable N_i.
    We flat-concat all ECGs into [sum(N_i), leads, length] and emit 'ecg_counts'
    so the decoder can distribute the spliced blocks per row. Single-ECG mode
    ('signal' is 2D [leads, length]) is collated exactly as before (no ecg_counts).
    """
    filtered_batch = [item for item in batch if item is not None]
    if len(filtered_batch) == 0:
        raise ValueError("All items in the batch were invalid. Check dataset integrity or file paths.")

    multi = any(np.asarray(item['signal']).ndim == 3 for item in filtered_batch)
    if not multi:
        return default_collate(filtered_batch)

    # Multi-ECG path: pull out signals + num_ecgs, default_collate the rest.
    signals = []
    ecg_counts = []
    rest = []
    for item in filtered_batch:
        sig = np.asarray(item['signal'])
        if sig.ndim == 2:  # tolerate a single-ECG item mixed in
            sig = sig[None, ...]
        n = int(item.get('num_ecgs', sig.shape[0]))
        if sig.shape[0] > 0:
            signals.append(torch.as_tensor(sig, dtype=torch.float32))
        ecg_counts.append(n)
        rest.append({k: v for k, v in item.items() if k not in ('signal', 'num_ecgs')})

    collated = default_collate(rest)
    if signals:
        collated['signal'] = torch.cat(signals, dim=0)  # [sum(N_i), leads, length]
    else:
        # whole batch is no-ECG; emit an empty signal tensor
        collated['signal'] = torch.zeros((0, 12, 2500), dtype=torch.float32)
    collated['ecg_counts'] = torch.as_tensor(ecg_counts, dtype=torch.long)
    return collated
