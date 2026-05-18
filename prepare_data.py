#!/usr/bin/env python3
"""
================================================================================
Nemotron-Indonesia Data Pipeline
================================================================================

A 4-phase pipeline for preparing Indonesian training data:

  PHASE 1: DOWNLOAD — Fetch datasets from HuggingFace / web
  PHASE 2: CLEAN    — Text normalization, filtering, language detection
  PHASE 3: QUALITY  — NER entity density scoring (optional), length filters
  PHASE 4: PACKAGE  — Deduplication, tokenization, language tagging, save

Usage:
    # Full pipeline with NER quality filter
    python prepare_data.py \
        --output_dir ./data/processed \
        --datasets indo4b_hf cc100 wikipedia seapile \
        --use_ner_filter \
        --ner_model cahya/bert-base-indonesian-NER \
        --quality_threshold 0.6

    # Quick mode (skip NER, just download + clean)
    python prepare_data.py \
        --output_dir ./data/processed \
        --datasets wikipedia liputan6

The pipeline runs entirely on your server. No external API calls except
HuggingFace dataset downloads.
"""

import os
import re
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import hashlib

from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import numpy as np
from tqdm import tqdm
from datasketch import MinHash, MinHashLSH

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


_BUCKET_PROGRESS_COUNTER = None  # mp.Value, set per worker via Pool initializer


def _bucket_init_worker(counter):
    """Pool initializer: store the shared row-counter in each worker."""
    global _BUCKET_PROGRESS_COUNTER
    _BUCKET_PROGRESS_COUNTER = counter


def _bucket_partition_worker(args):
    """Worker for parallel hash-partitioned bucketing in deduplicate().

    Each worker scans the full bands dataset but only buckets rows whose
    band_key hashes to its assigned partition. Different workers handle
    disjoint key spaces, so their results can be concatenated without merging.
    Progress is reported through a shared mp.Value so the main process can
    show a smooth tqdm bar of total rows scanned across all workers.

    Returns: list of buckets (each a list of doc_ids) with len >= 2.
    """
    bands_ds, partition_id, num_partitions = args
    from collections import defaultdict

    local: Dict[bytes, List[int]] = defaultdict(list)
    rows_unreported = 0
    REPORT_EVERY = 500_000  # batch counter updates so the lock isn't hammered

    for batch in bands_ds.iter(batch_size=200_000):
        n = len(batch['doc_id'])
        for did, bk in zip(batch['doc_id'], batch['band_key']):
            if hash(bk) % num_partitions == partition_id:
                local[bk].append(did)
        rows_unreported += n
        if rows_unreported >= REPORT_EVERY and _BUCKET_PROGRESS_COUNTER is not None:
            with _BUCKET_PROGRESS_COUNTER.get_lock():
                _BUCKET_PROGRESS_COUNTER.value += rows_unreported
            rows_unreported = 0
    if rows_unreported and _BUCKET_PROGRESS_COUNTER is not None:
        with _BUCKET_PROGRESS_COUNTER.get_lock():
            _BUCKET_PROGRESS_COUNTER.value += rows_unreported

    # Drop singletons here (workers, in parallel) so main process has less to do.
    return [v for v in local.values() if len(v) > 1]


class NERQualityFilter:
    """
    PHASE 3 COMPONENT: NER-based quality scoring
    
    Uses cahya/bert-base-indonesian-NER to score documents by entity density.
    Documents rich in named entities (people, orgs, locations) are typically
    higher quality (news, encyclopedia) vs entity-sparse text (spam, noise).
    
    This is OPTIONAL. Enable with --use_ner_filter flag.
    """
    
    def __init__(self, model_name: str = "cahya/bert-base-indonesian-NER"):
        logger.info(f"Loading NER model: {model_name}")
        try:
            self.nlp = pipeline(
                "ner",
                model=model_name,
                tokenizer=model_name,
                aggregation_strategy="simple",
                device=-1  # CPU; set to 0 for GPU if available
            )
            self.enabled = True
            logger.info("NER filter loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load NER model: {e}")
            logger.warning("NER quality filter disabled — proceeding without it")
            self.enabled = False
    
    def score_document(self, text: str) -> Tuple[float, List[Dict]]:
        """
        Score a document by entity density.
        
        Returns:
            score: float 0.0-1.0 (entity density ratio)
            entities: list of found entities [{word, entity_group, score}]
        """
        if not self.enabled or not text:
            return 0.0, []
        
        # Truncate very long documents for NER (speed)
        text_for_ner = text[:2000]
        
        try:
            entities = self.nlp(text_for_ner)
            # Deduplicate entities by word
            seen = set()
            unique_entities = []
            for ent in entities:
                key = (ent.get('word', '').lower(), ent.get('entity_group', ''))
                if key not in seen:
                    seen.add(key)
                    unique_entities.append(ent)
            
            # Score = unique_entities / word_count (capped at 1.0)
            word_count = len(text_for_ner.split())
            if word_count == 0:
                return 0.0, []
            
            score = min(len(unique_entities) / (word_count * 0.05), 1.0)
            return score, unique_entities
            
        except Exception as e:
            logger.debug(f"NER scoring failed for document: {e}")
            return 0.0, []
    
    def filter_dataset(self, dataset: Dataset, threshold: float = 0.1) -> Dataset:
        """
        Filter dataset: keep documents with entity_score >= threshold.
        Documents with score < threshold are likely low-quality (spam, noise).
        """
        if not self.enabled:
            logger.info("NER filter disabled, skipping quality scoring")
            return dataset
        
        logger.info(f"Running NER quality filter (threshold={threshold})...")
        scores = []
        
        for idx in tqdm(range(len(dataset)), desc="NER scoring"):
            text = dataset[idx].get('text', '')
            score, _ = self.score_document(text)
            scores.append(score)
        
        # Keep documents above threshold
        keep_mask = [s >= threshold for s in scores]
        kept = sum(keep_mask)
        logger.info(f"NER filter: kept {kept} / {len(dataset)} documents "
                   f"({kept/len(dataset)*100:.1f}%)")
        
        return dataset.select([i for i, keep in enumerate(keep_mask) if keep])


class IndonesianDataProcessor:
    """Process and curate Indonesian text data for LLM training"""

    def __init__(self, tokenizer_name: str = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16",
                 data_dir: str = "./data/raw"):
        self.data_dir = Path(data_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Language identifiers for multilingual training
        self.lang_tokens = {
            'id': '<|id|>',
            'jv': '<|jv|>',  # Javanese
            'su': '<|su|>',  # Sundanese
            'ban': '<|ban|>',  # Balinese
            'min': '<|min|>',  # Minangkabau
            'mad': '<|mad|>',  # Madurese
            'bug': '<|bug|>',  # Buginese
            'ace': '<|ace|>',  # Acehnese
            'bjn': '<|bjn|>',  # Banjarese
            'nij': '<|nij|>',  # Ngaju Dayak
        }
        
        # Add special tokens
        special_tokens = {'additional_special_tokens': list(self.lang_tokens.values())}
        self.tokenizer.add_special_tokens(special_tokens)

    def download_indo4b_hf(self):
        """Load Indo4B HF parquet mirror from local download, falling back to HuggingFace."""
        local_path = self.data_dir / 'indo4b-hf' / 'data'
        if local_path.exists():
            files = sorted(str(f) for f in local_path.glob('*.parquet'))
            if files:
                logger.info(f"Loading Indo4B HF from local: {local_path}")
                try:
                    return load_dataset('parquet', data_files=files, split='train')
                except Exception as e:
                    logger.warning(f"Local load failed, falling back to HF: {e}")

        logger.info("Downloading Indo4B HF mirror: taufiqdp/Indo4B-hf")
        try:
            return load_dataset('taufiqdp/Indo4B-hf', split='train')
        except Exception as e:
            logger.warning(f"Failed to load Indo4B HF mirror: {e}")
            return None
    
    def download_sealion_pile(self, language: str = 'id'):
        """Load SEA-PILE Indonesian subset from local download, falling back to HuggingFace."""
        local_path = self.data_dir / 'sea-pile-id' / 'sea-pile-mc4' / language
        if local_path.exists():
            files = sorted(str(f) for f in local_path.glob('*.jsonl.gz'))
            if files:
                logger.info(f"Loading SEA-PILE from local: {local_path}")
                try:
                    return load_dataset('json', data_files=files, split='train')
                except Exception as e:
                    logger.warning(f"Local load failed, falling back to HF: {e}")

        logger.info(f"Downloading SEA-LION Pile for language: {language}")
        try:
            ds = load_dataset('aisingapore/SEA-PILE-v1', split='train', streaming=True)
            ds = ds.filter(lambda x: x.get('file', '').startswith(f'c4-{language}'))
            logger.info(f"SEA-LION Pile filtered to {language} subset")
            return ds
        except Exception as e:
            logger.warning(f"Failed to load SEA-LION Pile: {e}")
            return None
    
    def download_cc100(self, language: str = 'id', max_examples: int = 0):
        """Load CC100 text corpus. Resolution order:

        1. data/raw/cc100/id.txt      — plain text, used directly (fastest)
        2. data/raw/cc100/id.7z       — auto-extracted to id.txt via py7zr
        3. data/raw/cc100/id.txt.xz   — read via lzma (corrupt-footer safe)

        The original id.txt.xz from statmt.org has a corrupt XZ footer on Linux.
        Workaround: extract on Windows with 7-Zip, repack as id.7z, copy to server.
        Set max_examples > 0 to read a subset.
        """
        import lzma
        from datasets import Dataset as HFDataset

        cc100_dir = self.data_dir / 'cc100'
        txt_file = cc100_dir / f'{language}.txt'
        z7_file  = cc100_dir / f'{language}.7z'
        xz_file  = cc100_dir / f'{language}.txt.xz'

        # Auto-extract .7z → .txt if needed
        if not txt_file.exists() and z7_file.exists():
            logger.info(f"Extracting {z7_file} → {txt_file} ...")
            try:
                import py7zr
                with py7zr.SevenZipFile(z7_file, mode='r') as archive:
                    archive.extractall(path=str(cc100_dir))
                logger.info("Extraction complete.")
            except Exception as e:
                logger.warning(f"Failed to extract {z7_file}: {e}")

        if txt_file.exists():
            cap_msg = f"capped at {max_examples:,}" if max_examples > 0 else "full corpus"
            logger.info(f"Loading CC100 from extracted text: {txt_file} ({cap_msg})")
            try:
                if max_examples > 0:
                    # stream-cap via generator to avoid loading the whole file
                    cache_dir = str(self.data_dir / 'cache')
                    def _gen_txt():
                        count = 0
                        with open(txt_file, 'r', encoding='utf-8') as f:
                            for line in f:
                                if count >= max_examples:
                                    break
                                line = line.rstrip('\n')
                                if line.strip():
                                    yield {'text': line}
                                    count += 1
                    return HFDataset.from_generator(_gen_txt, cache_dir=cache_dir, writer_batch_size=100_000)
                else:
                    return load_dataset('text', data_files=str(txt_file), split='train')
            except Exception as e:
                logger.warning(f"Failed to load extracted CC100: {e}")
                return None

        if not xz_file.exists():
            logger.warning(f"CC100 not found ({txt_file} or {xz_file})")
            logger.warning("Run: python download_sources.py --sources cc100_id")
            return None

        # Fallback: read directly from .xz, catching the corrupt-footer error
        cache_dir = str(self.data_dir / 'cache')
        cap_msg = f"capped at {max_examples:,}" if max_examples > 0 else "full corpus"
        logger.info(f"Loading CC100 from .xz: {xz_file} ({cap_msg}, cache → {cache_dir})")
        try:
            def _gen_xz():
                count = 0
                f = lzma.open(xz_file, 'rt', encoding='utf-8')
                try:
                    for line in f:
                        if max_examples > 0 and count >= max_examples:
                            break
                        line = line.rstrip('\n')
                        if line.strip():
                            yield {'text': line}
                            count += 1
                except lzma.LZMAError as e:
                    logger.warning(f"CC100 lzma stream closed with error (corrupt XZ footer): {e}")
                finally:
                    try:
                        f.close()
                    except lzma.LZMAError:
                        pass
            return HFDataset.from_generator(_gen_xz, cache_dir=cache_dir, writer_batch_size=100_000)
        except Exception as e:
            logger.warning(f"Failed to load CC100 from .xz: {e}")
            return None
    
    def download_wikipedia(self, language: str = 'id'):
        """Load Wikipedia from local download, falling back to HuggingFace."""
        local_path = self.data_dir / 'wikipedia-id' / f'20231101.{language}'
        if local_path.exists():
            files = sorted(str(f) for f in local_path.glob('*.parquet'))
            if files:
                logger.info(f"Loading Wikipedia from local: {local_path}")
                try:
                    return load_dataset('parquet', data_files=files, split='train')
                except Exception as e:
                    logger.warning(f"Local load failed, falling back to HF: {e}")

        logger.info(f"Downloading Wikipedia for language: {language}")
        try:
            return load_dataset('wikimedia/wikipedia', f'20231101.{language}', split='train')
        except Exception as e:
            logger.warning(f"Failed to load Wikipedia: {e}")
            return None
    
    def download_kaskus(self, max_examples: int = 0):
        """Load Kaskus forum corpus from local file (no public HuggingFace source available).

        The local file has 207M lines / 44 GB. Arrow writes the cache to data/cache/
        (same large partition as raw data) so the full corpus can be loaded.
        Set max_examples > 0 to cap the number of lines read.
        """
        import json as _json

        local_file = self.data_dir / 'kaskus' / 'kaskus.jsonl'
        if not local_file.exists():
            logger.warning(f"Kaskus local file not found: {local_file}")
            logger.warning("Provide data/raw/kaskus/kaskus.jsonl to use this dataset")
            return None

        cache_dir = str(self.data_dir / 'cache')
        cap_msg = f"capped at {max_examples:,}" if max_examples > 0 else "full corpus"
        logger.info(f"Loading Kaskus from local: {local_file} ({cap_msg}, cache → {cache_dir})")
        try:
            from datasets import Dataset as HFDataset

            def _gen():
                count = 0
                with open(local_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if max_examples > 0 and count >= max_examples:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = _json.loads(line)
                            text = obj.get('text', '')
                            if text:
                                yield {'text': text}
                                count += 1
                        except _json.JSONDecodeError:
                            continue

            return HFDataset.from_generator(_gen, cache_dir=cache_dir)
        except Exception as e:
            logger.warning(f"Failed to load Kaskus: {e}")
            return None
    
    def download_mc4_id(self):
        """Load mC4 Indonesian subset from local download, falling back to HuggingFace."""
        local_path = self.data_dir / 'mc4-id' / 'multilingual'
        if local_path.exists():
            files = sorted(str(f) for f in local_path.glob('c4-id*.json.gz'))
            if files:
                logger.info(f"Loading mC4-id from local: {local_path}")
                try:
                    return load_dataset('json', data_files=files, split='train')
                except Exception as e:
                    logger.warning(f"Local load failed, falling back to HF: {e}")

        logger.info("Downloading mC4 Indonesian from HuggingFace")
        try:
            return load_dataset('allenai/c4', 'multilingual', split='train', streaming=True)
        except Exception as e:
            logger.warning(f"Failed to load mC4-id: {e}")
            return None

    def download_culturax_id(self):
        """Load CulturaX Indonesian subset from local download, falling back to HuggingFace."""
        local_path = self.data_dir / 'culturax-id' / 'id'
        if local_path.exists():
            files = sorted(str(f) for f in local_path.glob('*.parquet'))
            if files:
                logger.info(f"Loading CulturaX-id from local: {local_path}")
                try:
                    return load_dataset('parquet', data_files=files, split='train')
                except Exception as e:
                    logger.warning(f"Local load failed, falling back to HF: {e}")

        logger.info("Downloading CulturaX Indonesian from HuggingFace (requires HF token)")
        try:
            return load_dataset('uonlp/CulturaX', 'id', split='train', streaming=True)
        except Exception as e:
            logger.warning(f"Failed to load CulturaX-id: {e}")
            return None

    def download_liputan6(self):
        """Download Liputan6 news corpus (parquet-native mirror; fajri91/liputan6 is gone).

        clean_article is list[list[str]] (sentences of tokens). We join tokens
        with spaces and sentences with newlines to produce a single article string.
        """
        logger.info("Downloading Liputan6: PetaniHandal/liputan6-canonical")
        try:
            ds = load_dataset('PetaniHandal/liputan6-canonical', split='train')

            def _flatten_article(example):
                ca = example.get('clean_article', '')
                if isinstance(ca, str):
                    text = ca
                elif isinstance(ca, list):
                    if ca and isinstance(ca[0], list):
                        text = '\n'.join(' '.join(tokens) for tokens in ca)
                    else:
                        text = '\n'.join(str(s) for s in ca)
                else:
                    text = str(ca)
                return {'text': text}

            ds = ds.map(_flatten_article, remove_columns=ds.column_names,
                        desc="Flattening Liputan6 articles")
            return ds
        except Exception as e:
            logger.warning(f"Failed to load Liputan6: {e}")
            return None
    
    def clean_text(self, text: str, min_length: int = 100, max_length: int = 10000) -> Optional[str]:
        """Clean and filter text"""
        if not text or not isinstance(text, str):
            return None
        
        # Length filter
        if len(text) < min_length or len(text) > max_length:
            return None
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove excessive repetition (e.g., "hahaha", "aaaa")
        text = re.sub(r'(.)\1{4,}', r'\1\1\1', text)
        
        # Basic Indonesian filter - check if text contains Indonesian words
        indonesian_markers = ['yang', 'dan', 'di', 'dari', 'ini', 'untuk', 'dengan', 'pada', 'adalah', 'sebagai']
        if not any(marker in text.lower() for marker in indonesian_markers):
            return None
        
        # Quality filter: check ratio of alphabetic characters
        alpha_ratio = sum(c.isalpha() or c.isspace() for c in text) / len(text) if text else 0
        if alpha_ratio < 0.7:
            return None
        
        return text.strip()
    
    @staticmethod
    def _shingles(text: str, k: int = 5) -> List[bytes]:
        words = text.split()
        if len(words) < k:
            return [text.encode('utf-8')] if text else []
        return [' '.join(words[i:i + k]).encode('utf-8') for i in range(len(words) - k + 1)]

    def deduplicate(self, dataset: Dataset, threshold: float = 0.85,
                    num_perm: int = 128, num_proc: Optional[int] = None,
                    num_bands: int = 16) -> Dataset:
        """Near-duplicate removal via parallel MinHash band-bucketing + union-find.

        All heavy phases use multiple cores:
          A. MinHash signatures per doc            (Dataset.map, num_proc workers)
          B. Split signature into num_bands hashes (Dataset.map, num_proc workers)
          C. group_by band-hash to find buckets    (PyArrow C++ thread pool)
          D. Union-find over duplicate buckets     (single-threaded; small)

        With num_perm=128 and num_bands=16 (rows_per_band=8), recall at Jaccard
        similarity 0.85 is ~99% and false-positive rate is small.
        """
        n_total = len(dataset)
        if n_total == 0:
            logger.warning("Empty dataset passed to deduplicate(); nothing to do.")
            return dataset
        rows_per_band = num_perm // num_bands
        if num_perm % num_bands != 0:
            raise ValueError(f"num_perm ({num_perm}) must be divisible by num_bands ({num_bands})")

        logger.info(f"Deduplicating {n_total:,} docs "
                    f"(threshold={threshold}, num_perm={num_perm}, "
                    f"bands={num_bands}×{rows_per_band}, num_proc={num_proc})")

        # Phase A: MinHash signatures (parallel)
        def _sig_batch(examples):
            sigs = []
            for text in examples['text']:
                m = MinHash(num_perm=num_perm)
                for sh in IndonesianDataProcessor._shingles(text or '', k=5):
                    m.update(sh)
                sigs.append(m.hashvalues.tolist())
            return {'_minhash': sigs}

        sig_ds = dataset.map(
            _sig_batch, batched=True, batch_size=1000,
            num_proc=num_proc, desc="MinHash signatures",
        )

        # Phase B: split each signature into num_bands hash keys (parallel)
        # Each input doc emits num_bands rows of (doc_id, band_key).
        def _band_batch(examples, indices):
            doc_ids: List[int] = []
            band_keys: List[bytes] = []
            for doc_id, sig in zip(indices, examples['_minhash']):
                arr = np.asarray(sig, dtype=np.uint64)
                for b in range(num_bands):
                    # 2-byte band index prefix prevents collisions across bands
                    key = b.to_bytes(2, 'little') + arr[b * rows_per_band:(b + 1) * rows_per_band].tobytes()
                    doc_ids.append(doc_id)
                    band_keys.append(key)
            return {'doc_id': doc_ids, 'band_key': band_keys}

        bands_ds = sig_ds.map(
            _band_batch, batched=True, with_indices=True, batch_size=1000,
            num_proc=num_proc, remove_columns=sig_ds.column_names,
            desc="LSH band hashes",
        )

        # Phase C: parallel hash-partitioned bucketing via multiprocessing.
        # Each worker handles a disjoint slice of the band_key space so their
        # local dicts can be concatenated without a merge step. Workers also
        # drop singletons so the main process only sees actual duplicate clusters.
        import multiprocessing as mp
        import threading
        import time

        n_workers = num_proc or (os.cpu_count() or 1)
        n_band_rows = len(bands_ds)
        # Each worker scans the FULL bands_ds (and filters by hash). Total rows
        # of work across the whole pool is therefore n_band_rows * n_workers.
        total_work = n_band_rows * n_workers
        logger.info(f"Parallel bucketing of {n_band_rows:,} band entries "
                    f"across {n_workers} workers (hash-partitioned, "
                    f"~{total_work:,} total row scans)...")

        ctx = mp.get_context('fork')
        progress_counter = ctx.Value('q', 0)

        # Background poller updates a single tqdm bar from the shared counter.
        pbar = tqdm(total=total_work, desc="Bucketing rows scanned",
                    unit='row', unit_scale=True)
        stop_event = threading.Event()

        def _poll_progress():
            last = 0
            while not stop_event.is_set():
                cur = progress_counter.value
                if cur > last:
                    pbar.update(cur - last)
                    last = cur
                time.sleep(0.5)

        poll_thread = threading.Thread(target=_poll_progress, daemon=True)
        poll_thread.start()

        worker_args = [(bands_ds, i, n_workers) for i in range(n_workers)]
        all_dup_buckets: List[List[int]] = []
        with ctx.Pool(n_workers,
                      initializer=_bucket_init_worker,
                      initargs=(progress_counter,)) as pool:
            for partition_buckets in pool.imap_unordered(
                    _bucket_partition_worker, worker_args):
                all_dup_buckets.extend(partition_buckets)

        # Stop poller and flush final delta into the bar.
        stop_event.set()
        poll_thread.join(timeout=2)
        final = progress_counter.value
        if final > pbar.n:
            pbar.update(final - pbar.n)
        pbar.close()

        n_dup_buckets = len(all_dup_buckets)
        logger.info(f"Union-find over {n_dup_buckets:,} duplicate buckets...")

        parent = list(range(n_total))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # path compression
                x = parent[x]
            return x

        for docs in all_dup_buckets:
            root = find(docs[0])
            for d in docs[1:]:
                r = find(d)
                if r != root:
                    if r < root:
                        parent[root] = r
                        root = r
                    else:
                        parent[r] = root
        del all_dup_buckets

        # Keep one doc per connected component: the canonical root index.
        keep_indices = [i for i in range(n_total) if find(i) == i]
        logger.info(f"Kept {len(keep_indices):,} / {n_total:,} documents after dedup "
                    f"({len(keep_indices) / n_total * 100:.1f}%)")
        return dataset.select(keep_indices)
    
    def tokenize_and_format(self, dataset: Dataset, max_length: int = 4096, lang: str = 'id',
                            num_proc: Optional[int] = None) -> Dataset:
        """Tokenize and format dataset for training"""
        logger.info(f"Tokenizing dataset with max_length={max_length} (num_proc={num_proc})")

        lang_token = self.lang_tokens.get(lang, '<|id|>')

        def format_and_tokenize(examples):
            texts = examples.get('text', examples.get('content', []))
            formatted = [f"{lang_token}\n{text}" for text in texts]
            tokenized = self.tokenizer(
                formatted,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_special_tokens_mask=True,
            )
            return tokenized

        return dataset.map(format_and_tokenize, batched=True,
                           remove_columns=dataset.column_names,
                           num_proc=num_proc, desc="Tokenizing")
    
    def process_dataset(self, name: str, dataset: Dataset, min_length: int = 100,
                       max_length: int = 10000, dedup_threshold: float = 0.85,
                       ner_filter: Optional[NERQualityFilter] = None,
                       ner_threshold: float = 0.1,
                       num_proc: Optional[int] = None) -> Optional[Dataset]:
        """
        Process a single dataset through all phases:
        
        1. Extract text field
        2. Clean text (regex, length, language markers)
        3. Apply NER quality filter (if enabled)
        4. Deduplicate (MinHash)
        """
        logger.info(f"\n{'='*50}")
        logger.info(f"PROCESSING: {name.upper()}")
        logger.info(f"{'='*50}")
        
        if dataset is None:
            logger.warning(f"Dataset {name} is None, skipping")
            return None
        
        # PHASE 2A: Extract text field
        logger.info("Phase 2A: Extracting text field...")
        if 'text' not in dataset.column_names:
            for col in ['content', 'article', 'body', 'document']:
                if col in dataset.column_names:
                    dataset = dataset.rename_column(col, 'text')
                    logger.info(f"  Renamed '{col}' -> 'text'")
                    break
        
        # PHASE 2B: Clean text
        logger.info(f"Phase 2B: Cleaning text (regex, length, language filters; num_proc={num_proc})...")
        def clean_batch(examples):
            cleaned = [self.clean_text(text, min_length, max_length) for text in examples['text']]
            return {'text': [c for c in cleaned if c is not None]}

        dataset = dataset.map(clean_batch, batched=True, remove_columns=dataset.column_names,
                              num_proc=num_proc, desc="Cleaning")
        dataset = dataset.filter(lambda x: x['text'] is not None and len(x['text']) > 0,
                                 num_proc=num_proc, desc="Filtering empties")
        logger.info(f"  After cleaning: {len(dataset)} documents")

        if len(dataset) == 0:
            logger.warning(f"[{name}] Cleaning filtered out all documents. "
                           f"Likely the dataset's 'text' field has an unexpected structure "
                           f"(list-of-tokens, list-of-sentences, etc). Returning empty dataset.")
            return dataset

        # PHASE 3: NER Quality Filter (optional)
        if ner_filter and ner_filter.enabled:
            dataset = ner_filter.filter_dataset(dataset, threshold=ner_threshold)
        else:
            logger.info("Phase 3: NER quality filter skipped (not enabled)")

        # PHASE 4A: Deduplicate
        if dedup_threshold < 1.0:
            logger.info("Phase 4A: Deduplicating (MinHash LSH)...")
            dataset = self.deduplicate(dataset, threshold=dedup_threshold, num_proc=num_proc)
        
        logger.info(f"\n✓ {name}: {len(dataset)} documents after full pipeline")
        logger.info(f"{'='*50}\n")
        return dataset
    
    def create_mixed_dataset(self, datasets: Dict[str, Dataset], output_path: str,
                            max_length: int = 4096, num_proc: Optional[int] = None) -> Dataset:
        """Create mixed dataset with language tags"""
        logger.info("Creating mixed dataset...")
        
        all_datasets = []
        
        for name, ds in datasets.items():
            if ds is None:
                continue
            
            # Detect language from name
            lang = 'id'  # Default to Indonesian
            for l in self.lang_tokens.keys():
                if l in name.lower():
                    lang = l
                    break
            
            # Tokenize
            tokenized = self.tokenize_and_format(ds, max_length=max_length, lang=lang, num_proc=num_proc)
            all_datasets.append(tokenized)
            
            logger.info(f"Added {name} ({lang}): {len(tokenized)} examples")
        
        # Concatenate
        if len(all_datasets) > 1:
            mixed = concatenate_datasets(all_datasets)
        elif len(all_datasets) == 1:
            mixed = all_datasets[0]
        else:
            raise ValueError("No datasets to mix")
        
        # Shuffle
        mixed = mixed.shuffle(seed=42)
        
        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        mixed.save_to_disk(str(output_path))
        logger.info(f"Saved mixed dataset to {output_path}: {len(mixed)} examples")
        
        return mixed


def main():
    parser = argparse.ArgumentParser(
        description='Nemotron-Indonesia Data Pipeline (4 phases)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline Phases:
  1. DOWNLOAD — Fetch from HuggingFace/direct sources (Indo4B, CC100, Wikipedia, etc.)
  2. CLEAN    — Regex cleaning, length filter, language detection
  3. QUALITY  — Optional NER entity-density scoring (BERT-based filter)
  4. PACKAGE  — Deduplication, tokenization, save to disk

Examples:
  # Full pipeline with NER quality filter (recommended)
  python prepare_data.py --datasets wikipedia liputan6 --use_ner_filter

  # Quick mode, no NER (faster, for initial exploration)
  python prepare_data.py --datasets wikipedia --min_length 200

  # All datasets, strict quality
  python prepare_data.py --datasets all --use_ner_filter --quality_threshold 0.15
        """
    )
    parser.add_argument('--output_dir', type=str, default='./data/processed',
                       help='Where processed data is saved (local server storage)')
    parser.add_argument('--data_dir', type=str, default='./data/raw',
                       help='Root directory of locally downloaded raw data (from download_sources.py)')
    parser.add_argument('--datasets', nargs='+', default=['cc100', 'wikipedia', 'liputan6', 'sealion'],
                       choices=['indo4b_hf', 'cc100', 'wikipedia', 'kaskus', 'liputan6',
                                'seapile', 'sealion', 'mc4_id', 'culturax_id', 'all'],
                       help='Which datasets to process')
    parser.add_argument('--num_proc', type=int, default=os.cpu_count() or 1,
                       help='Worker processes for cleaning, dedup signature computation, and tokenization (default: all CPU cores)')
    parser.add_argument('--cc100_max_examples', type=int, default=0,
                       help='Max raw lines to read from CC100 (0 = full 360M-line corpus; Arrow cache goes to data/raw/cache/)')
    parser.add_argument('--kaskus_max_examples', type=int, default=0,
                       help='Max lines to read from kaskus.jsonl (0 = full 207M-line corpus; Arrow cache goes to data/raw/cache/)')
    parser.add_argument('--min_length', type=int, default=100,
                       help='Minimum document length (characters)')
    parser.add_argument('--max_length', type=int, default=10000,
                       help='Maximum document length (characters)')
    parser.add_argument('--max_tokens', type=int, default=20_000_000_000,
                       help='Maximum tokens to process (20B default)')
    parser.add_argument('--dedup_threshold', type=float, default=0.85,
                       help='MinHash similarity threshold (0.85 = 85%% similar = duplicate)')
    parser.add_argument('--skip_dedup_for', nargs='*', default=[],
                       help='Dataset names to skip MinHash dedup for. Default: cc100 '
                            '(sentence-level corpus that produces skewed buckets and stalls). '
                            'Pass empty list to dedup everything.')
    parser.add_argument('--no_resume', action='store_true',
                       help='Force reprocess all datasets even if {output_dir}/per_dataset/{name} '
                            'checkpoints exist. Default behaviour is to resume from checkpoints.')
    parser.add_argument('--tokenizer', '--tokenizer_name', dest='tokenizer', type=str,
                       default='nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16',
                       help='Tokenizer to use for final tokenization')
    
    # NER Quality Filter arguments
    parser.add_argument('--use_ner_filter', action='store_true',
                       help='Enable NER-based quality scoring (slower but better quality)')
    parser.add_argument('--ner_model', type=str,
                       default='cahya/bert-base-indonesian-NER',
                       help='Indonesian NER model for quality scoring')
    parser.add_argument('--quality_threshold', type=float, default=0.1,
                       help='NER quality threshold (0.1 = keep docs with >=10%% entity density)')
    
    args = parser.parse_args()
    
    # ========================================================================
    # PHASE 0: Setup
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("NEMOTRON-INDONESIA DATA PIPELINE")
    logger.info("="*70)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Datasets: {args.datasets}")
    logger.info(f"Worker processes (num_proc): {args.num_proc}")
    logger.info(f"NER filter: {'ENABLED' if args.use_ner_filter else 'DISABLED'}")
    if args.use_ner_filter:
        logger.info(f"  NER model: {args.ner_model}")
        logger.info(f"  Quality threshold: {args.quality_threshold}")
    logger.info("="*70 + "\n")
    
    processor = IndonesianDataProcessor(tokenizer_name=args.tokenizer, data_dir=args.data_dir)
    
    # ========================================================================
    # PHASES 1-4: per-dataset DOWNLOAD → CLEAN → QUALITY → DEDUP → CHECKPOINT
    # ========================================================================
    # Checkpoint layout: each dataset's cleaned+deduped output is saved to
    #   {output_dir}/per_dataset/{name}/
    # On rerun, an existing directory is loaded as-is and that dataset is skipped.
    # Delete a per-dataset directory (or pass --no_resume) to force reprocess.
    logger.info("PHASES 1-4: DOWNLOAD → CLEAN → QUALITY → DEDUP (per dataset, resumable)")
    logger.info("=" * 70)

    datasets_to_process = []
    if 'all' in args.datasets:
        datasets_to_process = ['indo4b_hf', 'cc100', 'wikipedia', 'kaskus', 'liputan6',
                               'seapile', 'mc4_id', 'culturax_id']
    else:
        datasets_to_process = args.datasets

    skip_dedup_set = set(args.skip_dedup_for or [])
    per_dataset_root = Path(args.output_dir) / 'per_dataset'
    per_dataset_root.mkdir(parents=True, exist_ok=True)

    # Lazy-init NER filter only if at least one un-cached dataset needs it.
    ner_filter = None
    ner_filter_init_attempted = False

    def _download(name: str):
        if name == 'indo4b_hf':
            return processor.download_indo4b_hf()
        if name == 'cc100':
            return processor.download_cc100(max_examples=args.cc100_max_examples)
        if name == 'wikipedia':
            return processor.download_wikipedia()
        if name == 'kaskus':
            return processor.download_kaskus(max_examples=args.kaskus_max_examples)
        if name == 'liputan6':
            return processor.download_liputan6()
        if name in ('seapile', 'sealion'):
            return processor.download_sealion_pile()
        if name == 'mc4_id':
            return processor.download_mc4_id()
        if name == 'culturax_id':
            return processor.download_culturax_id()
        raise ValueError(f"Unknown dataset: {name}")

    processed = {}
    for name in datasets_to_process:
        ckpt_path = per_dataset_root / name
        # Resume: if a prior run produced this dataset, load it and move on.
        if not args.no_resume and ckpt_path.exists():
            try:
                logger.info(f"\n[{name}] RESUMING from checkpoint: {ckpt_path}")
                processed[name] = Dataset.load_from_disk(str(ckpt_path))
                logger.info(f"[{name}] Loaded {len(processed[name]):,} processed docs.")
                continue
            except Exception as e:
                logger.warning(f"[{name}] Failed to load checkpoint ({e}); reprocessing.")

        # Download
        logger.info(f"\n[{name}] DOWNLOAD")
        raw = _download(name)
        if raw is None:
            logger.warning(f"[{name}] download returned None — skipping.")
            processed[name] = None
            continue

        # NER filter loaded once, on first dataset that needs it.
        if args.use_ner_filter and not ner_filter_init_attempted:
            logger.info("Loading NER quality filter (first use)...")
            ner_filter = NERQualityFilter(model_name=args.ner_model)
            ner_filter_init_attempted = True
            if not ner_filter.enabled:
                logger.warning("NER filter failed to load — proceeding without it")

        # Clean + (optional NER) + dedup
        per_dataset_threshold = 1.0 if name in skip_dedup_set else args.dedup_threshold
        if name in skip_dedup_set:
            logger.info(f"[{name}] Dedup SKIPPED (in --skip_dedup_for list)")
        result = processor.process_dataset(
            name, raw,
            min_length=args.min_length,
            max_length=args.max_length,
            dedup_threshold=per_dataset_threshold,
            ner_filter=ner_filter,
            ner_threshold=args.quality_threshold,
            num_proc=args.num_proc,
        )
        processed[name] = result

        # Checkpoint: save the cleaned+deduped per-dataset output for resume.
        if result is not None:
            try:
                # Write to a temp dir then atomically rename so a crashed run
                # never leaves a half-written checkpoint that we'd then load.
                tmp_path = ckpt_path.with_name(ckpt_path.name + '.tmp')
                if tmp_path.exists():
                    import shutil
                    shutil.rmtree(tmp_path)
                result.save_to_disk(str(tmp_path))
                if ckpt_path.exists():
                    import shutil
                    shutil.rmtree(ckpt_path)
                tmp_path.rename(ckpt_path)
                logger.info(f"[{name}] Checkpoint saved: {ckpt_path} ({len(result):,} docs)")
            except Exception as e:
                logger.warning(f"[{name}] Failed to save checkpoint at {ckpt_path}: {e}")
    
    # ========================================================================
    # FINAL: Create mixed dataset and save
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("FINAL: CREATING MIXED DATASET")
    logger.info("="*70)
    
    output_path = Path(args.output_dir) / 'indonesian_corpus'
    mixed = processor.create_mixed_dataset(processed, str(output_path), num_proc=args.num_proc)
    
    # Save tokenizer with added tokens
    tokenizer_path = Path(args.output_dir) / 'tokenizer'
    processor.tokenizer.save_pretrained(str(tokenizer_path))
    
    # ========================================================================
    # Summary
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*70)
    logger.info(f"Total examples: {len(mixed):,}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Tokenizer saved to: {tokenizer_path}")
    
    # Estimate token count
    avg_length = 512  # rough estimate
    est_tokens = len(mixed) * avg_length
    logger.info(f"Estimated tokens: ~{est_tokens:,} (avg {avg_length} tokens/doc)")
    
    if est_tokens < args.max_tokens:
        logger.info(f"Target: {args.max_tokens:,} tokens — consider adding more datasets")
    else:
        logger.info(f"Target met: {est_tokens:,} >= {args.max_tokens:,}")
    
    logger.info("="*70)
    logger.info("\nNext step: Run training with")
    logger.info(f"  ./run_training.sh pretrain --data_path {output_path}")
    logger.info("="*70 + "\n")


if __name__ == '__main__':
    main()
