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
        --datasets oscar cc100 wikipedia \
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


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
    
    def __init__(self, tokenizer_name: str = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16"):
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
    
    def download_sealion_pile(self, language: str = 'id'):
        """Download SEA-LION Pile (SEA-PILE-v1) and filter for Indonesian only
        
        SEA-PILE-v1 contains 11 SEA languages. We filter for files starting
        with 'c4-id' to get only Indonesian content.
        """
        logger.info(f"Downloading SEA-LION Pile for language: {language}")
        
        try:
            ds = load_dataset('aisingapore/SEA-PILE-v1', split='train', streaming=True)
            # Filter for Indonesian files only (c4-id prefix)
            ds = ds.filter(lambda x: x.get('file', '').startswith(f'c4-{language}'))
            logger.info(f"SEA-LION Pile filtered to {language} subset")
            return ds
        except Exception as e:
            logger.warning(f"Failed to load SEA-LION Pile: {e}")
            return None
    
    def download_cc100(self, language: str = 'id'):
        """Download Common Crawl 100"""
        logger.info(f"Downloading CC100 for language: {language}")
        
        try:
            ds = load_dataset('cc100', lang=language, split='train', streaming=True)
            return ds
        except Exception as e:
            logger.warning(f"Failed to load CC100: {e}")
            return None
    
    def download_wikipedia(self, language: str = 'id'):
        """Download Wikipedia"""
        logger.info(f"Downloading Wikipedia for language: {language}")
        
        try:
            ds = load_dataset('wikimedia/wikipedia', f'20231101.{language}', split='train')
            return ds
        except Exception as e:
            logger.warning(f"Failed to load Wikipedia: {e}")
            return None
    
    def download_kaskus(self, data_dir: Optional[str] = None):
        """Load Kaskus forum corpus if available locally"""
        if data_dir and Path(data_dir).exists():
            logger.info(f"Loading Kaskus data from {data_dir}")
            try:
                ds = load_dataset('json', data_files=str(Path(data_dir) / 'kaskus.jsonl'), split='train')
                return ds
            except Exception as e:
                logger.warning(f"Failed to load Kaskus: {e}")
        return None
    
    def download_liputan6(self):
        """Download Liputan6 news corpus"""
        logger.info("Downloading Liputan6")
        
        try:
            ds = load_dataset('fajri91/liputan6', 'canonical', split='train')
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
    
    def compute_minhash(self, text: str, num_hashes: int = 128) -> List[int]:
        """Compute MinHash for deduplication"""
        shingles = set()
        words = text.split()
        
        # Create 5-gram shingles
        for i in range(len(words) - 4):
            shingle = ' '.join(words[i:i+5])
            shingles.add(hashlib.md5(shingle.encode()).hexdigest())
        
        # Compute minhash signatures
        signatures = []
        for i in range(num_hashes):
            min_hash = float('inf')
            for shingle in shingles:
                hash_val = int(hashlib.md5(f"{shingle}_{i}".encode()).hexdigest(), 16)
                min_hash = min(min_hash, hash_val)
            signatures.append(min_hash)
        
        return signatures
    
    def deduplicate(self, dataset: Dataset, threshold: float = 0.85, batch_size: int = 10000) -> Dataset:
        """Remove near-duplicate documents using MinHash LSH"""
        logger.info(f"Deduplicating dataset with threshold {threshold}")
        
        signatures = []
        keep_indices = []
        
        for idx in tqdm(range(len(dataset)), desc="Computing signatures"):
            text = dataset[idx].get('text', '')
            sig = self.compute_minhash(text)
            signatures.append(sig)
        
        # Simple pairwise comparison (for small datasets)
        # For large datasets, use LSH
        for i in tqdm(range(len(signatures)), desc="Deduplicating"):
            is_duplicate = False
            for j in keep_indices:
                # Compute Jaccard similarity
                intersection = sum(1 for a, b in zip(signatures[i], signatures[j]) if a == b)
                similarity = intersection / len(signatures[i])
                
                if similarity > threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                keep_indices.append(i)
        
        logger.info(f"Kept {len(keep_indices)} / {len(dataset)} documents after deduplication")
        return dataset.select(keep_indices)
    
    def tokenize_and_format(self, dataset: Dataset, max_length: int = 4096, lang: str = 'id') -> Dataset:
        """Tokenize and format dataset for training"""
        logger.info(f"Tokenizing dataset with max_length={max_length}")
        
        lang_token = self.lang_tokens.get(lang, '<|id|>')
        
        def format_and_tokenize(examples):
            texts = examples.get('text', examples.get('content', []))
            
            # Add language token
            formatted = [f"{lang_token}\n{text}" for text in texts]
            
            # Tokenize
            tokenized = self.tokenizer(
                formatted,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_special_tokens_mask=True,
            )
            
            return tokenized
        
        return dataset.map(format_and_tokenize, batched=True, remove_columns=dataset.column_names)
    
    def process_dataset(self, name: str, dataset: Dataset, min_length: int = 100, 
                       max_length: int = 10000, dedup_threshold: float = 0.85,
                       ner_filter: Optional[NERQualityFilter] = None,
                       ner_threshold: float = 0.1) -> Optional[Dataset]:
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
        logger.info("Phase 2B: Cleaning text (regex, length, language filters)...")
        def clean_batch(examples):
            cleaned = [self.clean_text(text, min_length, max_length) for text in examples['text']]
            return {'text': [c for c in cleaned if c is not None]}
        
        dataset = dataset.map(clean_batch, batched=True, remove_columns=dataset.column_names)
        dataset = dataset.filter(lambda x: x['text'] is not None and len(x['text']) > 0)
        logger.info(f"  After cleaning: {len(dataset)} documents")
        
        # PHASE 3: NER Quality Filter (optional)
        if ner_filter and ner_filter.enabled:
            dataset = ner_filter.filter_dataset(dataset, threshold=ner_threshold)
        else:
            logger.info("Phase 3: NER quality filter skipped (not enabled)")
        
        # PHASE 4A: Deduplicate
        if dedup_threshold < 1.0:
            logger.info("Phase 4A: Deduplicating (MinHash LSH)...")
            dataset = self.deduplicate(dataset, threshold=dedup_threshold)
        
        logger.info(f"\n✓ {name}: {len(dataset)} documents after full pipeline")
        logger.info(f"{'='*50}\n")
        return dataset
    
    def create_mixed_dataset(self, datasets: Dict[str, Dataset], output_path: str,
                            max_length: int = 4096) -> Dataset:
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
            tokenized = self.tokenize_and_format(ds, max_length=max_length, lang=lang)
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
  1. DOWNLOAD — Fetch from HuggingFace (OSCAR, CC100, Wikipedia, etc.)
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
    parser.add_argument('--datasets', nargs='+', default=['cc100', 'wikipedia', 'liputan6', 'sealion'],
                       choices=['oscar', 'cc100', 'wikipedia', 'kaskus', 'liputan6', 'sealion', 'all'],
                       help='Which datasets to download and process')
    parser.add_argument('--min_length', type=int, default=100,
                       help='Minimum document length (characters)')
    parser.add_argument('--max_length', type=int, default=10000,
                       help='Maximum document length (characters)')
    parser.add_argument('--max_tokens', type=int, default=20_000_000_000,
                       help='Maximum tokens to process (20B default)')
    parser.add_argument('--dedup_threshold', type=float, default=0.85,
                       help='MinHash similarity threshold (0.85 = 85%% similar = duplicate)')
    parser.add_argument('--tokenizer', type=str,
                       default='nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16',
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
    logger.info(f"NER filter: {'ENABLED' if args.use_ner_filter else 'DISABLED'}")
    if args.use_ner_filter:
        logger.info(f"  NER model: {args.ner_model}")
        logger.info(f"  Quality threshold: {args.quality_threshold}")
    logger.info("="*70 + "\n")
    
    processor = IndonesianDataProcessor(tokenizer_name=args.tokenizer)
    
    # ========================================================================
    # PHASE 1: DOWNLOAD
    # ========================================================================
    logger.info("PHASE 1: DOWNLOADING DATASETS")
    logger.info("-" * 70)
    
    datasets_to_download = []
    if 'all' in args.datasets:
        datasets_to_download = ['oscar', 'cc100', 'wikipedia', 'kaskus', 'liputan6', 'sealion']
    else:
        datasets_to_download = args.datasets
    
    datasets = {}
    for name in datasets_to_download:
        logger.info(f"\nDownloading {name}...")
        if name == 'oscar':
            datasets['oscar'] = processor.download_oscar()
        elif name == 'cc100':
            datasets['cc100'] = processor.download_cc100()
        elif name == 'wikipedia':
            datasets['wikipedia'] = processor.download_wikipedia()
        elif name == 'kaskus':
            datasets['kaskus'] = processor.download_kaskus()
        elif name == 'liputan6':
            datasets['liputan6'] = processor.download_liputan6()
        elif name == 'sealion':
            datasets['sealion'] = processor.download_sealion_pile()
    
    logger.info("\n" + "-" * 70)
    logger.info("PHASE 1 COMPLETE")
    for name, ds in datasets.items():
        status = f"{len(ds)} rows" if ds else "FAILED"
        logger.info(f"  {name}: {status}")
    logger.info("-" * 70 + "\n")
    
    # ========================================================================
    # PHASE 3 (Pre-load): Initialize NER filter if enabled
    # ========================================================================
    ner_filter = None
    if args.use_ner_filter:
        logger.info("PHASE 3 (PREP): Loading NER quality filter...")
        ner_filter = NERQualityFilter(model_name=args.ner_model)
        if not ner_filter.enabled:
            logger.warning("NER filter failed to load — will proceed without it")
        logger.info("-" * 70 + "\n")
    
    # ========================================================================
    # PHASE 2-4: Process each dataset (Clean → Quality → Package)
    # ========================================================================
    logger.info("PHASES 2-4: CLEAN → QUALITY → PACKAGE")
    logger.info("=" * 70)
    
    processed = {}
    for name, ds in datasets.items():
        processed[name] = processor.process_dataset(
            name, ds,
            min_length=args.min_length,
            max_length=args.max_length,
            dedup_threshold=args.dedup_threshold,
            ner_filter=ner_filter,
            ner_threshold=args.quality_threshold
        )
    
    # ========================================================================
    # FINAL: Create mixed dataset and save
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("FINAL: CREATING MIXED DATASET")
    logger.info("="*70)
    
    output_path = Path(args.output_dir) / 'indonesian_corpus'
    mixed = processor.create_mixed_dataset(processed, str(output_path))
    
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
