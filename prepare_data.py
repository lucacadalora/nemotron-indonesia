#!/usr/bin/env python3
"""
Data preparation script for Nemotron-Indonesia
Downloads, deduplicates, and tokenizes Indonesian datasets

Usage:
    python prepare_data.py \
        --output_dir ./data/processed \
        --datasets oscar cc100 wikipedia \
        --min_length 100 \
        --max_length 4096 \
        --dedup_threshold 0.85
"""

import os
import re
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict
import hashlib

from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    
    def download_oscar(self, language: str = 'id', split: str = 'train', streaming: bool = True):
        """Download OSCAR corpus"""
        logger.info(f"Downloading OSCAR for language: {language}")
        
        try:
            ds = load_dataset('oscar-corpus/OSCAR-2301', language=f'{language}', split=split, streaming=streaming)
            return ds
        except Exception as e:
            logger.warning(f"Failed to load OSCAR: {e}")
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
                       max_length: int = 10000, dedup_threshold: float = 0.85) -> Optional[Dataset]:
        """Process a single dataset"""
        logger.info(f"Processing {name}...")
        
        if dataset is None:
            return None
        
        # Extract text field
        if 'text' not in dataset.column_names:
            # Try to find text field
            for col in ['content', 'article', 'body', 'document']:
                if col in dataset.column_names:
                    dataset = dataset.rename_column(col, 'text')
                    break
        
        # Clean text
        def clean_batch(examples):
            cleaned = [self.clean_text(text, min_length, max_length) for text in examples['text']]
            return {'text': [c for c in cleaned if c is not None]}
        
        dataset = dataset.map(clean_batch, batched=True, remove_columns=dataset.column_names)
        dataset = dataset.filter(lambda x: x['text'] is not None and len(x['text']) > 0)
        
        # Deduplicate
        if dedup_threshold < 1.0:
            dataset = self.deduplicate(dataset, threshold=dedup_threshold)
        
        logger.info(f"{name}: {len(dataset)} documents after processing")
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
    parser = argparse.ArgumentParser(description='Prepare Indonesian training data')
    parser.add_argument('--output_dir', type=str, default='./data/processed')
    parser.add_argument('--datasets', nargs='+', default=['oscar', 'cc100', 'wikipedia'],
                       choices=['oscar', 'cc100', 'wikipedia', 'kaskus', 'liputan6', 'all'])
    parser.add_argument('--min_length', type=int, default=100)
    parser.add_argument('--max_length', type=int, default=10000)
    parser.add_argument('--max_tokens', type=int, default=20_000_000_000,
                       help='Maximum tokens to process (20B default)')
    parser.add_argument('--dedup_threshold', type=float, default=0.85)
    parser.add_argument('--tokenizer', type=str, default='nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16')
    
    args = parser.parse_args()
    
    processor = IndonesianDataProcessor(tokenizer_name=args.tokenizer)
    
    # Download datasets
    datasets = {}
    
    if 'all' in args.datasets or 'oscar' in args.datasets:
        datasets['oscar'] = processor.download_oscar()
    
    if 'all' in args.datasets or 'cc100' in args.datasets:
        datasets['cc100'] = processor.download_cc100()
    
    if 'all' in args.datasets or 'wikipedia' in args.datasets:
        datasets['wikipedia'] = processor.download_wikipedia()
    
    if 'all' in args.datasets or 'kaskus' in args.datasets:
        datasets['kaskus'] = processor.download_kaskus()
    
    if 'all' in args.datasets or 'liputan6' in args.datasets:
        datasets['liputan6'] = processor.download_liputan6()
    
    # Process datasets
    processed = {}
    for name, ds in datasets.items():
        processed[name] = processor.process_dataset(
            name, ds,
            min_length=args.min_length,
            max_length=args.max_length,
            dedup_threshold=args.dedup_threshold
        )
    
    # Create mixed dataset
    output_path = Path(args.output_dir) / 'indonesian_corpus'
    mixed = processor.create_mixed_dataset(processed, str(output_path))
    
    # Save tokenizer with added tokens
    tokenizer_path = Path(args.output_dir) / 'tokenizer'
    processor.tokenizer.save_pretrained(str(tokenizer_path))
    
    # Print statistics
    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE")
    print("="*60)
    print(f"Total examples: {len(mixed)}")
    print(f"Output path: {output_path}")
    print(f"Tokenizer saved to: {tokenizer_path}")
    print("="*60)


if __name__ == '__main__':
    main()
