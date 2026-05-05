#!/usr/bin/env python3
"""
Evaluation script for Nemotron-Indonesia
Runs IndoMMLU and other Indonesian benchmarks

Usage:
    python evaluate.py \
        --model_path ./models/nemotron-indonesia-omni-30b \
        --benchmark indommlu \
        --output results.json \
        --batch_size 8
"""

import csv
import os
import json
import argparse
import logging
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IndoBenchmark:
    """Base class for Indonesian benchmarks"""
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.model_path = model_path
        self.device = device
        
        logger.info(f"Loading model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map={'': 0},
            trust_remote_code=True,
        )
        self.model.eval()

        # The model's generation_config.json ships with do_sample=True and temperature=0.6.
        # The custom generate wrapper in modeling.py reads this config internally, so passing
        # do_sample=False as a kwarg to generate() is not enough — we must patch the config.
        self.model.generation_config = GenerationConfig(
            do_sample=False,
            bos_token_id=self.model.generation_config.bos_token_id,
            eos_token_id=self.model.generation_config.eos_token_id,
            pad_token_id=self.model.generation_config.pad_token_id,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_answer(self, prompt: str, max_new_tokens: int = 100) -> str:
        """Generate answer using greedy decoding (deterministic, no NaN risk from sampling)."""
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                generation_config=self.model.generation_config,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        answer = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return answer.strip()


class IndoMMLUEvaluator(IndoBenchmark):
    """Evaluate on IndoMMLU benchmark"""
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        super().__init__(model_path, device)
        
        logger.info("Loading IndoMMLU dataset")
        try:
            self.dataset = load_dataset('indommlu', split='test')
            logger.info(f"Loaded {len(self.dataset)} test examples")
        except Exception as e:
            logger.error(f"Failed to load IndoMMLU: {e}")
            self.dataset = None
    
    def format_prompt(self, question: str, choices: List[str]) -> str:
        """Format question for model"""
        prompt = f"<|user|>\n{question}\n\nPilihan:\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65+i)}. {choice}\n"
        prompt += "\n<|assistant|>\nJawaban:"
        return prompt
    
    def parse_answer(self, generated: str) -> Optional[str]:
        """Parse generated answer to extract choice letter"""
        generated = generated.strip().upper()
        
        # Look for A, B, C, D at start
        if generated and generated[0] in 'ABCD':
            return generated[0]
        
        # Look for "A.", "B.", etc.
        for letter in 'ABCD':
            if f"{letter}." in generated or f"{letter})" in generated:
                return letter
        
        # Look for "jawabannya adalah A" pattern
        for letter in 'ABCD':
            if letter in generated[:20]:  # Check first 20 chars
                return letter
        
        return None
    
    def evaluate(self, max_samples: Optional[int] = None) -> Dict:
        """Run evaluation"""
        if self.dataset is None:
            return {'error': 'Dataset not loaded'}
        
        results = {
            'total': 0,
            'correct': 0,
            'by_subject': defaultdict(lambda: {'total': 0, 'correct': 0}),
            'by_difficulty': defaultdict(lambda: {'total': 0, 'correct': 0}),
            'predictions': [],
        }
        
        dataset = self.dataset
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        for example in tqdm(dataset, desc="Evaluating IndoMMLU"):
            question = example['question']
            choices = example['choices']
            answer = example['answer']
            subject = example.get('subject', 'unknown')
            difficulty = example.get('difficulty', 'unknown')
            
            # Format prompt
            prompt = self.format_prompt(question, choices)
            
            # Generate
            generated = self.generate_answer(prompt, max_new_tokens=10)
            predicted = self.parse_answer(generated)
            
            # Check correctness
            is_correct = predicted == answer
            
            # Update stats
            results['total'] += 1
            if is_correct:
                results['correct'] += 1
            
            results['by_subject'][subject]['total'] += 1
            if is_correct:
                results['by_subject'][subject]['correct'] += 1
            
            results['by_difficulty'][difficulty]['total'] += 1
            if is_correct:
                results['by_difficulty'][difficulty]['correct'] += 1
            
            results['predictions'].append({
                'question': question,
                'choices': choices,
                'answer': answer,
                'predicted': predicted,
                'generated': generated,
                'correct': is_correct,
                'subject': subject,
            })
        
        # Calculate accuracies
        results['accuracy'] = results['correct'] / results['total'] if results['total'] > 0 else 0
        
        for subject, stats in results['by_subject'].items():
            stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        
        for diff, stats in results['by_difficulty'].items():
            stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        
        return results


class NusaXEvaluator(IndoBenchmark):
    """Evaluate on NusaX multilingual sentiment"""
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        super().__init__(model_path, device)
        
        logger.info("Loading NusaX dataset")
        try:
            self.datasets = {}
            for lang in ['ind', 'jav', 'sun', 'min', 'bug']:
                self.datasets[lang] = load_dataset('indonlp/nusatranslation_senti', lang, split='test')
        except Exception as e:
            logger.error(f"Failed to load NusaX: {e}")
    
    def evaluate(self) -> Dict:
        """Run evaluation across languages"""
        results = {}
        
        for lang, dataset in self.datasets.items():
            correct = 0
            total = 0
            
            for example in tqdm(dataset, desc=f"NusaX-{lang}"):
                text = example['text']
                label = example['label']
                
                prompt = f"<|user|>\nTeks: {text}\nSentimen (positif/negatif/netral): <|assistant|>\n"
                generated = self.generate_answer(prompt, max_new_tokens=10)
                
                # Map generated to label
                pred_label = self.map_sentiment(generated)
                
                if pred_label == label:
                    correct += 1
                total += 1
            
            results[lang] = {
                'accuracy': correct / total if total > 0 else 0,
                'correct': correct,
                'total': total,
            }
        
        # Average across languages
        accuracies = [r['accuracy'] for r in results.values()]
        results['average'] = np.mean(accuracies) if accuracies else 0
        
        return results
    
    def map_sentiment(self, generated: str) -> int:
        """Map generated text to sentiment label"""
        generated = generated.lower().strip()
        
        if 'positif' in generated or 'positive' in generated:
            return 1
        elif 'negatif' in generated or 'negative' in generated:
            return 0
        elif 'netral' in generated or 'neutral' in generated:
            return 2
        
        # Default based on first word
        if generated.startswith('p'):
            return 1
        elif generated.startswith('n'):
            return 0
        
        return 2  # Default neutral


class IndoNLUEvaluator(IndoBenchmark):
    """Evaluate on IndoNLU benchmark tasks (SmSA sentiment + EmoT emotion).

    Fetches raw test files directly from the IndoNLU GitHub repository,
    bypassing the HF datasets loading-script mechanism removed in datasets>=3.0.
    """

    # Source: indonlu.py BUILDER_CONFIGS — exact URLs, formats, and column order.
    TASK_CONFIGS = {
        'smsa': {
            'name': 'SmSA (Sentiment)',
            'test_url': (
                'https://raw.githubusercontent.com/IndoNLP/indonlu/master/'
                'dataset/smsa_doc-sentiment-prosa/test_preprocess.tsv'
            ),
            'delimiter': '\t',
            'quoting': csv.QUOTE_NONE,
            'has_header': False,
            'columns': ['text', 'label'],      # TSV column order: text, label
            'text_col': 'text',
            'label_col': 'label',
            'label_classes': ['positive', 'neutral', 'negative'],
            'prompt_template': (
                "<|user|>\nAnalisis sentimen dari kalimat berikut.\n"
                "Kalimat: {text}\n"
                "Pilih satu: positive / negative / neutral\n"
                "<|assistant|>\nSentimen:"
            ),
        },
        'emot': {
            'name': 'EmoT (Emotion)',
            'test_url': (
                'https://raw.githubusercontent.com/IndoNLP/indonlu/master/'
                'dataset/emot_emotion-twitter/test_preprocess.csv'
            ),
            'delimiter': ',',
            'quoting': csv.QUOTE_ALL,
            'has_header': True,
            'columns': ['label', 'tweet'],     # CSV column order: label, tweet
            'text_col': 'tweet',
            'label_col': 'label',
            'label_classes': ['sadness', 'anger', 'love', 'fear', 'happy'],
            'prompt_template': (
                "<|user|>\nIdentifikasi emosi dominan dari tweet berikut.\n"
                "Tweet: {text}\n"
                "<|assistant|>\nEmosi:"
            ),
        },
    }

    def __init__(self, model_path: str, device: str = 'cuda'):
        super().__init__(model_path, device)
        self.loaded_tasks: Dict[str, List[Dict]] = {}
        for task_key, cfg in self.TASK_CONFIGS.items():
            try:
                rows = self._fetch_task(cfg)
                self.loaded_tasks[task_key] = rows
                logger.info(f"Loaded IndoNLU {cfg['name']}: {len(rows)} examples")
            except Exception as e:
                logger.warning(f"Could not load IndoNLU {cfg['name']}: {e}")

    @staticmethod
    def _fetch_task(cfg: dict) -> List[Dict]:
        """Download raw CSV/TSV from GitHub and return a list of row dicts."""
        with urllib.request.urlopen(cfg['test_url']) as resp:
            content = resp.read().decode('utf-8')
        reader = csv.reader(
            content.splitlines(),
            delimiter=cfg['delimiter'],
            quotechar='"',
            quoting=cfg['quoting'],
        )
        if cfg['has_header']:
            next(reader)
        return [
            {col: row[i] for i, col in enumerate(cfg['columns'])}
            for row in reader if row
        ]

    def _parse_label(self, generated: str, label_classes: List[str]) -> Optional[int]:
        text = generated.lower().strip()
        for i, name in enumerate(label_classes):
            if name.lower() in text[:40]:
                return i
        return None

    def _evaluate_task(self, task_key: str, max_samples: Optional[int]) -> Dict:
        cfg = self.TASK_CONFIGS[task_key]
        rows = self.loaded_tasks[task_key]
        if max_samples:
            rows = rows[:max_samples]

        label_classes = cfg['label_classes']
        correct = 0
        total = 0
        predictions = []

        for example in tqdm(rows, desc=f"IndoNLU/{cfg['name']}"):
            text = example[cfg['text_col']]
            true_label = example[cfg['label_col']]
            try:
                true_idx = label_classes.index(true_label)
            except ValueError:
                logger.warning(f"Skipping unknown label '{true_label}'")
                continue

            prompt = cfg['prompt_template'].format(text=text)
            generated = self.generate_answer(prompt, max_new_tokens=10)
            pred_idx = self._parse_label(generated, label_classes)

            is_correct = pred_idx == true_idx
            correct += int(is_correct)
            total += 1
            predictions.append({
                'text': text,
                'true': true_label,
                'predicted': label_classes[pred_idx] if pred_idx is not None else None,
                'generated': generated,
                'correct': is_correct,
            })

        return {
            'accuracy': correct / total if total > 0 else 0,
            'correct': correct,
            'total': total,
            'predictions': predictions,
        }

    def evaluate(self, max_samples: Optional[int] = None) -> Dict:
        if not self.loaded_tasks:
            return {'error': 'No IndoNLU tasks loaded', 'average': 0.0}

        task_results = {}
        for task_key in self.loaded_tasks:
            task_results[task_key] = self._evaluate_task(task_key, max_samples)

        accuracies = [r['accuracy'] for r in task_results.values()]
        return {
            'tasks': task_results,
            'average': sum(accuracies) / len(accuracies) if accuracies else 0.0,
        }


class BenchmarkSuite:
    """Run multiple benchmarks"""
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.model_path = model_path
        self.device = device
        self.evaluators = {
            'indommlu': IndoMMLUEvaluator,
            'nusax': NusaXEvaluator,
            'indonlu': IndoNLUEvaluator,
        }
    
    def run(self, benchmarks: List[str], output_path: Optional[str] = None) -> Dict:
        """Run specified benchmarks"""
        all_results = {
            'model': self.model_path,
            'benchmarks': {},
        }
        
        for benchmark_name in benchmarks:
            if benchmark_name not in self.evaluators:
                logger.warning(f"Unknown benchmark: {benchmark_name}")
                continue
            
            logger.info(f"Running {benchmark_name}...")
            evaluator = self.evaluators[benchmark_name](self.model_path, self.device)
            results = evaluator.evaluate()
            
            all_results['benchmarks'][benchmark_name] = results
            
            # Print summary
            if 'accuracy' in results:
                print(f"\n{benchmark_name.upper()} Accuracy: {results['accuracy']:.2%}")
            elif 'average' in results:
                print(f"\n{benchmark_name.upper()} Average: {results['average']:.2%}")
        
        # Save results
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results saved to {output_path}")
        
        return all_results


def main():
    parser = argparse.ArgumentParser(description='Evaluate Nemotron-Indonesia')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--benchmark', type=str, default='indommlu',
                       choices=['indommlu', 'nusax', 'indonlu', 'all'])
    parser.add_argument('--output', type=str, default='results.json')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_samples', type=int, default=None)
    
    args = parser.parse_args()
    
    # Determine benchmarks
    if args.benchmark == 'all':
        benchmarks = ['indommlu', 'nusax', 'indonlu']
    else:
        benchmarks = [args.benchmark]
    
    # Run evaluation
    suite = BenchmarkSuite(args.model_path, args.device)
    results = suite.run(benchmarks, args.output)
    
    # Print final summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    for name, result in results['benchmarks'].items():
        if 'accuracy' in result:
            print(f"{name:20s}: {result['accuracy']:.2%}")
        elif 'average' in result:
            print(f"{name:20s}: {result['average']:.2%}")
    
    print("="*60)


if __name__ == '__main__':
    main()
