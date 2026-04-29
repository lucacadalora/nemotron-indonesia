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

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from transformers import AutoModelForCausalLM, AutoTokenizer
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
            device_map='auto',
            trust_remote_code=True,
        )
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_answer(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.1) -> str:
        """Generate answer for a prompt"""
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
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


class BenchmarkSuite:
    """Run multiple benchmarks"""
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.model_path = model_path
        self.device = device
        self.evaluators = {
            'indommlu': IndoMMLUEvaluator,
            'nusax': NusaXEvaluator,
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
                       choices=['indommlu', 'nusax', 'all'])
    parser.add_argument('--output', type=str, default='results.json')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_samples', type=int, default=None)
    
    args = parser.parse_args()
    
    # Determine benchmarks
    if args.benchmark == 'all':
        benchmarks = ['indommlu', 'nusax']
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
