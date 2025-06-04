import torch
import time
import csv
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class InferenceBenchmark:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()
        
    def _time_fn(self, fn, *args, **kwargs):
        # Warmup
        for _ in range(10):
            fn(*args, **kwargs)
            
        # Measure
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        for _ in range(100):
            fn(*args, **kwargs)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start_time) / 100
        return elapsed

    def benchmark_input_length(self, output_dir="benchmarks/latency/"):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        results = []
        
        input_lengths = [16, 32, 64, 128, 256, 512]
        for seq_len in input_lengths:
            text = "formal " * seq_len  # Generate dummy input
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            
            latency = self._time_fn(self.model, **inputs)
            results.append({
                'sequence_length': seq_len,
                'latency_ms': latency * 1000,
                'device': str(self.device)
            })
            
        # Save results
        with open(f"{output_dir}/input_length_latency.csv", 'w') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
            
        return results

    def benchmark_batch_size(self, max_batch=64, seq_len=128):
        results = []
        text = "summer dress " * seq_len
        single_input = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        for batch_size in [1, 2, 4, 8, 16, 32, 64]:
            try:
                inputs = {k: v.repeat(batch_size, 1) for k,v in single_input.items()}
                latency = self._time_fn(self.model, **inputs)
                results.append({
                    'batch_size': batch_size,
                    'throughput_qps': batch_size / latency,
                    'latency_ms': latency * 1000
                })
            except RuntimeError as e:  # Handle OOM
                break
                
        return results

class EndToEndLatency:
    def __init__(self, model_path):
        from concurrent.futures import ThreadPoolExecutor
        self.model = InferenceBenchmark(model_path)
        self.executor = ThreadPoolExecutor(max_workers=8)
        
    def stress_test(self, requests=1000):
        """Simulate concurrent user requests"""
        text = "business casual outfit with blazer but not too formal"
        futures = [self.executor.submit(self.model._time_fn, 
                                      self.model.model,
                                      **self.model.tokenizer(text, return_tensors="pt"))
                   for _ in range(requests)]
        
        latencies = [f.result() * 1000 for f in futures]
        return {
            'p50_ms': np.percentile(latencies, 50),
            'p95_ms': np.percentile(latencies, 95),
            'max_ms': max(latencies)
        }

if __name__ == "__main__":
    benchmark = InferenceBenchmark("your-finetuned-model")
    # Single-input tests
    length_results = benchmark.benchmark_input_length()
    # Batch processing
    batch_results = benchmark.benchmark_batch_size()
    # Concurrent load
    stress_test = EndToEndLatency("your-finetuned-model").stress_test(1000)