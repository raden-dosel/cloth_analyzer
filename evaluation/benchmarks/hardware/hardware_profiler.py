import time
import subprocess
import psutil
import pandas as pd
from pathlib import Path

class HardwareProfiler:
    def __init__(self, output_dir="benchmarks/hardware/"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_gpu_stats(self):
        try:
            result = subprocess.check_output([
                'nvidia-smi', 
                '--query-gpu=memory.used,memory.total,utilization.gpu',
                '--format=csv,nounits,noheader'
            ]).decode().strip().split(',')
            return {
                'gpu_mem_used_mb': int(result[0]),
                'gpu_mem_total_mb': int(result[1]),
                'gpu_util_percent': int(result[2])
            }
        except:
            return None
            
    def _get_cpu_stats(self):
        return {
            'cpu_percent': psutil.cpu_percent(),
            'cpu_mem_used_gb': psutil.virtual_memory().used / 1e9,
            'cpu_mem_total_gb': psutil.virtual_memory().total / 1e9
        }

    class ProfileSession:
        def __init__(self, profiler, interval=0.1):
            self.profiler = profiler
            self.interval = interval
            self.running = False
            self.data = []
            
        def __enter__(self):
            self.running = True
            self.start_time = time.time()
            self._profile_loop()
            return self
            
        def __exit__(self, *args):
            self.running = False
            
        def _profile_loop(self):
            from threading import Thread
            self.thread = Thread(target=self._collect_stats)
            self.thread.start()
            
        def _collect_stats(self):
            while self.running:
                gpu_stats = self.profiler._get_gpu_stats() or {}
                record = {
                    'timestamp': time.time() - self.start_time,
                    **self.profiler._get_cpu_stats(),
                    **gpu_stats
                }
                self.data.append(record)
                time.sleep(self.interval)

    def profile_inference(self, model, inputs):
        """Context manager for hardware profiling"""
        with self.ProfileSession(self) as session:
            with torch.no_grad():
                outputs = model(**inputs)
        return pd.DataFrame(session.data)

    def generate_report(self, df, filename="hardware_report.md"):
        report_path = self.output_dir / filename
        with open(report_path, 'w') as f:
            f.write("# Hardware Performance Report\n\n")
            f.write("## Peak Resource Usage\n")
            
            if 'gpu_mem_used_mb' in df:
                f.write(f"- **GPU Memory**: {df.gpu_mem_used_mb.max()}/{df.gpu_mem_total_mb.max()} MB\n")
                f.write(f"- **GPU Utilization**: {df.gpu_util_percent.max()}%\n")
                
            f.write(f"- **CPU Usage**: {df.cpu_percent.max()}%\n")
            f.write(f"- **RAM Usage**: {df.cpu_mem_used_gb.max():.1f}/{df.cpu_mem_total_gb.max():.1f} GB\n")
            
            f.write("\n## Time Series Metrics\n")
            df.to_csv(str(report_path).with_suffix('.csv'), index=False)
            
        return report_path

if __name__ == "__main__":
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    
    # Initialize model and sample input
    model = AutoModelForSequenceClassification.from_pretrained("your-finetuned-model")
    tokenizer = AutoTokenizer.from_pretrained("your-finetuned-model")
    inputs = tokenizer("A formal black dress with lace details", return_tensors="pt")
    
    # Profile hardware during inference
    profiler = HardwareProfiler()
    df = profiler.profile_inference(model, inputs)
    report = profiler.generate_report(df)