import torch
from torch.nn import KLDivLoss
from transformers import AutoModelForSequenceClassification, AutoConfig
from pathlib import Path

class DistillationTrainer:
    def __init__(self, teacher_model, config_path="configs/optimization.yaml"):
        self.config = self._load_config(config_path)['distillation']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.teacher = teacher_model.to(self.device)
        self.student = self._create_student()
        self.optimizer = torch.optim.AdamW(
            self.student.parameters(), 
            lr=self.config['learning_rate']
        )
        self.loss_fn = KLDivLoss(reduction='batchmean')

    def _load_config(self, path):
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)

    def _create_student(self):
        """Create smaller student model"""
        teacher_config = self.teacher.config
        student_config = AutoConfig.from_pretrained(self.config['student_model'])
        student_config.num_labels = teacher_config.num_labels
        return AutoModelForSequenceClassification.from_config(student_config)

    def _softmax_with_temp(self, logits, temperature):
        return torch.softmax(logits / temperature, dim=-1)

    def train_step(self, batch, temperature=2.0):
        self.teacher.eval()
        self.student.train()
        
        # Get teacher logits
        with torch.no_grad():
            teacher_outputs = self.teacher(**batch)
        
        # Student forward
        student_outputs = self.student(**batch)
        
        # Calculate distillation loss
        soft_teacher = self._softmax_with_temp(teacher_outputs.logits, temperature)
        soft_student = self._softmax_with_temp(student_outputs.logits, temperature)
        loss = self.loss_fn(soft_student.log(), soft_teacher)
        
        # Backpropagate
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

    def save_student(self, output_dir="distillation/"):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.student.save_pretrained(output_dir)
        print(f"Student model saved to {output_dir}")

if __name__ == "__main__":
    # Example usage
    teacher = AutoModelForSequenceClassification.from_pretrained("your-finetuned-model")
    trainer = DistillationTrainer(teacher)
    
    # Training loop
    for epoch in range(3):
        for batch in train_dataloader:
            loss = trainer.train_step(batch)
            print(f"Loss: {loss:.4f}")
    
    trainer.save_student()