import torch
import os
import math
import time
import math
import os
import csv
import torch.nn.functional as F
from predictive_coding.config import GPTConfig
from predictive_coding.pc_layer import PCLayer
from model_architecture.pc_t_model import PCTransformer
from Data_preprocessing.dataloader import train_loader
from utils.model_utils import load_tokenizer, reset_pc_modules
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

"""Usage: python training.py"""

# Define CSV log file
csv_log_path = "logs/lr_tracking.csv"
os.makedirs(os.path.dirname(csv_log_path), exist_ok=True)


# Write CSV header once at the beginning (e.g. in `main()` or at start of train)
if not os.path.exists(csv_log_path):
    with open(csv_log_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["global_step", "module_name", "local_lr"])
        
def train(model, dataloader, tokenizer, config):
    model.train()
    total_energy = 0.0
    total_ce_loss = 0.0
    batch_count = 0
    global_step = 0

    pad_token_id = tokenizer.token_to_id("[PAD]")

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"]
        target_ids = batch["target_ids"]
        
        # Update learning rate (linear warmup)
        if global_step < config.warmup_steps:
            lr = global_step / config.warmup_steps * config.local_learning_rate
        else:
            lr = config.peak_learning_rate

            
        # Set this learning rate for each PCLayer
        for module in model.modules():
            # print(f"Module: {module.__class__.__name__}, Local LR: {getattr(module, 'local_lr', None)}")
            if hasattr(module, 'local_lr'):
                module.local_lr = lr
                # Log to CSV
                with open(csv_log_path, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([global_step, module.__class__.__name__, module.local_lr])

        if global_step % 50 == 0:
            print(f"[Step {global_step}] Learning Rate: {lr:.6f}")
            
        global_step += 1

        logits = model(target_ids, input_ids)

        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
            ignore_index= pad_token_id
        )
        
        perplexity = math.exp(ce_loss.item()) if ce_loss.item() < 100 else float("inf")
        
        total_ce_loss += ce_loss.item()

        layer_energies = []
        attn_block_idx = 0 
        for module in model.modules():
            if isinstance(module, PCLayer) and hasattr(module, "get_energy"):
                energy = module.get_energy()
                if energy is not None:
                    layer_energies.append(energy)
                if hasattr(module, "_head_similarity"):
                    avg_sim = module._head_similarity_avg
                    max_sim = module._head_similarity_max
                    # print(f"  Attn Layer {attn_block_idx} | Avg Head Sim: {avg_sim:.4f}, Max Pair: {max_sim:.4f}")
                    # attn_block_idx += 1

        # Compute average energy for current batch
        batch_energy = ce_loss.item() if not layer_energies else sum(layer_energies) / len(layer_energies)
        total_energy += batch_energy
        batch_count += 1
        perplexity = math.exp(ce_loss.item()) if ce_loss.item() < 100 else float("inf")
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)} | Batch Energy: {batch_energy:.4f} | Perplexity: {perplexity:.4f}", flush=True)

        reset_pc_modules(model)

    avg_energy = total_energy / batch_count if batch_count > 0 else 0.0
    avg_ce_loss = total_ce_loss / batch_count if batch_count > 0 else 0.0
    avg_perplexity = math.exp(avg_ce_loss) if avg_ce_loss < 100 else float("inf")    
    return avg_energy, avg_perplexity

def main():
    tokenizer = load_tokenizer()
    vocab_size = tokenizer.get_vocab_size()

    config = GPTConfig(
        vocab_size = vocab_size,
        block_size= 256,
        n_embed=64,
        dropout=0.1,
        T=5,
        is_holding_error = True,
        num_heads=2,
        n_blocks=4,
        num_epochs=5,
        update_bias=True,
        use_lateral = True,
        energy_fn_name="kld",
        warmup_steps= 1000,
        peak_learning_rate = 1e-5
    )

    model = PCTransformer(config)
    train_energies = []
    perplexities = []

    print("========== Training started ==========", flush=True) 
    # Measure total training time
    start_training_time = time.time()
    for epoch in range(config.num_epochs):
        print(f"Epoch {epoch+1} started", flush=True)
        avg_energy, perplexity = train(model, train_loader, tokenizer, config)
        train_energies.append(avg_energy)
        perplexities.append(perplexity)
        print(f"Epoch {epoch+1} | Avg Energy: {avg_energy:.4f} | Perplexity: {perplexity:.4f}", flush=True)
    total_training_time = time.time() - start_training_time
    print(f"Total Training Time: {total_training_time:.2f} seconds", flush=True)
    print("========== Training completed ==========", flush=True)

    # Saving trained model
    save_path = "checkpoints/pc_transformer.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if os.path.exists(save_path):
        os.remove(save_path)
    torch.save({"model_state": model.state_dict()}, save_path)
    print("Model saved.")

    # Plotting average energy vs. epoch
    epochs = list(range(1, len(train_energies) + 1))
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_energies, marker='o', linestyle='-', color='b', label='Average Batch Energy')
    plt.xlabel('Epoch')
    plt.ylabel('Average Batch Energy')
    plt.title('Average Batch Energy vs. Epoch')
    plt.grid(True)
    plt.legend()
    # Force x-axis to show only whole numbers
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig('assets/energy_plot.png')
    plt.show()

if __name__ == "__main__":
    main()
