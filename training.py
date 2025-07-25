import os
import torch
import os
import math
import time
import torch.nn.functional as F
from predictive_coding.config import GPTConfig
from predictive_coding.pc_layer import PCLayer
from model_architecture.pc_t_model import PCTransformer
from Data_preprocessing.dataloader import get_loaders
from utils.model_utils import load_tokenizer, reset_pc_modules
from visualization import plot_metrics

"""
Usage: python training.py

This script trains a predictive coding transformer model on a dataset.
It tracks and plots the average predictive coding energy per epoch and saves the trained model.
"""

def train(model, dataloader, tokenizer, global_step, device):
    model.train()
    total_energy = 0.0
    total_ce_loss = 0.0
    batch_count = 0
    pad_token_id = tokenizer.token_to_id("[PAD]")
    vocab_size = tokenizer.get_vocab_size()

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        
        if global_step < GPTConfig.warmup_steps:
            lr = GPTConfig.local_learning_rate + global_step / GPTConfig.warmup_steps * (
                GPTConfig.peak_learning_rate - GPTConfig.local_learning_rate)
        else:
            lr = GPTConfig.peak_learning_rate

        for module in model.modules():
            if hasattr(module, 'local_lr'):
                module.local_lr = lr

        global_step += 1
        
        if target_ids.max() >= vocab_size:
            target_ids = torch.clamp(target_ids, max=vocab_size-1)

        logits = model(target_ids, input_ids)

        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
            ignore_index=pad_token_id)
        total_ce_loss += ce_loss.item()

        layer_energies = []
        for module in model.modules():
            if isinstance(module, PCLayer) and hasattr(module, "get_energy"):
                energy = module.get_energy()
                if energy is not None and not (torch.isnan(torch.tensor(energy)) if isinstance(energy, (int, float)) else False):
                    layer_energies.append(energy)
                if hasattr(module, "_head_similarity"):
                    _ = module._head_similarity_avg
                    _ = module._head_similarity_max
                    
        if layer_energies:
            valid_energies = [e for e in layer_energies if not (torch.isnan(torch.tensor(e)) if isinstance(e, (int, float)) else True)]
            batch_energy = sum(valid_energies) / len(valid_energies) if valid_energies else ce_loss.item()
        else:
            batch_energy = ce_loss.item()
        total_energy += batch_energy
        batch_count += 1
        perplexity = math.exp(ce_loss.item()) if ce_loss.item() < 100 else float("inf")
        
        if (batch_idx + 1) % 10 == 0:
             print(f"  Batch {batch_idx + 1}/{len(dataloader)} | Batch Energy: {batch_energy:.4f} | Perplexity: {perplexity:.4f} | LR: {lr:.6f}", flush=True)

        reset_pc_modules(model)

    avg_energy = total_energy / batch_count if batch_count > 0 else 0.0
    avg_ce_loss = total_ce_loss / batch_count if batch_count > 0 else 0.0
    avg_perplexity = math.exp(avg_ce_loss) if avg_ce_loss < 100 else float("inf")    
    return avg_energy, avg_perplexity, global_step

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")

    tokenizer = load_tokenizer()
    vocab_size = tokenizer.get_vocab_size()

    config = GPTConfig(
        vocab_size = vocab_size,
        block_size= 256, 
        peak_learning_rate= 1.29e-04,
        warmup_steps= 58,
        n_embed=64,
        dropout=0.1,
        local_learning_rate= 0.0,
        T= 1,
        is_holding_error = True,
        num_heads=8,
        n_blocks=4,
        num_epochs= 2,
        update_bias=True,
        use_lateral = True,
        energy_fn_name="scaled_mse",
        eos_token_id = tokenizer.token_to_id("[EOS]")
    )
    model = PCTransformer(config)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    
    model = model.to(device)

    if hasattr(model, 'module'): 
        model.module.register_all_lateral_weights()
    else:
        model.register_all_lateral_weights()

    train_loader, valid_loader, _ = get_loaders()

    print("========== Training started ==========") 
    start_time = time.time()
    global_step = 0
    
    train_energies = []
    val_energies = []

    for epoch in range(config.num_epochs):
        print(f"Epoch {epoch+1}/{config.num_epochs}")
        model.train()
        train_energy, train_perplexity, _ = train(model, train_loader, tokenizer, global_step, device)
        train_energies.append(train_energy)
        
        model.eval()
        val_energy, val_perplexity, global_step = train(model, valid_loader, tokenizer, global_step, device)
        val_energies.append(val_energy)
        
        print(f"Epoch {epoch+1}/{config.num_epochs} | "
        f"Train Energy: {train_energy:.4f} | Train Perplexity: {train_perplexity:.4f} | "
        f"Val Energy: {val_energy:.4f} | Val Perplexity: {val_perplexity:.4f}")


        if (epoch + 1) % 5 == 0:
                os.makedirs("checkpoints", exist_ok=True)
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                    'train_energy': train_energy,
                    'val_energy': val_energy,
                    'train_perplexity': train_perplexity,
                    'val_perplexity': val_perplexity
                }
                
                # Save checkpoint
                checkpoint_path = f'checkpoints/model_epoch_{epoch+1}.pt'
                torch.save(checkpoint, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
                
    # Save visualization
    plot_metrics(train_energies, val_energies)
    
    # Save final model
    final_checkpoint = {
        'epoch': config.num_epochs,
        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'train_energy': train_energy,
        'val_energy': val_energy,
        'train_perplexity': train_perplexity,
        'val_perplexity': val_perplexity
    }
    torch.save(final_checkpoint, 'checkpoints/final_model.pt')
    
    total_time = (time.time() - start_time) / 3600 
    print(f"\nTraining completed in {total_time:.2f} hours")
    print("Final model saved to: checkpoints/final_model.pt")
    print("========== Training completed ==========")

if __name__ == "__main__":
    main()
