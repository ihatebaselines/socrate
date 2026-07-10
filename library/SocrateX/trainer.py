import torch
from tqdm import tqdm

class Trainer:
    """
    The Trainer class enables fully customizable training of the SOCRATE model.
    You can pass a custom loss function (criterion), optimizer, scheduler, and a scaler for AMP.
    """
    def __init__(self, model, optimizer, scheduler, criterion, device, scaler=None, save_dir=".", save_name="socrate", save_interval=100):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.scaler = scaler
        self.save_dir = save_dir
        self.save_name = save_name
        self.save_interval = save_interval
        
        import os
        if self.save_dir and self.save_dir != ".":
            os.makedirs(self.save_dir, exist_ok=True)

    def train_epoch(self, dataloader, best_loss, epoch_num=1, total_epochs=1):
        """
        Trains a full epoch over the dataloader.
        Returns the new `best_loss`.
        """
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch_num}/{total_epochs}")

        for step, (image, t1, t2) in enumerate(pbar, 1):
            image = image.to(self.device, non_blocking=True)
            t1 = t1.to(self.device, non_blocking=True)
            t2 = t2.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            if self.scaler is not None:
                # Mixed Precision Training (AMP)
                with torch.amp.autocast(device_type="cuda"):
                    output = self.model(image, t1)
                    loss = self.criterion(
                        output.reshape(-1, output.size(-1)),
                        t2.reshape(-1)
                    )

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard Training (FP32)
                output = self.model(image, t1)
                loss = self.criterion(
                    output.reshape(-1, output.size(-1)),
                    t2.reshape(-1)
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()
            avg_loss = total_loss / step

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                avg_loss=f"{avg_loss:.4f}",
                best=f"{best_loss:.4f}"
            )

            # Periodic saving during the epoch (can be extracted if too rigid)
            if step % self.save_interval == 0:
                import os
                self.save_checkpoint(os.path.join(self.save_dir, f"{self.save_name}_previous.pt"), best_loss, avg_loss, step)
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    self.save_checkpoint(os.path.join(self.save_dir, f"{self.save_name}_best.pt"), best_loss, avg_loss, step)

        epoch_avg_loss = total_loss / len(dataloader)
        if epoch_avg_loss < best_loss:
            best_loss = epoch_avg_loss
            import os
            self.save_checkpoint(os.path.join(self.save_dir, f"{self.save_name}_best.pt"), best_loss, epoch_avg_loss, step)

        print(f"Train Loss pt Epoch {epoch_num}: {epoch_avg_loss:.4f}")
        return best_loss

    def save_checkpoint(self, filename, best_loss, avg_loss, step):
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "best_loss": best_loss,
            "avg_loss": avg_loss,
            "step": step,
        }
        torch.save(checkpoint, filename)

def train(model, dataloader, optimizer, scheduler, criterion, device, best_loss, scaler, save_dir=".", save_name="socrate", save_interval=100):
    """
    The original function exposed for backward compatibility with the old script.
    """
    trainer = Trainer(model, optimizer, scheduler, criterion, device, scaler, save_dir, save_name, save_interval)
    return trainer.train_epoch(dataloader, best_loss)
