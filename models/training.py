import torch
from tqdm.notebook import trange, tqdm


def train_score_model(
    model,
    train_loader,
    test_loader,
    loss_fn,
    optimizer,
    perturbation_kernel_std,
    device,
    ckpt_path,
    n_epochs=50,
    patience=5
):
    """
    Train a time-dependent score-based model with early stopping.

    Args:
        model (torch.nn.Module): Score model to train.
        train_loader (DataLoader): Training data loader.
        test_loader (DataLoader): Validation data loader.
        loss_fn (callable): Loss function taking (model, x, perturbation_kernel_std).
        optimizer (torch.optim.Optimizer): Optimizer.
        perturbation_kernel_std (callable): Function std(t) giving noise std at time t.
        device (torch.device): Device to use (e.g., 'cuda' or 'cpu').
        ckpt_path (str): Path to save the best model checkpoint.
        n_epochs (int): Maximum number of training epochs.
        patience (int): Early stopping patience.

    Returns:
        (train_losses, val_losses): Lists of training and validation losses per epoch.
    """
    tqdm_epoch = trange(n_epochs, leave=True)
    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    counter_es = 0

    for epoch in tqdm_epoch:
        model.train()
        avg_train_loss = 0.0

        for batch in tqdm(train_loader, leave=False):
            x = batch[0] if isinstance(batch, (tuple, list)) else batch # handle datasets with or without labels
            x = x.to(device)
            loss = loss_fn(model, x, perturbation_kernel_std)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_train_loss += loss.item()

        avg_train_loss /= len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        avg_val_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                x = batch[0] if isinstance(batch, (tuple, list)) else batch
                x = x.to(device)
                val_loss = loss_fn(model, x, perturbation_kernel_std)
                avg_val_loss += val_loss.item()
        avg_val_loss /= len(test_loader)
        val_losses.append(avg_val_loss)

        tqdm_epoch.set_description(f"Epoch {epoch+1} - Train: {avg_train_loss:.4f} - Val: {avg_val_loss:.4f}")
        tqdm_epoch.refresh()
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter_es = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            counter_es += 1
            if counter_es >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return train_losses, val_losses