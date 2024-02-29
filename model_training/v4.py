import wandb
wandb.init(project='pytorch-deep', name='model_training_v4')

# Defines number of epochs
n_epochs = 200

losses = []
val_losses = []

wandb.config = {
  "epochs": n_epochs,
}

for epoch in range(n_epochs):
    # inner loop
    loss = mini_batch(device, train_loader, train_step_fn)
    losses.append(loss)
    wandb.log({"epoch": epoch, "train_loss": loss})
    
    # VALIDATION
    # no gradients in validation!
    with torch.no_grad():
        val_loss = mini_batch(device, val_loader, val_step_fn)
        val_losses.append(val_loss)
        wandb.log({"epoch": epoch, "val_loss": val_loss})

wandb.finish()
