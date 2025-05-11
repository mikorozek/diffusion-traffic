import torch
import torch.optim as optim
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os

from src.unet import UNet
from src.diffusion import Diffusion


DATASET_PATH = "/home/mrozek/ssne-2025l/diffusion-traffic/trafic_32"
IMAGE_SIZE = 32
BATCH_SIZE = 8
VAL_SPLIT_RATIO = 0.1

IMG_CHANNELS = 3
BASE_CHANNELS = 64
CHANNEL_MULTIPLIERS = (1, 2, 4)

NUM_RESIDUAL_BLOCKS = 2

DROPOUT_RATE = 0.1
GN_GROUPS = 16

NUM_TIMESTEPS = 1000
COSINE_S = 0.008

LEARNING_RATE = 1e-4
EPOCHS = 20
MODEL_SAVE_PATH = (
    "/home/mrozek/ssne-2025/diffusion-traffic/models/traffic_sign_unet.pth"
)
SAVE_EVERY_N_EPOCHS = 2
LOG_IMAGES_EVERY_N_EPOCHS = 5
NUM_IMAGES_TO_LOG = 8


def denormalize_images(images_tensor):
    return (images_tensor + 1) / 2


def log_generated_images_to_wandb(
    unet_model,
    diffusion_process,
    epoch_num,
    num_images=8,
    image_shape=(3, 32, 32),
):
    unet_model.eval()
    with torch.no_grad():
        print(f"Generating images for wandb at epoch {epoch_num+1}...")
        generated_images_tensor = diffusion_process.sample(
            unet_model=unet_model,
            num_images=num_images,
            image_shape=image_shape,
            batch_size=num_images,
        )
        denormalized_images = denormalize_images(generated_images_tensor)

        wandb_images = [wandb.Image(img) for img in denormalized_images]
        wandb.log({"Generated Samples": wandb_images}, step=epoch_num + 1)
    unet_model.train()
    print("Images logged to wandb.")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    hyperparameters = {
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "val_split_ratio": VAL_SPLIT_RATIO,
        "img_channels": IMG_CHANNELS,
        "base_channels": BASE_CHANNELS,
        "channel_multipliers": CHANNEL_MULTIPLIERS,
        "num_residual_blocks": NUM_RESIDUAL_BLOCKS,
        "dropout_rate": DROPOUT_RATE,
        "gn_groups": GN_GROUPS,
        "num_timesteps": NUM_TIMESTEPS,
        "cosine_s": COSINE_S,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
    }
    wandb.init(project="diffusion-traffic-signs", config=hyperparameters)

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(24),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    full_dataset = ImageFolder(DATASET_PATH, transform=train_transform)
    dataset_size = len(full_dataset)
    val_size = int(VAL_SPLIT_RATIO * dataset_size)
    train_size = dataset_size - val_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1
    )

    print(f"Loaded dataset from {DATASET_PATH}. Number of images: {len(full_dataset)}")

    unet_model = UNet(
        in_channels=IMG_CHANNELS,
        model_channels=BASE_CHANNELS,
        out_channels=IMG_CHANNELS,
        num_residual_blocks=NUM_RESIDUAL_BLOCKS,
        num_normalization_groups=GN_GROUPS,
        dropout_rate=DROPOUT_RATE,
        channel_mult=CHANNEL_MULTIPLIERS,
    ).to(device)

    diffusion_process = Diffusion(
        num_timesteps=NUM_TIMESTEPS,
        cosine_s=COSINE_S,
        device=device,
    )

    optimizer = optim.AdamW(unet_model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    start_epoch = 0
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Loading checkpoint from {MODEL_SAVE_PATH}...")
        checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)
        unet_model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming training from epoch {start_epoch}")

    print(
        f"Number of parameters in UNet model: {sum(p.numel() for p in unet_model.parameters() if p.requires_grad):,}"
    )
    print("Starting training...")

    for epoch in range(start_epoch, EPOCHS):
        unet_model.train()
        train_loss = 0.0

        for i, (batch_images, _) in enumerate(train_loader):
            optimizer.zero_grad()

            x_0 = batch_images.to(device)

            t = torch.randint(0, NUM_TIMESTEPS, (x_0.shape[0],), device=device).long()

            epsilon_noise = torch.randn_like(x_0)

            x_t = diffusion_process.q_sample(x_0, t, noise=epsilon_noise)

            predicted_noise = unet_model(x_t, t)

            loss = criterion(predicted_noise, epsilon_noise)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        epoch_train_loss = train_loss / len(train_loader)
        wandb.log({"epoch": epoch + 1, "avg_train_loss": epoch_train_loss})

        unet_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, (batch_images, _) in enumerate(val_loader):
                x_0 = batch_images.to(device)
                t = torch.randint(
                    0, NUM_TIMESTEPS, (x_0.shape[0],), device=device
                ).long()
                epsilon_noise = torch.randn_like(x_0)
                x_t = diffusion_process.q_sample(x_0, t, noise=epsilon_noise)
                predicted_noise = unet_model(x_t, t)
                loss = criterion(predicted_noise, epsilon_noise)
                val_loss += loss.item()

        epoch_val_loss = val_loss / len(val_loader)
        print(
            f"Epoch {epoch+1}, Train Loss: {epoch_train_loss}, Val Loss: {epoch_val_loss:.4f}"
        )
        wandb.log({"epoch": epoch + 1, "avg_val_loss": epoch_val_loss})

        if (epoch + 1) % SAVE_EVERY_N_EPOCHS == 0 or (epoch + 1) == EPOCHS:
            print(f"Saving model at epoch {epoch+1}...")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": unet_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": epoch_train_loss,
                },
                MODEL_SAVE_PATH,
            )

    print("Training finished.")
    print(f"Final model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
