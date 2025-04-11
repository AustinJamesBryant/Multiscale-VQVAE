import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse

# Import dataset and model components
from data.PD12M import PD12MDataModule
from model.model import ModelArgs, FinetuneLitMultiscaleVQVAE

dirpath="checkpoints_q8"

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train the model with a configuration setting.')
    parser.add_argument('--config', type=int, required=True, help='Configuration index (0-1)')
    args = parser.parse_args()

    # Static settings
    batch_size = 16
    num_workers = 32
    every_n_train_steps = 1000

    # Use match-case to select settings based on the config argument
    match args.config:
        case 0:
            # Small 1_2
            model_args = ModelArgs(
                codebook_size=16384*4,
                codebook_embed_dim=16,
                codebook_l2_norm=True,
                codebook_show_usage=True,
                commit_loss_beta=0.25,
                entropy_loss_ratio=0.0,
                encoder_ch_mult=[1, 2, 2, 4],
                decoder_ch_mult=[1, 2, 2, 4],
                z_channels=32,
                dropout_p=0.0,
            )
            patch_nums = (1, 2, 4, 6, 8, 10, 16, 24, 32)
            dirpath="checkpoints_q8_16_64k"
            every_n_train_steps = 2000
        case 1:
                # Small 1_3
                model_args = ModelArgs(
                    codebook_size=16384*4,
                    codebook_embed_dim=32,
                    codebook_l2_norm=True,
                    codebook_show_usage=True,
                    commit_loss_beta=0.25,
                    entropy_loss_ratio=0.0,
                    encoder_ch_mult=[1, 2, 2, 4],
                    decoder_ch_mult=[1, 2, 2, 4],
                    z_channels=32,
                    dropout_p=0.0,
                )
                patch_nums = (1, 2, 4, 6, 8, 10, 16, 24, 32)
                dirpath="checkpoints_q8_32_64k"
                every_n_train_steps = 2000
        case _:
            print("Invalid config argument!")
            exit(1)

    # Create dataset and model accordingly
    dm = PD12MDataModule(batch_size=batch_size, num_workers=num_workers)
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    model = FinetuneLitMultiscaleVQVAE(
        model_args=model_args,
        patch_nums=patch_nums,
        learning_rate=1e-4,
    )

    logger = TensorBoardLogger(".tensorboard", name="multiscale_vqvae_q8")

    checkpoint_callback = ModelCheckpoint(
        dirpath=dirpath,
        filename="model-{epoch:02d}-{step:08d}",
        every_n_train_steps=every_n_train_steps,
        save_top_k=-1
    )

    # Initialize the Trainer.
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="gpu",
        devices="auto",
        logger=logger,
        log_every_n_steps=20,
        callbacks=[checkpoint_callback]
    )

    # Train the model.
    trainer.fit(model, train_loader)

if __name__ == '__main__':
    main()
