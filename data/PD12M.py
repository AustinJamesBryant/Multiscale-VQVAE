import os
import torch
import asyncio
import aiohttp
from io import BytesIO
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_dataset

Image.MAX_IMAGE_PIXELS = None

# Example transform pipeline
transform = transforms.Compose([
    transforms.RandomCrop(256),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
])

# async def fetch_image(session, url, transform=transform):
#     try:
#         async with session.get(url, timeout=10) as response:
#             response.raise_for_status()
#             content = await response.read()
#             img = Image.open(BytesIO(content)).convert("RGB")
#             return transform(img), 0  # 0 => success
#     except Exception as e:
#         print(f"Error loading {url}: {e}")
#         return torch.rand(3, 256, 256), 1  # 1 => fallback

async def fetch_image(session, url, transform=transform, retries=3, timeout=10):
    for attempt in range(retries):
        try:
            async with session.get(url, timeout=timeout) as response:
                response.raise_for_status()
                content = await response.read()
                img = Image.open(BytesIO(content)).convert("RGB")
                return transform(img), 0  # Success
        except Exception as e:
            # Optionally log the attempt failure
            if attempt == retries - 1:
                print(f"Error loading {url}: {e}")
                return torch.zeros((3, 256, 256)), 1  # Fallback image
            await asyncio.sleep(1)  # Wait a moment before retrying

async def fetch_batch(batch, transform=transform):
    # We'll store the final results (tensors) for each item in the batch
    batch_images = []
    batch_labels = []

    # Create one aiohttp session per batch
    async with aiohttp.ClientSession() as session:
        # For each item in the batch, gather tasks for all its URLs
        tasks = []
        for item in batch:
            urls = item["url"]
            if not isinstance(urls, list):
                urls = [urls]

            subtasks = [fetch_image(session, u, transform=transform) for u in urls]
            tasks.append(asyncio.gather(*subtasks))

        # Wait for all items in the batch to finish fetching
        all_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Now, `all_results` is a list (one entry per item). Each entry is a list of (tensor, label).
    for item_result in all_results:
        images_list, labels_list = zip(*item_result)  # unpack
        images_tensor = torch.stack(images_list, dim=0)  # shape: (num_urls, 3, 256, 256)
        labels_tensor = torch.tensor(labels_list)       # shape: (num_urls,)
        batch_images.append(images_tensor)
        batch_labels.append(labels_tensor)

    # Stack over the batch dimension
    batch_images = torch.stack(batch_images, dim=0)  # shape: (batch_size, num_urls, 3, 256, 256)
    batch_labels = torch.stack(batch_labels, dim=0)  # shape: (batch_size, num_urls)
    return batch_images, batch_labels

def collate_fn_async(batch):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    images, labels = loop.run_until_complete(fetch_batch(batch, transform))
    images = images.squeeze(1)
    loop.close()
    return images, labels

class PD12MDataset(Dataset):
    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset

    def __getitem__(self, idx):
        return self.hf_dataset[idx]

    def __len__(self):
        return len(self.hf_dataset)

class PD12MDataModule:
    def __init__(
        self,
        batch_size=16,
        num_workers=4,
        test_size=0.8,
        seed=42
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_size = test_size
        self.seed = seed

        dataset = load_dataset("Spawning/PD12M")
        columns_to_remove = ["id", "caption", "width", "height", "mime_type", "hash", "license", "source"]
        dataset = dataset.remove_columns(columns_to_remove)

        if "train" in dataset:
            full_dataset = dataset["train"]
        else:
            full_dataset = list(dataset.values())[0]

        split_dataset = full_dataset.train_test_split(test_size=self.test_size, seed=self.seed)
        train_hf = split_dataset["train"]
        val_hf = split_dataset["test"]

        self.train_dataset = PD12MDataset(train_hf)
        self.val_dataset   = PD12MDataset(val_hf)

        self._train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn_async
        )

        self._val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn_async
        )

    def train_dataloader(self):
        return self._train_loader

    def val_dataloader(self):
        return self._val_loader