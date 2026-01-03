class LMDBDataset(Dataset):
    def __init__(self, lmdb_path, transform=None):
        self.lmdb_path = lmdb_path
        self.transform = transform

        # Open once ONLY to read keys, then close
        env = lmdb.open(
            lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            subdir=False
        )

        with env.begin() as txn:
            self.keys = [k for k, _ in txn.cursor() if k != b"__len__"]

        env.close()

        # Critical: length derived from keys, not __len__
        self.length = len(self.keys)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx >= self.length:
            raise IndexError

        # Open LMDB locally (safe for Windows)
        env = lmdb.open(
            self.lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            subdir=False
        )

        with env.begin() as txn:
            data = pickle.loads(txn.get(self.keys[idx]))

        env.close()

        img = Image.open(io.BytesIO(data["image"])).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, data["label"]