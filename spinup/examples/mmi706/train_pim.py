import pathlib
from argparse import ArgumentParser, Namespace

import skimage.io
import torch
import torch.nn.functional as F
import torchvision
import yaml
from pytorch_lightning import LightningModule, Trainer, seed_everything
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, random_split

from spinup.examples.mmi706.path_integration import PathIntegrationModule
from spinup.examples.mmi706.vision import make_vision_module

Tensor = torch.Tensor


def collate_fn_pad(batch):
    n_timesteps = max([image.shape[0] for image, _ in batch])

    images = []
    actions = []
    masks = []
    for image, action in batch:
        t = image.shape[0]
        mask = torch.ones(n_timesteps)
        if t < n_timesteps:
            image_pad = torch.zeros(n_timesteps - t, *image.shape[1:], dtype=image.dtype)
            action_pad = torch.zeros(n_timesteps - t, *action.shape[1:], dtype=action.dtype)
            image = torch.cat([image, image_pad], 0)
            action = torch.cat([action, action_pad], 0)
            mask[t:] = 0.0

        images.append(image)
        actions.append(action)
        masks.append(mask)

    return torch.stack(images, 0), torch.stack(actions, 0), torch.stack(masks, 0)


class PIMDataset(Dataset):
    def __init__(self, root):
        self.root = pathlib.Path(root)
        self.episodes = sorted([p.stem for p in (self.root / 'images').glob('*') if p.is_dir()])

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, index):
        episode = self.episodes[index]

        image_dir = self.root / 'images' / episode
        image_filepaths = list(image_dir.glob("*.jpg"))
        images = [
            torch.from_numpy(skimage.io.imread(str(fp)).transpose(2, 0, 1)) / 255.0
            for fp in image_filepaths
        ]

        action_dir = self.root / 'actions' / episode
        action_filepaths = [action_dir / (fp.stem + '.txt') for fp in image_filepaths]
        actions = [torch.tensor(int(fp.read_text()), dtype=torch.int64)
                   for fp in action_filepaths]

        image_tensor = torch.stack(images, 0)
        action_tensor = torch.stack(actions, 0)
        return image_tensor, action_tensor


class PIMExperiment(LightningModule):
    def __init__(self, vision_module, hparams) -> None:
        super().__init__()

        self.vision_module = vision_module
        hparams.model['visual_feature_size'] = self.vision_module.output_shape[0]

        self.model = PathIntegrationModule(**hparams.model)
        self.hparams = hparams
        self.current_device = None

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # data args
        parser.add_argument('--data_dir', type=str, default=str(pathlib.Path('./data')))
        parser.add_argument('--num_workers', type=int, default=1)
        parser.add_argument('--batch_size', type=int, default=4)
        # optimizer args
        parser.add_argument('--learning_rate', type=float, default=1e-4)
        parser.add_argument('--weight_decay', type=float, default=0.0)
        # vision module
        parser.add_argument('--vision_module_checkpoint', type=str, required=True)
        return parser

    def forward(self, image: Tensor) -> Tensor:
        return self.model(image)

    def training_step(self, batch, batch_idx):
        image, action, mask = batch  # (B, T, C, H, W), (B, T), (B, T)
        batch_size, n_timesteps = image.shape[:2]

        image = image.permute(1, 0, 2, 3, 4)  # (T, B, C, H, W)
        action = action.permute(1, 0)  # (T, B)
        action = F.one_hot(action, num_classes=self.model.action_space_dim).float()  # (T, B, A)
        mask = mask.permute(1, 0)  # (T, B)

        loss = 0
        hx, cx = self.model.initial_hidden_state()
        hx = hx.repeat(batch_size, 1)
        cx = cx.repeat(batch_size, 1)
        for i in range(n_timesteps):
            with torch.no_grad():
                vfm = self.vision_module(image[i])[0]  # (B, F)
            a = action[i]  # (B, A)
            m = mask[i].unsqueeze(1)  # (B, 1)
            grid_activations, pvfm, (hx, cx) = self.model(vfm, a, (hx, cx))
            loss += self.model.loss(vfm * m, pvfm * m)

        log = dict(
            loss=loss.detach(),
        )

        return dict(loss=loss, log=log)

    def validation_step(self, batch, batch_idx):
        image, action, mask = batch  # (B, T, C, H, W), (B, T), (B, T)
        batch_size, n_timesteps = image.shape[:2]

        image = image.permute(1, 0, 2, 3, 4)  # (T, B, C, H, W)
        action = action.permute(1, 0)  # (T, B)
        action = F.one_hot(action, num_classes=self.model.action_space_dim).float()  # (T, B, A)
        mask = mask.permute(1, 0)  # (T, B)

        loss = 0
        hx, cx = self.model.initial_hidden_state()
        hx = hx.repeat(batch_size, 1)
        cx = cx.repeat(batch_size, 1)
        for i in range(n_timesteps):
            with torch.no_grad():
                vfm = self.vision_module(image[i])[0]  # (B, F)
            a = action[i]  # (B, A)
            m = mask[i].unsqueeze(1)  # (B, 1)
            grid_activations, pvfm, (hx, cx) = self.model(vfm, a, (hx, cx))
            loss += self.model.loss(vfm * m, pvfm * m)

        out = dict(val_loss=loss)
        if batch_idx == 0:
            out['images'] = image[:, 0]
            out['actions'] = action[:, 0]
        return out

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        # log predictions
        images = outputs[0]['images']
        actions = outputs[0]['actions']
        self.simulate_episode(images, actions)

        return {'val_loss': avg_val_loss}

    def simulate_episode(self, images, actions):  # (T, C, H, W), (T, A)
        n_timesteps = images.shape[0]
        encoding_shape = self.vision_module.vqvae.output_shapes['quantized']

        recs = []
        precs = []
        hx, cx = self.model.initial_hidden_state()
        for i in range(n_timesteps):
            image = images[i].unsqueeze(0)  # (1, C, H, W)
            action = actions[i].unsqueeze(0)  # (1, A)
            with torch.no_grad():
                vfm = self.vision_module(image)[0]  # (1, F)
            _, pvfm, (hx, cx) = self.model(vfm, action, (hx, cx))

            with torch.no_grad():
                rec = self.vision_module.vqvae.decode(vfm.view(1, *encoding_shape))
                prec = self.vision_module.vqvae.decode(pvfm.view(1, *encoding_shape))

            recs.append(rec)
            precs.append(prec)

        episode_rec = torch.cat(recs, 0)
        episode_prec = torch.cat(precs, 0)
        grid = torchvision.utils.make_grid(
            torch.cat([images, episode_rec, episode_prec]),
            nrow=n_timesteps, pad_value=0, padding=1,
        )
        self.logger.experiment.add_image('episode', grid, self.current_epoch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(),
                                lr=self.hparams.learning_rate,
                                weight_decay=self.hparams.weight_decay)

    def prepare_data(self):
        ds = PIMDataset(self.hparams.data_dir)
        n = len(ds)
        tn = int(n * 0.8)
        vn = n - tn
        self.train_dataset, self.val_dataset = random_split(ds, [tn, vn])

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          collate_fn=collate_fn_pad)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          collate_fn=collate_fn_pad)


if __name__ == '__main__':
    # For reproducibility
    seed_everything(42)
    cudnn.deterministic = True
    cudnn.benchmark = False

    parser = ArgumentParser()
    parser.add_argument('--hparams_file', type=str, default=None)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # make vision module
    vision_module_checkpoint = "/Users/bdsaglam/PycharmProjects/LearnRL/data/checkpoints/vqvae.pt"
    vision_module = make_vision_module('VQVAE', vision_module_checkpoint)

    # prepare hparams
    hparams_file = pathlib.Path(args.hparams_file)
    hparams = yaml.safe_load(hparams_file.read_text())

    experiment = PIMExperiment(
        vision_module=vision_module,
        hparams=Namespace(**hparams),
    )

    # prepare trainer params
    trainer_params = vars(args)
    del trainer_params['hparams_file']
    runner = Trainer(**trainer_params)

    runner.fit(experiment)
