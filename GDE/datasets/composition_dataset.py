from itertools import product
import torch
import os
import torch.nn.functional as F
from collections import defaultdict

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import (CenterCrop, Compose, InterpolationMode,
                                    Normalize, RandomHorizontalFlip,
                                    RandomPerspective, RandomRotation, Resize,
                                    ToTensor)
from torchvision.transforms.transforms import RandomResizedCrop

BICUBIC = InterpolationMode.BICUBIC
n_px = 224


def transform_image(split="train", imagenet=False):
    if imagenet:
        # from czsl repo.
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        transform = Compose(
            [
                RandomResizedCrop(n_px),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(
                    mean,
                    std,
                ),
            ]
        )
        return transform

    if split == "test" or split == "val":
        transform = Compose(
            [
                Resize(n_px, interpolation=BICUBIC),
                CenterCrop(n_px),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
    else:
        transform = Compose(
            [
                # RandomResizedCrop(n_px, interpolation=BICUBIC),
                Resize(n_px, interpolation=BICUBIC),
                CenterCrop(n_px),
                RandomHorizontalFlip(),
                RandomPerspective(),
                RandomRotation(degrees=5),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    return transform

class ImageLoader:
    def __init__(self, root):
        self.img_dir = root

    def __call__(self, img):
        file = '%s/%s' % (self.img_dir, img)
        img = Image.open(file).convert('RGB')
        return img


class CompositionDataset(Dataset):
    def __init__(
            self,
            root,
            phase,
            split='compositional-split-natural',
            open_world=False,
            transform=None
    ):
        # General attributes
        self.root = root
        self.phase = phase
        self.split = split
        self.open_world = open_world

        self.feat_dim = None
        self.transform = transform
        self.loader = ImageLoader(self.root + '/images/')


        #added attrs1 and attrs2 for hadling 2 attribute positions in the triplet
        self.attrs1, self.attrs2, self.objs, self.triplets, self.train_triplets, self.val_triplets, self.test_triplets = self.parse_split()

        #product done with 3-positional triplets
        self.full_triplets = list(product(self.attrs1, self.attrs2, self.objs))
        if self.open_world:
            self.triplets = self.full_triplets

        # phase-specific attributes
        self.train_data, self.val_data, self.test_data = self.get_split_info()
        if self.phase == 'train':
            self.data = self.train_data
        elif self.phase == 'val':
            self.data = self.val_data
        elif self.phase == 'test':
            self.data = self.test_data
        else: # get all data
            self.data = self.train_data + self.val_data + self.test_data
        _, self.all_attrs1, self.all_attrs2, self.all_objs = zip(*self.data)
        self.all_triplets = list(zip(self.all_attrs1, self.all_attrs2, self.all_objs))

        self.obj2idx = {obj: idx for idx, obj in enumerate(self.objs)}
        self.attr1_idx = {attr1: idx for idx, attr1 in enumerate(self.attrs1)} #new
        self.attr2_idx = {attr2: idx for idx, attr2 in enumerate(self.attrs2)} #new
        self.triplet2idx = {triplet: idx for idx, triplet in enumerate(self.triplets)}

        self.train_triplet_to_idx = dict(
            [(triplet, idx) for idx, triplet in enumerate(self.train_triplets)]
        )

        # Some potentially usefull info
        seen_triplets = set(self.train_triplets)
        self.seen_mask = torch.BoolTensor(
            [triplet in seen_triplets for triplet in self.triplets]
        )

        self.objs_by_attr= {(a1, a2): [] for (a1, a2, o) in self.triplets}
        self.attrs_by_obj = {k: [] for k in self.objs}
        for (a1, a2, o) in self.all_triplets:
            self.objs_by_attr[(a1, a2)].append(o)
            self.attrs_by_obj[o].append((a1, a2))

    def get_split_info(self):
        data = torch.load(self.root + '/metadata_{}.t7'.format(self.split), weights_only=False)
        train_data, val_data, test_data = [], [], []
        triplets = set(self.triplets)
        for instance in data:
            image, attr1, attr2, obj, settype = instance['image'], instance[
                'attr1'], instance['attr2'], instance['obj'], instance['set']

            if attr1 == 'NA' or attr2 == 'NA' or (attr1, attr2, obj) not in triplets or settype == 'NA':
                # ignore instances with unlabeled attributes
                # ignore instances that are not in current split
                continue

            data_i = [image, attr1, attr2, obj]
            if settype == 'train':
                train_data.append(data_i)
            elif settype == 'val':
                val_data.append(data_i)
            else:
                test_data.append(data_i)

        return train_data, val_data, test_data

    def parse_split(self):
        def parse_triplets(triplet_list):
            with open(triplet_list, 'r') as f:
                triplets = f.read().strip().split('\n')
                # triplets = [t.split() if not '_' in t else t.split('_') for t in triplets]
                triplets = [t.split() for t in triplets]
                triplets = list(map(tuple, triplets))
            attrs1, attrs2, objs = zip(*triplets)
            return attrs1, attrs2, objs, triplets

        tr_attrs1, tr_attrs2, tr_objs, tr_triplets = parse_triplets(
            '%s/%s/train_triplets.txt' % (self.root, self.split))
        vl_attrs1, vl_attrs2, vl_objs, vl_triplets = parse_triplets(
            '%s/%s/val_triplets.txt' % (self.root, self.split))
        ts_attrs1, ts_attrs2, ts_objs, ts_triplets = parse_triplets(
            '%s/%s/test_triplets.txt' % (self.root, self.split))

        all_attrs1, all_attrs2, all_objs = sorted(
            list(set(tr_attrs1 + vl_attrs1 + ts_attrs1))), sorted(
                list(set(tr_attrs2 + vl_attrs2 + ts_attrs2))), sorted(
                    list(set(tr_objs + vl_objs + ts_objs)))
        all_triplets = sorted(list(set(tr_triplets + vl_triplets + ts_triplets)))

        return all_attrs1, all_attrs2, all_objs, all_triplets, tr_triplets, vl_triplets, ts_triplets

    def __getitem__(self, index):
        image, attr1, attr2, obj = self.data[index]
        img = self.loader(image)
        if self.transform is not None:
            img = self.transform(img)

        if self.phase == 'train':
            data = [
                img, self.attr1_idx[attr1], self.attr2_idx[attr2], self.obj2idx[obj], self.train_triplet_to_idx[(attr1, attr2, obj)]
            ]
        else:
            data = [
                img, self.attr1_idx[attr1], self.attr2_idx[attr2], self.obj2idx[obj], self.triplet2idx[(attr1, attr2, obj)]
            ]

        return data

    def __len__(self):
        return len(self.data)
    
    def __str__(self) -> str:
        n_seen_val = len(set(self.train_triplets) & set(self.val_triplets))
        n_seen_test = len(set(self.train_triplets) & set(self.test_triplets))
        descr_triplets = ' # train triplets : {:<7} | # val triplets : {:<7} ({:^5} seen) | # test triplets : {:<7} ({:^5} seen)'.format(
            len(self.train_triplets),
            len(self.val_triplets), n_seen_val,
            len(self.test_triplets), n_seen_test)

        _, attr1_val, attr2_val, obj_val = zip(*self.val_data)
        all_val_triplets = zip(attr1_val, attr2_val, obj_val)
        seen_triplets = set(self.train_triplets)
        n_seen_img_val = sum(p in seen_triplets for p in all_val_triplets)
        _, attr1_test, attr2_test, obj_test = zip(*self.test_data)
        all_test_triplets = zip(attr1_test, attr2_test, obj_test)
        n_seen_img_test = sum(p in seen_triplets for p in all_test_triplets)
        descr_n_img = ' # train images: {:<7} | # val images: {:<7} ({:^5} seen) | # test images: {:<7} ({:^5} seen)'.format(
            len(self.train_data),
            len(self.val_data), n_seen_img_val,
            len(self.test_data), n_seen_img_test)
        descr_ao = ' # attrs1 : {:<10} # attrs2 : {:<10} # objs : {:<10} # full triplets {:<10}'.format(
            len(self.attrs1), len(self.attrs2), len(self.objs), len(self.full_triplets))
        return descr_triplets + '\n' + descr_n_img + '\n' + descr_ao



class CompositionDatasetEmbeddings(CompositionDataset):
    '''
    Class for loading pre-computed embeddings of a compositional dataset.
    '''
    def __init__(
            self,
            root,
            phase,
            split='compositional-split-natural',
            open_world=False,
            model_architecture: str='ViT-L-14',
            model_pretraining: str='openai',
            normalize: bool=True,
            ):
        super().__init__(root, phase, split, open_world)
        loadfile_id = f"emb_{model_architecture}_{model_pretraining}.pt"
        self.text_embs_path = os.path.join(root, 'TEXT' + loadfile_id)
        self.image_embs_path = os.path.join(root, 'IMG' + loadfile_id)
        self.normalize = normalize

    def load_text_embs(self, triplets):
        '''
        Loads text embeddings for given triplets.
        '''
        text_embs_data = torch.load(self.text_embs_path, weights_only=False)  # All text embeddings
        triplet2idx = defaultdict(list)
        for i, triplet in enumerate(text_embs_data['triplets']):
            triplet2idx[triplet].append(i)

        indices = [i for triplet in triplets
                     for i in triplet2idx[triplet]]
        text_embs = text_embs_data['embeddings'][indices]

        if self.normalize:
            text_embs = F.normalize(text_embs, p=2, dim=-1)
        
        if len(triplets)==len(text_embs):
            return text_embs
        else:
            all_triplets = [text_embs_data['triplets'][i] for i in indices]
            return text_embs, all_triplets

    def load_all_image_embs(self):
        '''
        Loads all image embeddings for `self.phase`.
        '''
        image_embs_data = torch.load(self.image_embs_path, weights_only=False)  # All image embeddings
        image_id2idx = {id: i for i, id in enumerate(image_embs_data['image_ids'])}

        all_image_id, all_attrs1, all_attrs2, all_objs = zip(*self.data)
        all_triplets = list(zip(all_attrs1, all_attrs2, all_objs))
        indices = [image_id2idx[id] for id in all_image_id]
        image_embs = image_embs_data['embeddings'][indices]

        if self.normalize:
            image_embs = F.normalize(image_embs, p=2, dim=-1)

        return image_embs, all_triplets