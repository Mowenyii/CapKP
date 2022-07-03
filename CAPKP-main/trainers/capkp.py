from html import entities
import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

import pandas as pd
import numpy as np
import random
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

_tokenizer = _Tokenizer()

# food101
food101_item_idss=torch.tensor([407,2593,3120,3120,3120,713,1161,3226,1246,407,2593,3120,3120,3120,713,1161,3226,1246,407,2593,3120,3120,3120,713,1161,3226,1246,407,2593,3120,3120,3120,713,1161,3226,1246,407,2593,3120,3120,3120,713,1161,3226,1246,407,2593,3120,3120,3120,713,1161,3226,1246,
                        407,2593,3120,3120,3120,713,1161,3226,1246,407,2593,3120,3120,3120,713,1161,3226,1246,407,2593,3120,3120,3120,713,1161,3226,1246,407,2593,3120,3120,3120,713,1161,3226,1246,407,2593,3120,3120,3120,713,1161,3226,1246,3226,1246])

class Aggregator(torch.nn.Module):
    '''
    Aggregator class
    Mode in ['sum', 'concat', 'neighbor']
    '''   
    def __init__(self, batch_size, dim, aggregator):
        super(Aggregator, self).__init__()
        self.batch_size = batch_size
        self.dim = dim
        if aggregator == 'concat':
            self.weights = torch.nn.Linear(2 * dim, dim, bias=True)
        else:
            self.weights = torch.nn.Linear(dim, dim, bias=True)
        self.aggregator = aggregator
        
    def forward(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, act):
        batch_size = user_embeddings.size(0)
        if batch_size != self.batch_size:
            self.batch_size = batch_size
        neighbors_agg = self._mix_neighbor_vectors(neighbor_vectors, neighbor_relations, user_embeddings)
        
        if self.aggregator == 'sum':
            output = (self_vectors + neighbors_agg).view((-1, self.dim))
            
        elif self.aggregator == 'concat':
            output = torch.cat((self_vectors, neighbors_agg), dim=-1)
            output = output.view((-1, 2 * self.dim))
            
        else:
            output = neighbors_agg.view((-1, self.dim))
            
        output = self.weights(output)
        return act(output.view((self.batch_size, -1, self.dim)))
        
    def _mix_neighbor_vectors(self, neighbor_vectors, neighbor_relations, user_embeddings):
        '''
        This aims to aggregate neighbor vectors
        '''
        # [batch_size, 1, dim] -> [batch_size, 1, 1, dim]
        user_embeddings = user_embeddings.view((self.batch_size, 1, 1, self.dim))
        
        # [batch_size, -1, n_neighbor, dim] -> [batch_size, -1, n_neighbor]
        user_relation_scores = (user_embeddings * neighbor_relations).sum(dim = -1)
        user_relation_scores_normalized = F.softmax(user_relation_scores, dim = -1)
        
        # [batch_size, -1, n_neighbor] -> [batch_size, -1, n_neighbor, 1]
        user_relation_scores_normalized = user_relation_scores_normalized.unsqueeze(dim = -1)
        
        # [batch_size, -1, n_neighbor, 1] * [batch_size, -1, n_neighbor, dim] -> [batch_size, -1, dim]
        neighbors_aggregated = (user_relation_scores_normalized * neighbor_vectors).sum(dim = 2)
        
        return neighbors_aggregated

class DataLoader:
    '''
    Data Loader class which makes dataset for training / knowledge graph dictionary
    '''
    def __init__(self, data):
        self.cfg = {
            'wiki': {
                'item2id_path': '/home/hujie/clip/CAPKP-main/trainers/data/wiki/item_index2entity_id.txt',
                'kg_path': '/home/hujie/clip/CAPKP-main/trainers/data/wiki/kg.txt',
                'rating_path': '/home/hujie/clip/CAPKP-main/trainers/data/wiki/user_artists.dat',
                'rating_sep': '\t',
                'threshold': 0.0
            },
            'nell': {
                'item2id_path': '/home/hujie/clip/CAPKP-main/trainers/data/nell/item_index2entity_id.txt',
                'kg_path': '/home/hujie/clip/CAPKP-main/trainers/data/nell/kg.txt',
                'rating_path': '/home/hujie/clip/CAPKP-main/trainers/data/nell/user_artists.dat',
                'rating_sep': '\t',
                'threshold': 0.0
            }
        }
        self.data = data
        
        df_item2id = pd.read_csv(self.cfg[data]['item2id_path'], sep='\t', header=None, names=['item','id'])
        df_kg = pd.read_csv(self.cfg[data]['kg_path'], sep='\t', header=None, names=['head','relation','tail'])
        df_rating = pd.read_csv(self.cfg[data]['rating_path'], sep=self.cfg[data]['rating_sep'], names=['userID', 'itemID', 'rating'], skiprows=1)
        
        # df_rating['itemID'] and df_item2id['item'] both represents old entity ID
        df_rating = df_rating[df_rating['itemID'].isin(df_item2id['item'])]
        df_rating.reset_index(inplace=True, drop=True)
        
        self.df_item2id = df_item2id
        self.df_kg = df_kg
        self.df_rating = df_rating
        
        self.user_encoder = LabelEncoder()
        self.entity_encoder = LabelEncoder()
        self.relation_encoder = LabelEncoder()

        self._encoding()
        
    def _encoding(self):
        '''
        Fit each label encoder and encode knowledge graph
        '''
        self.user_encoder.fit(self.df_rating['userID'])
        # df_item2id['id'] and df_kg[['head', 'tail']] represents new entity ID
        self.entity_encoder.fit(pd.concat([self.df_item2id['id'], self.df_kg['head'], self.df_kg['tail']]))
        self.relation_encoder.fit(self.df_kg['relation'])
        
        # encode df_kg
        self.df_kg['head'] = self.entity_encoder.transform(self.df_kg['head'])
        self.df_kg['tail'] = self.entity_encoder.transform(self.df_kg['tail'])
        self.df_kg['relation'] = self.relation_encoder.transform(self.df_kg['relation'])+1

    def _build_dataset(self):
        '''
        Build dataset for training (rating data)
        It contains negative sampling process
        '''
        print('Build dataset dataframe ...', end=' ')
        # df_rating update
        df_dataset = pd.DataFrame()
        df_dataset['userID'] = self.user_encoder.transform(self.df_rating['userID'])
        
        # update to new id
        item2id_dict = dict(zip(self.df_item2id['item'], self.df_item2id['id']))
        self.df_rating['itemID'] = self.df_rating['itemID'].apply(lambda x: item2id_dict[x])
        df_dataset['itemID'] = self.entity_encoder.transform(self.df_rating['itemID'])
        df_dataset['label'] = self.df_rating['rating'].apply(lambda x: 0 if x < self.cfg[self.data]['threshold'] else 1)
        
        # negative sampling
        df_dataset = df_dataset[df_dataset['label']==1]
        # df_dataset requires columns to have new entity ID
        full_item_set = set(range(len(self.entity_encoder.classes_)))
        user_list = []
        item_list = []
        label_list = []
        for user, group in df_dataset.groupby(['userID']):
            item_set = set(group['itemID'])
            negative_set = full_item_set - item_set
            negative_sampled = random.sample(negative_set, len(item_set))
            user_list.extend([user] * len(negative_sampled))
            item_list.extend(negative_sampled)
            label_list.extend([0] * len(negative_sampled))
        negative = pd.DataFrame({'userID': user_list, 'itemID': item_list, 'label': label_list})
        df_dataset = pd.concat([df_dataset, negative])
        
        df_dataset = df_dataset.sample(frac=1, replace=False, random_state=999)
        df_dataset.reset_index(inplace=True, drop=True)
        print('Done')
        return df_dataset
        
        
    def _construct_kg(self):
        '''
        Construct knowledge graph
        Knowledge graph is dictionary form
        'head': [(relation, tail), ...]
        '''
        print('Construct knowledge graph ...', end=' ')
        kg = dict()
        for i in range(len(self.df_kg)):
            head = self.df_kg.iloc[i]['head']
            relation = self.df_kg.iloc[i]['relation']
            tail = self.df_kg.iloc[i]['tail']
            if head in kg:
                kg[head].append((relation, tail))
            else:
                kg[head] = [(relation, tail)]
            if tail in kg:
                kg[tail].append((relation, head))
            else:
                kg[tail] = [(relation, head)]
        print('Done')
        return kg
        
    def load_dataset(self):
        return self._build_dataset()

    def load_kg(self):
        return self._construct_kg()
    
    def get_encoders(self):
        return (self.user_encoder, self.entity_encoder, self.relation_encoder)
    
    def get_num(self):
        return (len(self.user_encoder.classes_), len(self.entity_encoder.classes_), len(self.relation_encoder.classes_))

class KGCNDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        user_id = np.array(self.df.iloc[idx]['userID'])
        item_id = np.array(self.df.iloc[idx]['itemID'])
        label = np.array(self.df.iloc[idx]['label'], dtype=np.float32)
        return user_id, item_id, label  
    



class KGCN(torch.nn.Module):
    def __init__(self,cfg, num_user, num_ent, num_rel, kg, device):
        super(KGCN, self).__init__()
        self.num_user = num_user
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.n_iter = 1
        self.batch_size = cfg.TRAINER.CAPKP.NCLASS
        self.dim = 512
        self.n_neighbor = 8
        self.kg = kg
        self.Linear = torch.nn.Linear(self.batch_size, cfg.TRAINER.CAPKP.N_CTX, bias=True)
        self.device = device
        self.aggregator = Aggregator(self.batch_size, self.dim, 'sum')
        self._gen_adj()
        self.usr = torch.nn.Embedding(num_user, self.dim)
        self.ent = torch.nn.Embedding(num_ent, self.dim)
        self.rel = torch.nn.Embedding(num_rel+1, self.dim)
        self.isLinear = False # No linear layer
        if (cfg.TRAINER.CAPKP.N_CTX == 16 or cfg.TRAINER.CAPKP.N_CTX == 4) and cfg.TRAINER.CAPKP.CSC == False:
            # SHR, [102,512]---linear-->[16,512]
            self.isLinear = True
        
    def _gen_adj(self):
        '''
        Generate adjacency matrix for entities and relations
        Only cares about fixed number of samples
        '''
        self.adj_ent = torch.empty(self.num_ent, self.n_neighbor, dtype=torch.long)
        self.adj_rel = torch.empty(self.num_ent, self.n_neighbor, dtype=torch.long)
        
        for e in self.kg:
            if len(self.kg[e]) >= self.n_neighbor:
                neighbors = random.sample(self.kg[e], self.n_neighbor)
            else:
                neighbors = random.choices(self.kg[e], k=self.n_neighbor)
                
            self.adj_ent[e] = torch.LongTensor([ent for _, ent in neighbors])
            self.adj_rel[e] = torch.LongTensor([rel for rel, _ in neighbors])
        
    def forward(self, u, v,r=None,entities=None,relations=None):
        '''
        input: u, v are batch sized indices for users and items
        u: [batch_size]
        v: [batch_size]
        '''
        batch_size = u.size(0)
        
        if batch_size != self.batch_size:
            self.batch_size = batch_size
        
        # change to [batch_size, 1]
        u = u.view((-1, 1))
        v = v.view((-1, 1))
        
        # [batch_size, dim]
        user_embeddings = self.usr(u).squeeze(dim = 1)
        if entities==None and relations==None:
            entities, relations = self._get_neighbors(v)

        # Prune each type of edge in the whole graph according to r.
        if r!=None:
            for i in range(self.batch_size):
                for j in range(8):
                    if relations[0][i][j]==r:
                        entities[1][i][j]=entities[0][i]# Change the corresponding position to itself
                        relations[0][i][j]=0            # The relationship points to itself
                        
        item_embeddings = self._aggregate(user_embeddings, entities, relations)
        if self.isLinear==True:# SHR, [102,512]---linear-->[16,512]
            c=item_embeddings.t()
            c=self.Linear(c)
            c=c.t()
        else: # SPE, [102,512]-->[102,1,512]
            a=torch.unsqueeze(item_embeddings, 2)
            c=a.permute(0,2,1)
        scores = (user_embeddings * item_embeddings).sum(dim = 1)

        return torch.sigmoid(scores),c, entities, relations
    
    def _get_neighbors(self, v):
        '''
        v is batch sized indices for items
        v: [batch_size, 1]
        '''
        entities = [v]
        relations = []
        
        for h in range(self.n_iter):
            neighbor_entities = torch.LongTensor(self.adj_ent[entities[h]]).view((self.batch_size, -1)).to(self.device)
            neighbor_relations = torch.LongTensor(self.adj_rel[entities[h]]).view((self.batch_size, -1)).to(self.device)
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
            
        return entities, relations
    
    def _aggregate(self, user_embeddings, entities, relations):
        '''
        Make item embeddings by aggregating neighbor vectors
        '''
        entity_vectors = [self.ent(entity) for entity in entities]
        relation_vectors = [self.rel(relation) for relation in relations]
        
        for i in range(self.n_iter):
            if i == self.n_iter - 1:
                act = torch.tanh
            else:
                act = torch.sigmoid
            
            entity_vectors_next_iter = []
            for hop in range(self.n_iter - i):
                vector = self.aggregator(
                    self_vectors=entity_vectors[hop],
                    neighbor_vectors=entity_vectors[hop + 1].view((self.batch_size, -1, self.n_neighbor, self.dim)),
                    neighbor_relations=relation_vectors[hop].view((self.batch_size, -1, self.n_neighbor, self.dim)),
                    user_embeddings=user_embeddings,
                    act=act)
                
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        return entity_vectors[0].view((self.batch_size, self.dim))
    

    

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        # load KG
        data_loader = DataLoader('wiki')# or nell-ZS
        kg = data_loader.load_kg()
        df_dataset = data_loader.load_dataset()
        x_train, x_test, y_train, y_test = train_test_split(df_dataset, df_dataset['label'], test_size=1 - 0.8, shuffle=False, random_state=999)
        train_dataset = KGCNDataset(x_train)
        num_user, num_entity, num_relation = data_loader.get_num()
        
        self.KGCN = KGCN(cfg,num_user, num_entity, num_relation, kg, torch.device('cuda'))
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.TRAINER.CAPKP.NCLASS)
        self.lam=cfg.TRAINER.CAPKP.KGCN # lambda

        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.CAPKP.N_CTX
        ctx_init = cfg.TRAINER.CAPKP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        
        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.CAPKP.CSC: 
                print("Initializing class-specific contexts") #SPE
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context") # SHR
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.dataset=cfg.TRAINER.CAPKP.DATASET
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.CAPKP.CLASS_TOKEN_POSITION
        if cfg.TRAINER.CAPKP.DATASET == "food101":
            self.item_idss=food101_item_idss

        self.register_buffer("KGCN_entities_0", self.KGCN._get_neighbors(self.item_idss)[0][0])  # entities[0]
        self.register_buffer("KGCN_entities_1", self.KGCN._get_neighbors(self.item_idss)[0][1])  # entities[1]
        self.register_buffer("KGCN_relations_0",self.KGCN._get_neighbors(self.item_idss)[1][0])

    def forward(self,r=None,entities_0=None,entities_1=None,relations_0=None):
        if entities_0==None and r==None:# phase of training
            for i, (user_ids, item_ids, labels) in enumerate(self.train_loader):
                user_ids, item_ids= user_ids.cuda(), self.item_idss.cuda()
                outputs = self.KGCN(user_ids, item_ids)
                outputs=outputs[1]
                if i>=0:
                    break
        elif entities_0==None and r!=None:# Phase of testing. Pass r into KGCN and prune each type of edge in the whole graph according to r.
            for i, (user_ids, item_ids, labels) in enumerate(self.train_loader):
                user_ids, item_ids= user_ids.cuda(), self.item_idss.cuda()
                outputs = self.KGCN(user_ids, item_ids,r)
                outputs=outputs[1]
                if i>=0:
                    break
        else:# Prune each type of edge in each subgraph
            entities_0=entities_0.unsqueeze(1)# Dimension transformation [101]->[101,1]
            entities_list=[entities_0,entities_1]
            relations_list=[relations_0]
            for i, (user_ids, item_ids, labels) in enumerate(self.train_loader):
                user_ids, item_ids= user_ids.cuda(), self.item_idss.cuda()
                entities_list[0]
                outputs = self.KGCN(user_ids, item_ids,None,entities_list,relations_list)
                outputs=outputs[1]
                if i>=0:
                    break
        
        ctx = self.ctx
        # 𝜇 + 𝜆 · 𝜑 ([𝒀]i)
        ctx = (ctx +  self.lam *outputs)

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image,r=None,entities_0=None,entities_1=None,relations_0=None):
        image_features = self.image_encoder(image.type(self.dtype))
        if entities==None and r==None:# phase of training
            prompts = self.prompt_learner().half()
        elif r!=None: # phase of testing, pass r into KGCN
            prompts = self.prompt_learner(r,None,None,None).half()
        else:# Prune each type of edge in each subgraph
            prompts = self.prompt_learner(None,entities_0,entities_1,relations_0).half()
        tokenized_prompts = self.tokenized_prompts

        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


@TRAINER_REGISTRY.register()
class CAPKP(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.CAPKP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.CAPKP.PREC == "fp32" or cfg.TRAINER.CAPKP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner.ctx" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.CAPKP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count})")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.CAPKP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()

        else:
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)
            

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)
            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))
            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)

        return state_dict["KGCN_entities_0"],state_dict["KGCN_entities_1"],state_dict["KGCN_relations_0"]
       
    def getER(self):
        entities_0=self.model.prompt_learner.KGCN._get_neighbors(self.model.prompt_learner.item_idss)[0][0].to(self.device)
        entities_1=self.model.prompt_learner.KGCN._get_neighbors(self.model.prompt_learner.item_idss)[0][1].to(self.device)
        relations_0=self.model.prompt_learner.KGCN._get_neighbors(self.model.prompt_learner.item_idss)[1][0].to(self.device)
        return entities_0,entities_1,relations_0